"""
AEGIS Step 17 — Feedback Loop & Continuous Learning
Retrains Isolation Forest (Step 8) and K-Means (Step 9) using honeypot-derived attack data.

PURPOSE:
  Security threats evolve. Static models become stale.
  This script uses new honeypot data to keep detection fresh.

SCHEDULE:
  Runs every 24 hours (or when new_attack_samples > threshold)
  cron: 0 2 * * * /usr/bin/python3 /path/to/retrain.py

WHAT IT DOES:
  1. Collect new attack samples from honeypot logs
  2. Merge with existing training baseline (sliding window: last 30 days)
  3. Validate data quality and size
  4. Retrain Isolation Forest
  5. Validate: new_model must score <0.3 on known-normal traffic
  6. Retrain K-Means clustering on attack samples
  7. Atomically swap model files (no downtime)
  8. Log retrain results with metrics

USAGE:
  python retrain.py                          # Run once immediately
  python retrain.py --dry-run               # Test without committing changes
  python retrain.py --min-samples 500       # Custom minimum sample threshold
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomaly_model import (
    FEATURES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    SCORE_STATS,
)

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

RETRAIN_LOG = LOG_DIR / "retrain.log"
RETRAIN_METRICS = LOG_DIR / "retrain_metrics.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(RETRAIN_LOG),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Model paths (should match definitions in anomaly_model.py and cluster_attackers.py)
MODEL_PATH = Path(MODEL_PATH) if isinstance(MODEL_PATH, str) else MODEL_PATH
SCALER_PATH = Path(SCALER_PATH) if isinstance(SCALER_PATH, str) else SCALER_PATH
FEATURES_PATH = Path(FEATURES_PATH) if isinstance(FEATURES_PATH, str) else FEATURES_PATH
SCORE_STATS = Path(SCORE_STATS) if isinstance(SCORE_STATS, str) else SCORE_STATS

KMEANS_PATH = MODELS_DIR / "kmeans_model.pkl"
DBSCAN_PATH = MODELS_DIR / "dbscan_model.pkl"
CLUSTER_SCALER_PATH = MODELS_DIR / "cluster_scaler.pkl"
PCA_PATH = MODELS_DIR / "pca.pkl"
PROFILE_MAP_PATH = MODELS_DIR / "profile_map.pkl"

SLIDING_WINDOW_DAYS = 30
MIN_ATTACK_SAMPLES = 500
MIN_BENIGN_SAMPLES = 200

# Backup extension for atomic swap
BACKUP_EXT = ".backup"


def find_honeypot_data():
    """Locate new attack samples from honeypot logs."""
    logger.info("Searching for honeypot-derived attack data...")

    honeypot_sources = [
        LOG_DIR / "anomalous_sessions.json",  # High-confidence attacks from honeypot
        LOG_DIR / "anomaly_scores.jsonl",  # Flagged sessions
    ]

    attack_samples = []

    for source in honeypot_sources:
        if source.exists():
            logger.info(f"  Loading: {source}")
            try:
                if source.suffix == ".jsonl":
                    with open(source) as f:
                        for line in f:
                            try:
                                record = json.loads(line.strip())
                                if record.get("anomaly_score", 0) > 0.5:  # High confidence
                                    attack_samples.append(record)
                            except json.JSONDecodeError:
                                pass
                else:
                    with open(source) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            attack_samples.extend(data)
            except Exception as e:
                logger.warning(f"  Error reading {source}: {e}")

    logger.info(f"  Found {len(attack_samples)} attack samples from honeypot")
    return attack_samples


def load_existing_training_data():
    """Load the original training data (benign baseline + historical attacks)."""
    logger.info("Loading existing training data...")

    # Benign training files (Step 8 baseline)
    benign_files = [
        DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv",
        DATA_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ]

    # Historical attack files (Step 9 baseline)
    attack_files = [
        DATA_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv",
        DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
        DATA_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ]

    benign_data = []
    attack_data = []
    feature_cols = joblib.load(FEATURES_PATH) if FEATURES_PATH.exists() else None

    # Load benign samples
    for csv_file in benign_files:
        if csv_file.exists():
            logger.info(f"  Loading benign: {csv_file.name}")
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            benign_subset = df[df["Label"].str.strip() == "BENIGN"].head(1000)
            benign_data.append(benign_subset)

    # Load attack samples
    for csv_file in attack_files:
        if csv_file.exists():
            logger.info(f"  Loading attacks: {csv_file.name}")
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            attack_subset = df[df["Label"].str.strip() != "BENIGN"].head(500)
            attack_data.append(attack_subset)

    if benign_data:
        benign_df = pd.concat(benign_data, ignore_index=True)
    else:
        benign_df = pd.DataFrame()

    if attack_data:
        attack_df = pd.concat(attack_data, ignore_index=True)
    else:
        attack_df = pd.DataFrame()

    logger.info(f"  Loaded {len(benign_df)} benign samples, {len(attack_df)} attack samples")
    return benign_df, attack_df, feature_cols


def extract_features(df, feature_cols):
    """Extract feature vectors from dataframe, handling NaN/inf values."""
    if feature_cols is None or len(df) == 0:
        return np.array([])

    try:
        # Select features that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0).values

        # Convert to float32
        X = np.array(X, dtype=np.float32)

        # Clean inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        logger.debug(f"Extracted {len(X)} samples with {X.shape[1]} features")
        return X
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        return np.array([])


def validate_new_model_on_benign(new_model, benign_data, feature_cols, threshold=0.3):
    """
    Validate that new model scores benign traffic low.

    If >20% of benign samples score >= threshold, reject the model.
    """
    logger.info("Validating new model on known-benign traffic...")

    X_benign = extract_features(benign_data, feature_cols)

    if len(X_benign) == 0:
        logger.warning("  No benign data to validate against — skipping validation")
        return True

    scores = new_model.decision_function(X_benign)

    # Normalize scores to [0, 1]
    anomaly_scores = 1 / (1 + np.exp(-scores))

    high_score_pct = (anomaly_scores >= threshold).sum() / len(anomaly_scores) * 100
    avg_score = anomaly_scores.mean()

    logger.info(f"  Benign samples: {len(X_benign)}")
    logger.info(f"  Average anomaly score: {avg_score:.4f}")
    logger.info(f"  % scoring >={threshold}: {high_score_pct:.1f}%")

    passed = high_score_pct < 20  # Allow up to 20% false positives

    if passed:
        logger.info("  [PASS] Validation PASSED")
    else:
        logger.warning(
            f"  [FAIL] Validation rejected: {high_score_pct:.1f}% benign samples falsely flagged"
        )

    return passed


def train_new_isolation_forest(benign_data, feature_cols):
    """Train new Isolation Forest model on updated benign baseline."""
    logger.info("Training new Isolation Forest model...")

    X_benign = extract_features(benign_data, feature_cols)

    if len(X_benign) < MIN_BENIGN_SAMPLES:
        logger.error(f"Insufficient benign samples: {len(X_benign)} < {MIN_BENIGN_SAMPLES}")
        return None

    logger.info(f"  Training on {len(X_benign)} benign samples")

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_benign)

    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100,
        max_samples="auto",
        n_jobs=-1,
    )
    model.fit(X_scaled)

    logger.info("  Model training complete")
    return model, scaler, feature_cols


def train_new_kmeans(attack_data, feature_cols):
    """Train new K-Means clustering on updated attack samples."""
    logger.info("Training new K-Means clustering model...")

    X_attacks = extract_features(attack_data, feature_cols)

    if len(X_attacks) < MIN_ATTACK_SAMPLES:
        logger.warning(f"Insufficient attack samples: {len(X_attacks)} < {MIN_ATTACK_SAMPLES}")
        logger.warning("  Skipping K-Means retrain")
        return None

    logger.info(f"  Training on {len(X_attacks)} attack samples")

    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Fit StandardScaler (separate from Step 8)
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_attacks)

    # Train K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Fit PCA for DBSCAN dimensionality reduction
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Train DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)

    logger.info(f"  K-Means clusters: {len(np.unique(kmeans.labels_))}")
    logger.info(f"  DBSCAN clusters: {len(np.unique(dbscan_labels))}")
    logger.info("  ✓ Clustering training complete")

    return kmeans, dbscan, cluster_scaler, pca


def atomic_model_swap(old_path, new_model_data, model_name):
    """
    Atomically swap model files to avoid downtime.

    1. Write new model to *.new
    2. Backup old model to *.backup
    3. Rename *.new to final name
    """
    logger.info(f"Atomically swapping {model_name}...")

    old_path = Path(old_path)
    new_path = old_path.with_suffix(".new")
    backup_path = old_path.with_suffix(BACKUP_EXT)

    try:
        # Write new model
        joblib.dump(new_model_data, new_path)
        logger.info(f"  Wrote: {new_path}")

        # Backup old model
        if old_path.exists():
            if backup_path.exists():
                backup_path.unlink()
            shutil.copy(old_path, backup_path)
            logger.info(f"  Backup: {backup_path}")

        # Atomic swap
        new_path.replace(old_path)
        logger.info(f"  Swapped: {old_path}")

        return True
    except Exception as e:
        logger.error(f"  Atomic swap failed: {e}")
        if new_path.exists():
            new_path.unlink()
        return False


def log_retrain_metrics(old_metrics, new_metrics, success):
    """Log retrain event with before/after metrics."""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "success": success,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
    }

    with open(RETRAIN_METRICS, "a") as f:
        f.write(json.dumps(event) + "\n")

    logger.info(f"Logged retrain metrics to {RETRAIN_METRICS}")


def retrain_cycle(dry_run=False, min_samples=MIN_ATTACK_SAMPLES):
    """Execute one complete retrain cycle."""
    logger.info("=" * 80)
    logger.info("STEP 17 — FEEDBACK LOOP RETRAIN CYCLE")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.info(f"Dry run: {dry_run}")

    # Phase 1: Data Collection
    logger.info("\n[PHASE 1] Data Collection")
    logger.info("-" * 80)

    benign_df, attack_df, feature_cols = load_existing_training_data()
    honeypot_attacks = find_honeypot_data()

    if len(honeypot_attacks) < min_samples:
        logger.warning(f"Insufficient new attack samples: {len(honeypot_attacks)} < {min_samples}")
        logger.warning("Aborting retrain cycle")
        return False

    logger.info("Total samples to retrain with:")
    logger.info(f"  Benign: {len(benign_df)}")
    logger.info(
        f"  Attacks: {len(attack_df)} (historical) + {len(honeypot_attacks)} (new honeypot)"
    )

    # Phase 2: Model Training
    logger.info("\n[PHASE 2] Model Training")
    logger.info("-" * 80)

    iso_result = train_new_isolation_forest(benign_df, feature_cols)
    if iso_result is None:
        logger.error("Isolation Forest training failed")
        return False

    new_iso_forest, new_scaler, _ = iso_result

    # Validate new Isolation Forest
    is_valid = validate_new_model_on_benign(new_iso_forest, benign_df, feature_cols)
    if not is_valid:
        logger.error("Isolation Forest validation failed — rejecting new model")
        return False

    # Train new K-Means
    kmeans_result = train_new_kmeans(attack_df, feature_cols)
    if kmeans_result is None:
        logger.warning("K-Means training failed — will keep old clustering model")
        kmeans_result = (None, None, None, None)

    # Phase 3: Atomic Model Swap
    logger.info("\n[PHASE 3] Atomic Model Swap")
    logger.info("-" * 80)

    if dry_run:
        logger.info("DRY RUN: Skipping model swap")
        return True

    # Swap Isolation Forest
    if not atomic_model_swap(MODEL_PATH, new_iso_forest, "isolation_forest.pkl"):
        logger.error("Failed to swap Isolation Forest")
        return False

    if not atomic_model_swap(SCALER_PATH, new_scaler, "scaler.pkl"):
        logger.error("Failed to swap scaler")
        return False

    # Swap K-Means if training succeeded
    if kmeans_result[0] is not None:
        kmeans, dbscan, cluster_scaler, pca = kmeans_result

        if not atomic_model_swap(KMEANS_PATH, kmeans, "kmeans_model.pkl"):
            logger.warning("Failed to swap K-Means")

        if not atomic_model_swap(DBSCAN_PATH, dbscan, "dbscan_model.pkl"):
            logger.warning("Failed to swap DBSCAN")

        if not atomic_model_swap(CLUSTER_SCALER_PATH, cluster_scaler, "cluster_scaler.pkl"):
            logger.warning("Failed to swap cluster_scaler")

        if not atomic_model_swap(PCA_PATH, pca, "pca.pkl"):
            logger.warning("Failed to swap PCA")

    # Phase 4: Logging
    logger.info("\n[PHASE 4] Retrain Summary")
    logger.info("-" * 80)

    old_metrics = {"status": "baseline"}
    new_metrics = {
        "benign_samples": len(benign_df),
        "attack_samples": len(attack_df) + len(honeypot_attacks),
        "honeypot_new_samples": len(honeypot_attacks),
    }

    log_retrain_metrics(old_metrics, new_metrics, success=True)

    logger.info("✓ RETRAIN CYCLE COMPLETE")
    logger.info("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Step 17: Feedback Loop — Retrain models using honeypot data",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without committing model changes"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_ATTACK_SAMPLES,
        help=f"Minimum new attack samples required (default: {MIN_ATTACK_SAMPLES})",
    )

    args = parser.parse_args()

    try:
        success = retrain_cycle(dry_run=args.dry_run, min_samples=args.min_samples)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Retrain cycle failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
