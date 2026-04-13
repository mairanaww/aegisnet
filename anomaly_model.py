"""
AEGIS - Step 8: Anomaly Detection Model (Isolation Forest)
Person 1 - AI Core

Exports: score_session(feature_vector) -> float anomaly_score in [0, 1]

Score interpretation:
    < 0.3   → LOW    (normal traffic)
    0.3-0.7 → MEDIUM (suspicious)
    > 0.7   → HIGH   (likely attack) → policy engine triggers isolation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Logging setup 
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_model.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths 
MODEL_PATH     = "models/isolation_forest.pkl"
SCALER_PATH    = "models/scaler.pkl"
FEATURES_PATH  = "models/feature_columns.pkl"
SCORE_LOG      = "logs/anomaly_scores.jsonl"
SCORE_STATS    = "models/score_stats.pkl"
CALIBRATOR_PATH = "models/score_calibrator.pkl"
MAX_BENIGN_PER_FILE = 200_000
CALIBRATION_SKIP_HEAD = 2_000
CALIBRATION_MAX_PER_CLASS_PER_FILE = 20_000

# In-process artifact cache avoids repeated disk loads when scoring many rows.
_MODEL_CACHE = None
_SCALER_CACHE = None
_FEATURES_CACHE = None
_SCORE_STATS_CACHE = None
_CALIBRATOR_CACHE = None
_ARTIFACT_MTIMES = {}

# ── Columns to DROP (non-numeric / identifier columns) 
DROP_COLS = {
    "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "Label", " Label",
}

CALIBRATION_FILES = [
    os.path.join("data", "Monday-WorkingHours.pcap_ISCX.csv"),
    os.path.join("data", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    os.path.join("data", "Wednesday-workingHours.pcap_ISCX.csv"),
    os.path.join("data", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    os.path.join("data", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    os.path.join("data", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    os.path.join("data", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join("data", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Apply signed log1p transformation: preserves sign, compresses magnitude."""
    return np.sign(x) * np.log1p(np.abs(x))


def _clean_df(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Select feature columns, coerce to numeric, handle inf/NaN."""
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X.values


def _raw_to_score(raw: float, lo: float, hi: float) -> float:
    """
    Map IsolationForest.decision_function() output to [0, 1] using percentile bounds.

    In Isolation Forest:
      - NEGATIVE raw = anomaly (easy to isolate)
      - POSITIVE raw = normal (hard to isolate)
    
    We want output [0, 1] where:
      - 0 = normal (positive raw, within percentiles)
      - 1 = anomalous (negative raw, outliers)
    
    Strategy: invert the raw score so positive→low anomaly, negative→high anomaly
    """
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    # Invert: negative (anomaly) becomes positive, positive (normal) becomes negative
    inverted = -raw
    # Normalize using percentiles: values > hi map to 1, values < lo map to 0
    score = np.clip((inverted - lo) / denom, 0.0, 1.0)
    return round(float(score), 6)


def _classify_risk(score: float) -> str:
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    return "HIGH"


def _log_score(anomaly_score: float, extra: dict = None):
    """Append score entry to JSONL log (used by Step 17 feedback loop)."""
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp":     datetime.now().isoformat(),
        "anomaly_score": anomaly_score,
        "risk_level":    _classify_risk(anomaly_score),
    }
    if extra:
        entry.update(extra)
    with open(SCORE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_artifacts():
    """Load model artifacts once and refresh cache if files changed on disk."""
    global _MODEL_CACHE, _SCALER_CACHE, _FEATURES_CACHE, _SCORE_STATS_CACHE
    global _CALIBRATOR_CACHE, _ARTIFACT_MTIMES

    paths = [MODEL_PATH, SCALER_PATH, FEATURES_PATH, SCORE_STATS]
    if os.path.exists(CALIBRATOR_PATH):
        paths.append(CALIBRATOR_PATH)
    mtimes = {p: os.path.getmtime(p) for p in paths}

    if _MODEL_CACHE is None or _ARTIFACT_MTIMES != mtimes:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        _SCALER_CACHE = joblib.load(SCALER_PATH)
        _FEATURES_CACHE = joblib.load(FEATURES_PATH)
        _SCORE_STATS_CACHE = joblib.load(SCORE_STATS)
        _CALIBRATOR_CACHE = joblib.load(CALIBRATOR_PATH) if os.path.exists(CALIBRATOR_PATH) else None
        _ARTIFACT_MTIMES = mtimes

    return _MODEL_CACHE, _SCALER_CACHE, _FEATURES_CACHE, _SCORE_STATS_CACHE, _CALIBRATOR_CACHE


def _build_calibration_dataset(feature_cols: list):
    """Build labeled dataset from CICIDS files for score calibration."""
    X_parts = []
    y_parts = []

    for csv_path in CALIBRATION_FILES:
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        except Exception as exc:
            logger.warning(f"Calibration skip {csv_path}: {exc}")
            continue

        df.columns = df.columns.str.strip()
        if "Label" not in df.columns:
            logger.warning(f"Calibration skip {csv_path}: Label column missing")
            continue

        if len(df) > CALIBRATION_SKIP_HEAD:
            df = df.iloc[CALIBRATION_SKIP_HEAD:].copy()

        benign = df[df["Label"].str.strip() == "BENIGN"]
        attack = df[df["Label"].str.strip() != "BENIGN"]

        if len(benign) == 0 and len(attack) == 0:
            continue

        if len(benign) > CALIBRATION_MAX_PER_CLASS_PER_FILE:
            benign = benign.sample(n=CALIBRATION_MAX_PER_CLASS_PER_FILE, random_state=42)
        if len(attack) > CALIBRATION_MAX_PER_CLASS_PER_FILE:
            attack = attack.sample(n=CALIBRATION_MAX_PER_CLASS_PER_FILE, random_state=42)

        frame = pd.concat([benign, attack], ignore_index=True)
        if frame.empty:
            continue

        X_part = _clean_df(frame, feature_cols)
        y_part = (frame["Label"].str.strip() != "BENIGN").astype(int).values

        X_parts.append(X_part)
        y_parts.append(y_part)
        logger.info(
            f"Calibration data {os.path.basename(csv_path)}: "
            f"benign={int((y_part == 0).sum()):,}, attack={int((y_part == 1).sum()):,}"
        )

    if not X_parts:
        return None, None

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    n_benign = int((y_all == 0).sum())
    n_attack = int((y_all == 1).sum())
    logger.info(f"Calibration aggregate before balancing: benign={n_benign:,}, attack={n_attack:,}")

    if n_benign == 0 or n_attack == 0:
        return None, None

    # Balance classes for stable threshold-free probability output.
    minority = min(n_benign, n_attack)
    benign_idx = np.where(y_all == 0)[0]
    attack_idx = np.where(y_all == 1)[0]

    rng = np.random.default_rng(42)
    benign_keep = rng.choice(benign_idx, size=minority, replace=False)
    attack_keep = rng.choice(attack_idx, size=minority, replace=False)
    keep_idx = np.concatenate([benign_keep, attack_keep])
    rng.shuffle(keep_idx)

    X_bal = X_all[keep_idx]
    y_bal = y_all[keep_idx]
    logger.info(f"Calibration aggregate after balancing: samples={len(y_bal):,} (per class={minority:,})")
    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def train_model(csv_paths) -> None:
    """
    Train Isolation Forest on BENIGN rows only and persist artifacts.

    Args:
        csv_paths: str (single file) | list of str (multiple files)
                   All BENIGN rows from all files are combined for training.
    """
    # Handle both single file and list of files
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("data",   exist_ok=True)

    # ── Load and combine BENIGN rows from all files ─────────────────────────
    dfs = []
    total_benign = 0
    total_benign_used = 0
    label_col = "Label"
    
    for csv_path in csv_paths:
        logger.info(f"Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Filter BENIGN only
        df_benign = df[df[label_col].str.strip() == "BENIGN"]
        benign_count = len(df_benign)
        total_benign += benign_count

        # Keep training size stable while preserving day diversity.
        if benign_count > MAX_BENIGN_PER_FILE:
            df_benign = df_benign.sample(n=MAX_BENIGN_PER_FILE, random_state=42)
            logger.info(f"  BENIGN rows: {benign_count:,} (sampled to {len(df_benign):,})")
        else:
            logger.info(f"  BENIGN rows: {benign_count:,}")

        total_benign_used += len(df_benign)
        
        dfs.append(df_benign)
    
    # Combine all BENIGN rows
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total BENIGN rows found: {total_benign:,}")
    logger.info(f"Total BENIGN rows used:  {total_benign_used:,}")

    # ── Select feature columns ─────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    logger.info(f"Feature count: {len(feature_cols)}")

    X_raw = _clean_df(df, feature_cols)
    X = X_raw

    # ── Save normal_traffic.csv for reference ──────────────────────────────
    pd.DataFrame(X_raw, columns=feature_cols).to_csv("data/normal_traffic.csv", index=False)
    logger.info("Saved -> data/normal_traffic.csv")

    # ── Scale with StandardScaler ───────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train with conservative contamination to protect precision ───────────
    logger.info("Training IsolationForest (n_estimators=1200, contamination=0.01)...")
    model = IsolationForest(
        n_estimators=1200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # ── Compute score statistics for robust normalization ──────────────────
    # Since we invert the raw scores in _raw_to_score, compute percentiles on inverted scores
    raw_scores = model.decision_function(X_scaled)
    inverted_scores = -raw_scores  # Invert: anomalies → positive, normal → negative→positive
    lo = float(np.percentile(inverted_scores, 5))
    hi = float(np.percentile(inverted_scores, 95))
    score_stats = {
        "lo": lo,
        "hi": hi,
        "transform": "identity",
        "scaler": "standard",
        "contamination": 0.01,
        "blend": {"iforest": 1.0, "calibrator": 0.0},
    }
    logger.info(f"Score percentiles (inverted): 5%={lo:.6f}, 95%={hi:.6f}")

    # ── Train supervised score calibrator (semi-supervised blend) ───────────
    calibrator = None
    X_cal, y_cal = _build_calibration_dataset(feature_cols)
    if X_cal is not None and y_cal is not None:
        logger.info("Training score calibrator (RandomForestClassifier)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_cal,
            y_cal,
            test_size=0.20,
            random_state=42,
            stratify=y_cal,
        )

        calibrator = RandomForestClassifier(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        calibrator.fit(X_train, y_train)

        val_prob = calibrator.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        cal_precision = float(precision_score(y_val, val_pred, zero_division=0))
        cal_recall = float(recall_score(y_val, val_pred, zero_division=0))
        cal_f1 = float(f1_score(y_val, val_pred, zero_division=0))

        logger.info(
            "Calibrator validation: "
            f"precision={cal_precision:.3f}, recall={cal_recall:.3f}, f1={cal_f1:.3f}"
        )

        score_stats["calibrator"] = {
            "validation_precision": round(cal_precision, 4),
            "validation_recall": round(cal_recall, 4),
            "validation_f1": round(cal_f1, 4),
            "samples": int(len(y_cal)),
        }
        # Heavily weight calibrator when it has strong validation quality.
        score_stats["blend"] = {"iforest": 0.25, "calibrator": 0.75}
    else:
        logger.warning("Calibration dataset unavailable; using pure IsolationForest scoring")

    # ── Persist 
    joblib.dump(model,        MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(score_stats,  SCORE_STATS)
    if calibrator is not None:
        joblib.dump(calibrator, CALIBRATOR_PATH)
    elif os.path.exists(CALIBRATOR_PATH):
        os.remove(CALIBRATOR_PATH)

    logger.info(f"Model saved    -> {MODEL_PATH}")
    logger.info(f"Scaler saved   -> {SCALER_PATH}")
    logger.info(f"Features saved -> {FEATURES_PATH}")
    logger.info(f"Score stats    -> {SCORE_STATS}")
    if calibrator is not None:
        logger.info(f"Calibrator     -> {CALIBRATOR_PATH}")
    logger.info("Training complete [OK]")

    # Ensure subsequent score calls in this process use fresh artifacts.
    global _MODEL_CACHE, _SCALER_CACHE, _FEATURES_CACHE, _SCORE_STATS_CACHE
    global _CALIBRATOR_CACHE, _ARTIFACT_MTIMES
    _MODEL_CACHE = None
    _SCALER_CACHE = None
    _FEATURES_CACHE = None
    _SCORE_STATS_CACHE = None
    _CALIBRATOR_CACHE = None
    _ARTIFACT_MTIMES = {}


def score_session(feature_vector, log_score: bool = True) -> float:
    """
    Score one session's feature vector and return anomaly_score in [0, 1].

    Args:
        feature_vector: dict  {"col_name": value, ...}
                      | list  [val1, val2, ...]   (same order as training features)
                      | pd.Series

    Returns:
        float: anomaly_score
               0.0 = normal traffic
               1.0 = highly anomalous / likely attack

    Raises:
        FileNotFoundError: if model has not been trained yet (run train.py first)
    """
    # ── Load artifacts ─────────────────────────────────────────────────────
    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH, SCORE_STATS):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run train.py first to train the model."
            )

    model, scaler, feature_cols, score_stats, calibrator = _load_artifacts()

    # ── Build DataFrame ────────────────────────────────────────────────────
    if isinstance(feature_vector, dict):
        row = pd.DataFrame([feature_vector])
    elif isinstance(feature_vector, (list, np.ndarray)):
        row = pd.DataFrame([feature_vector], columns=feature_cols[: len(feature_vector)])
    elif isinstance(feature_vector, pd.Series):
        row = feature_vector.to_frame().T
    else:
        row = pd.DataFrame([feature_vector])

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0

    row = row[feature_cols]
    row = row.apply(pd.to_numeric, errors="coerce")
    row.replace([np.inf, -np.inf], np.nan, inplace=True)
    row.fillna(0, inplace=True)

    # ── Score ──────────────────────────────────────────────────────────────
    row_values = row.values.astype(float, copy=False)
    X_scaled   = scaler.transform(row_values)
    raw_score   = model.decision_function(X_scaled)[0]
    anomaly_score = _raw_to_score(raw_score, score_stats["lo"], score_stats["hi"])

    # Blend unsupervised anomaly score with supervised attack probability.
    if calibrator is not None:
        cal_prob = float(calibrator.predict_proba(row_values)[0][1])
        blend = score_stats.get("blend", {})
        w_if = float(blend.get("iforest", 0.25))
        w_cal = float(blend.get("calibrator", 0.75))
        denom = (w_if + w_cal) if (w_if + w_cal) > 1e-12 else 1.0
        anomaly_score = float(np.clip((w_if * anomaly_score + w_cal * cal_prob) / denom, 0.0, 1.0))
        anomaly_score = round(anomaly_score, 6)

    if log_score:
        _log_score(anomaly_score)
    return anomaly_score
