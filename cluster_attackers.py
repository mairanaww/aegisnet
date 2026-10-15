"""
STEP 9: AI Model – Attacker Behavior Clustering
================================================
WHAT:  Groups attackers by behavior style using K-Means and DBSCAN
WHY:   Different attacker profiles need different response strategies
USES:  scikit-learn, pandas, CICIDS 2017 (MachineLearningCVE dataset)

Cluster profiles produced:
  - Scanner           → rapid port/IP sweeps, low session depth
  - Automated Bot     → high volume, repetitive, no variation
  - Human Attacker    → deliberate, varied commands, longer sessions
  - Insider-like      → low volume, internal targets, privilege abuse

USAGE:
  python cluster_attackers.py --data data/ --output outputs/

NOTE (combined package):
  scaler is saved as models/cluster_scaler.pkl (NOT scaler.pkl)
  to avoid overwriting the Step 8 Isolation Forest scaler.
  profile_map.pkl is also saved so predict_profile() returns
  real names (Scanner / Automated Bot / Human Attacker / Insider-like)
  instead of generic "Cluster-N" strings.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Create required dirs before logging init
Path("logs").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/clustering.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Feature configuration ─────────────────────────────────────────────────────
BEHAVIORAL_FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    " Fwd IAT Total",
    " Fwd IAT Mean",
    " Bwd IAT Total",
    " Bwd IAT Mean",
    " Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    " Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    " FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
]

LABEL_COL = " Label"

ATTACK_CLASSES = [
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack  Brute Force",
    "Web Attack  XSS",
    "Infiltration",
    "Web Attack  Sql Injection",
    "Heartbleed",
]


# ── Cluster label mapping heuristics ─────────────────────────────────────────
def assign_profile_names_from_centroids(centers: np.ndarray, feature_names: list) -> dict:
    """
    Assigns profile names by RANKING clusters relative to each other.
    Returns: {cluster_id: profile_name}
    """
    n = len(centers)
    feat_map = {f.strip(): i for i, f in enumerate(feature_names)}

    def col(name):
        idx = feat_map.get(name.strip())
        if idx is None:
            return np.zeros(n)
        return centers[:, idx]

    pkt_rate = col("Flow Packets/s")
    syn = col("SYN Flag Count")
    flow_dur = col("Flow Duration")
    pkt_var = col("Packet Length Variance")
    ack = col("ACK Flag Count")
    bwd_pkts = col("Total Backward Packets")

    def rank(arr):
        order = np.argsort(arr)
        r = np.empty_like(order)
        r[order] = np.arange(n)
        return r

    r_pkt_rate = rank(pkt_rate)
    r_syn = rank(syn)
    r_dur = rank(flow_dur)
    r_var = rank(pkt_var)
    r_ack = rank(ack)
    r_bwd = rank(bwd_pkts)

    scanner_score = r_pkt_rate + r_syn + (n - 1 - r_dur)
    bot_score = r_pkt_rate + r_ack + (n - 1 - r_var)
    human_score = r_var + r_dur + r_bwd
    insider_score = r_dur + (n - 1 - r_pkt_rate) + r_bwd

    scores = np.stack([scanner_score, bot_score, human_score, insider_score], axis=1)
    profile_names = ["Scanner", "Automated Bot", "Human Attacker", "Insider-like"]

    assigned = {}
    used_profiles = set()
    used_clusters = set()

    pairs = []
    for cid in range(n):
        for pid in range(len(profile_names)):
            pairs.append((scores[cid, pid], cid, pid))
    pairs.sort(reverse=True)

    for score_val, cid, pid in pairs:
        if cid in used_clusters or pid in used_profiles:
            continue
        assigned[cid] = profile_names[pid]
        used_clusters.add(cid)
        used_profiles.add(pid)

    for cid in range(n):
        if cid not in assigned:
            assigned[cid] = f"Cluster-{cid}"

    return assigned


# ── Data loading ──────────────────────────────────────────────────────────────
def load_cicids_data(data_dir: str, max_rows: int = 200_000) -> pd.DataFrame:
    """
    Loads all CSV files from data_dir.
    Filters to attack traffic only (excludes BENIGN rows).
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Place your MachineLearningCVE CSV files in the data/ folder."
        )

    log.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    dfs = []
    rows_loaded = 0

    for f in csv_files:
        if rows_loaded >= max_rows:
            break
        remaining = max_rows - rows_loaded
        try:
            df = pd.read_csv(f, nrows=remaining, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            rows_loaded += len(df)
            log.info(f"  Loaded {len(df):,} rows from {f.name}")
        except Exception as e:
            log.warning(f"  Skipped {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total rows loaded: {len(combined):,}")

    if "Label" in combined.columns:
        combined.rename(columns={"Label": " Label"}, inplace=True)

    if LABEL_COL in combined.columns:
        attack_df = combined[combined[LABEL_COL].str.strip() != "BENIGN"].copy()
        log.info(f"Attack rows (non-BENIGN): {len(attack_df):,}")
        log.info(f"Attack types found:\n{attack_df[LABEL_COL].value_counts().to_string()}")
        return attack_df
    else:
        log.warning(f"Label column '{LABEL_COL}' not found. Using all rows.")
        return combined


# ── Feature engineering ───────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Selects, cleans, and scales behavioral features.
    Returns: (scaled_array, feature_names_used, labels_series_or_None, scaler)
    """
    col_map = {c.strip(): c for c in df.columns}
    resolved = []

    for feat in BEHAVIORAL_FEATURES:
        if feat in df.columns:
            resolved.append(feat)
        elif feat.strip() in col_map:
            resolved.append(col_map[feat.strip()])

    resolved = list(dict.fromkeys(resolved))

    if len(resolved) < 5:
        raise ValueError(
            f"Too few behavioral features found ({len(resolved)}).\n"
            f"Available columns sample: {list(df.columns[:10])}"
        )

    log.info(f"Using {len(resolved)} behavioral features for clustering.")

    X = df[resolved].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    thresh = 0.5 * len(X)
    X = X.dropna(axis=1, thresh=int(thresh))
    resolved = list(X.columns)

    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = df[LABEL_COL] if LABEL_COL in df.columns else None

    return X_scaled, resolved, labels, scaler


# ── Clustering ────────────────────────────────────────────────────────────────
def run_kmeans(X: np.ndarray, n_clusters: int = 4) -> tuple:
    log.info(f"Running K-Means with k={n_clusters} ...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=min(10_000, len(X)), random_state=42)
    log.info(f"K-Means silhouette score: {score:.4f}")
    return km, labels, score


def run_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 10) -> tuple:
    log.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples}) ...")
    pca = PCA(n_components=min(15, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log.info(f"DBSCAN found {n_clusters} clusters, {n_noise:,} noise points.")

    score = None
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > 1000:
            score = silhouette_score(
                X_pca[mask], labels[mask], sample_size=min(10_000, mask.sum()), random_state=42
            )
            log.info(f"DBSCAN silhouette score (excluding noise): {score:.4f}")

    return db, labels, score, pca


# ── Profile assignment ────────────────────────────────────────────────────────
def build_profile_report(
    df: pd.DataFrame,
    km_labels: np.ndarray,
    db_labels: np.ndarray,
    km_model,
    feature_names: list,
    true_labels,
    output_dir: str,
) -> tuple:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = df.copy().reset_index(drop=True)
    result_df["kmeans_cluster"] = km_labels
    result_df["dbscan_cluster"] = db_labels

    if true_labels is not None:
        result_df["true_label"] = true_labels.values

    profiles = assign_profile_names_from_centroids(km_model.cluster_centers_, feature_names)
    result_df["attacker_profile"] = result_df["kmeans_cluster"].map(profiles)

    report = {"kmeans_profiles": {}, "dbscan_summary": {}}

    for cid, name in profiles.items():
        mask = result_df["kmeans_cluster"] == cid
        cluster_rows = result_df[mask]
        entry = {
            "profile": name,
            "size": int(mask.sum()),
            "pct": round(float(mask.mean()) * 100, 2),
        }
        if true_labels is not None and LABEL_COL in result_df.columns:
            top_attacks = cluster_rows["true_label"].value_counts().head(5).to_dict()
            top_attacks = {str(k): int(v) for k, v in top_attacks.items()}
            entry["top_attack_types"] = top_attacks
        report["kmeans_profiles"][str(cid)] = entry
        log.info(f"  Cluster {cid} → '{name}': {mask.sum():,} flows ({mask.mean() * 100:.1f}%)")

    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    report["dbscan_summary"] = {
        "n_clusters": n_db_clusters,
        "noise_points": int((db_labels == -1).sum()),
    }

    result_df.to_csv(output_dir / "clustered_flows.csv", index=False)
    log.info(f"Saved clustered_flows.csv → {output_dir}/clustered_flows.csv")

    with open(output_dir / "cluster_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved cluster_report.json → {output_dir}/cluster_report.json")

    return report, result_df, profiles


# ── Model persistence ─────────────────────────────────────────────────────────
def save_models(km_model, db_model, scaler, pca, profile_map: dict, output_dir: str):
    """
    Save all Step 9 models.

    NOTE: scaler is saved as cluster_scaler.pkl (NOT scaler.pkl) so it
    does not overwrite the Step 8 Isolation Forest scaler.pkl.
    profile_map.pkl stores {cluster_id: profile_name} so predict_profile()
    can return real names at inference time.
    """
    output_dir = Path(output_dir)
    joblib.dump(km_model, output_dir / "kmeans_model.pkl")
    joblib.dump(db_model, output_dir / "dbscan_model.pkl")
    joblib.dump(scaler, output_dir / "cluster_scaler.pkl")  # ← renamed
    joblib.dump(pca, output_dir / "pca.pkl")
    joblib.dump(profile_map, output_dir / "profile_map.pkl")  # ← new
    log.info("Step 9 models saved to models/")
    log.info(
        "  kmeans_model.pkl | dbscan_model.pkl | cluster_scaler.pkl | pca.pkl | profile_map.pkl"
    )


# ── Inference (predict on new flow) ──────────────────────────────────────────
def predict_profile(flow_features: dict, models_dir: str = "models") -> dict:
    """
    Given a dict of raw flow features, returns the attacker profile.
    Called by the controller node (and ai_engine.py) at runtime.

    Returns:
        {"cluster_id": int, "attacker_profile": str}
        attacker_profile is one of:
            "Scanner" | "Automated Bot" | "Human Attacker" | "Insider-like"
    """
    models_dir = Path(models_dir)

    scaler = joblib.load(models_dir / "cluster_scaler.pkl")  # ← renamed
    km_model = joblib.load(models_dir / "kmeans_model.pkl")
    profile_map = joblib.load(models_dir / "profile_map.pkl")  # ← load saved map

    feat_vec = np.array([[flow_features.get(f, 0) for f in BEHAVIORAL_FEATURES]])
    feat_scaled = scaler.transform(feat_vec)
    cluster_id = int(km_model.predict(feat_scaled)[0])

    # Use the saved profile map — NOT the broken single-cluster wrapper
    profile = profile_map.get(cluster_id, f"Cluster-{cluster_id}")

    return {"cluster_id": cluster_id, "attacker_profile": profile}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 9: Attacker Behavior Clustering")
    parser.add_argument("--data", default="data/", help="Directory with CSVs")
    parser.add_argument("--output", default="outputs/", help="Directory for results")
    parser.add_argument("--models", default="models/", help="Directory to save models")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--dbscan-eps", type=float, default=0.8)
    args = parser.parse_args()

    Path(args.models).mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 9: Attacker Behavior Clustering")
    log.info("=" * 60)

    df = load_cicids_data(args.data, max_rows=args.max_rows)
    X_scaled, feature_names, true_labels, scaler = prepare_features(df)
    log.info(f"Feature matrix shape: {X_scaled.shape}")

    km_model, km_labels, km_score = run_kmeans(X_scaled, n_clusters=args.clusters)
    db_model, db_labels, db_score, pca = run_dbscan(X_scaled, eps=args.dbscan_eps)

    report, result_df, profiles = build_profile_report(
        df, km_labels, db_labels, km_model, feature_names, true_labels, args.output
    )

    save_models(km_model, db_model, scaler, pca, profiles, args.models)

    log.info("\n" + "=" * 60)
    log.info("CLUSTERING COMPLETE — ATTACKER PROFILES")
    log.info("=" * 60)
    for cid, info in report["kmeans_profiles"].items():
        log.info(f"  [{info['profile']:20s}]  {info['size']:>7,} flows  ({info['pct']}%)")
        if "top_attack_types" in info:
            for attack, cnt in list(info["top_attack_types"].items())[:3]:
                log.info(f"      ↳ {attack}: {cnt:,}")

    log.info(
        f"\nDBSCAN: {report['dbscan_summary']['n_clusters']} clusters, "
        f"{report['dbscan_summary']['noise_points']:,} noise points"
    )
    log.info(f"\nOutputs saved to: {args.output}")
    log.info(f"Models  saved to: {args.models}")


if __name__ == "__main__":
    main()
