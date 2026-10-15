"""
AEGIS — train_all.py
====================
Trains BOTH models in one command:
  Step 8 → Isolation Forest  (anomaly detection)
  Step 9 → K-Means + DBSCAN  (attacker clustering)

Usage
-----
    python train_all.py

    # Or with options:
    python train_all.py --clusters 4 --max-rows 200000 --dbscan-eps 0.8

Outputs (saved to models/)
--------------------------
Step 8:
    models/isolation_forest.pkl
    models/scaler.pkl              ← Isolation Forest scaler
    models/feature_columns.pkl
    models/score_stats.pkl
    data/normal_traffic.csv

Step 9:
    models/kmeans_model.pkl
    models/dbscan_model.pkl
    models/cluster_scaler.pkl      ← Clustering scaler (NOT scaler.pkl)
    models/pca.pkl
    models/profile_map.pkl         ← {cluster_id: profile_name}
    outputs/clustered_flows.csv
    outputs/cluster_report.json

After training run:
    python ai_engine.py            ← smoke test both models together
    python test_model.py           ← Step 8 precision/recall validation
    python visualise_clusters.py   ← Step 9 plots
"""

import argparse
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# STEP 8 — Isolation Forest training
# ─────────────────────────────────────────────────────────────────
from anomaly_model import train_model as train_iforest

# ─────────────────────────────────────────────────────────────────
# STEP 9 — Clustering training
# ─────────────────────────────────────────────────────────────────
from cluster_attackers import (
    build_profile_report,
    load_cicids_data,
    prepare_features,
    run_dbscan,
    run_kmeans,
)
from cluster_attackers import (
    save_models as save_cluster_models,
)

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

STEP8_TRAINING_FILES = [
    os.path.join(DATA_DIR, "Monday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Tuesday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Wednesday-workingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
]


def banner(text: str):
    print()
    print("=" * 65)
    print(text)
    print("=" * 65)


# ── Step 8 ────────────────────────────────────────────────────────
def run_step8():
    banner("STEP 8 — Training Isolation Forest (Anomaly Detection)")

    missing = [f for f in STEP8_TRAINING_FILES if not os.path.exists(f)]
    if missing:
        print("\nERROR: Step 8 training file not found:")
        for f in missing:
            print(f"  {f}")
        print("\nFix: Place Monday-WorkingHours.pcap_ISCX.csv in data/")
        print("     Or edit STEP8_TRAINING_FILES in train_all.py")
        return False

    print(f"\nTraining on: {[os.path.basename(f) for f in STEP8_TRAINING_FILES]}")
    train_iforest(STEP8_TRAINING_FILES)

    print("\nStep 8 artifacts:")
    for f in ["isolation_forest.pkl", "scaler.pkl", "feature_columns.pkl", "score_stats.pkl"]:
        path = os.path.join(MODELS_DIR, f)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {status}  models/{f}")
    return True


# ── Step 9 ────────────────────────────────────────────────────────
def run_step9(n_clusters: int, max_rows: int, dbscan_eps: float):
    banner("STEP 9 — Training K-Means + DBSCAN (Attacker Clustering)")

    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(OUTPUTS_DIR).mkdir(exist_ok=True)

    print(f"\nLoading attack traffic from {DATA_DIR}/ ...")
    df = load_cicids_data(DATA_DIR, max_rows=max_rows)

    print("Preparing features ...")
    X_scaled, feature_names, true_labels, scaler = prepare_features(df)
    print(f"Feature matrix: {X_scaled.shape}")

    km_model, km_labels, km_score = run_kmeans(X_scaled, n_clusters=n_clusters)
    db_model, db_labels, db_score, pca = run_dbscan(X_scaled, eps=dbscan_eps)

    report, result_df, profiles = build_profile_report(
        df,
        km_labels,
        db_labels,
        km_model,
        feature_names,
        true_labels,
        OUTPUTS_DIR,
    )

    save_cluster_models(km_model, db_model, scaler, pca, profiles, MODELS_DIR)

    print("\nStep 9 artifacts:")
    for f in [
        "kmeans_model.pkl",
        "dbscan_model.pkl",
        "cluster_scaler.pkl",
        "pca.pkl",
        "profile_map.pkl",
    ]:
        path = os.path.join(MODELS_DIR, f)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {status}  models/{f}")

    print("\nAttacker profiles discovered:")
    for cid, info in report["kmeans_profiles"].items():
        print(
            f"  Cluster {cid} → {info['profile']:20s}  {info['size']:>7,} flows  ({info['pct']}%)"
        )

    return True


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Step 8 + Step 9 models")
    parser.add_argument("--clusters", type=int, default=4, help="K-Means clusters (default 4)")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Max rows for clustering")
    parser.add_argument("--dbscan-eps", type=float, default=0.8, help="DBSCAN epsilon")
    parser.add_argument("--skip-step8", action="store_true", help="Skip Isolation Forest training")
    parser.add_argument("--skip-step9", action="store_true", help="Skip clustering training")
    args = parser.parse_args()

    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(OUTPUTS_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print("\nAEGIS — Training all AI models")
    print(f"  Data dir   : {DATA_DIR}/")
    print(f"  Models dir : {MODELS_DIR}/")

    step8_ok = True
    step9_ok = True

    if not args.skip_step8:
        step8_ok = run_step8()
    else:
        print("\n[skipped] Step 8")

    if not args.skip_step9:
        step9_ok = run_step9(args.clusters, args.max_rows, args.dbscan_eps)
    else:
        print("\n[skipped] Step 9")

    banner("TRAINING COMPLETE")
    print(f"  Step 8 (Isolation Forest) : {'OK' if step8_ok else 'FAILED / SKIPPED'}")
    print(f"  Step 9 (Clustering)       : {'OK' if step9_ok else 'FAILED / SKIPPED'}")
    print()
    print("Next steps:")
    print("  python ai_engine.py          ← smoke test combined engine")
    print("  python test_model.py         ← Step 8 precision/recall report")
    print("  python visualise_clusters.py ← Step 9 plots")


if __name__ == "__main__":
    main()
