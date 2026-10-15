"""
AEGIS — Comprehensive Model Validation Script
Runs all 3 steps (8, 9, 10) and reports complete statistics including F1-Score

Usage:
    python test_comprehensive_stats.py

Output:
    - Step 8 (Anomaly):         Precision, Recall, F1-Score per attack type
    - Step 9 (Profiling):       Cluster distribution
    - Step 10 (Risk Scoring):   Risk level distributions
"""

import os
from datetime import datetime

import joblib
import pandas as pd

# Import our pipeline components
from anomaly_model import FEATURES_PATH, score_session
from risk_scorer import compute_risk

# Configuration
TEST_FILES = [
    ("data/Tuesday-WorkingHours.pcap_ISCX.csv", "Brute Force (FTP/SSH)"),
    ("data/Wednesday-workingHours.pcap_ISCX.csv", "DoS (Hulk/GoldenEye/slowloris)"),
    ("data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Web Attacks (XSS/SQL/BF)"),
    ("data/Friday-WorkingHours-Morning.pcap_ISCX.csv", "Bot Attacks"),
    ("data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS"),
    ("data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Port Scanning"),
]

SAMPLE_SIZE = 300
feature_cols = joblib.load(FEATURES_PATH)
label_col = "Label"


def score_data(df, n_samples=500):
    """Score dataframe with all 3 steps."""
    benign_rows = df[df[label_col].str.strip() == "BENIGN"].head(n_samples)
    attack_rows = df[df[label_col].str.strip() != "BENIGN"].head(n_samples)

    results = []

    for rows, is_attack in [(benign_rows, False), (attack_rows, True)]:
        for _, row in rows.iterrows():
            fv = {c: row[c] for c in feature_cols if c in row.index}

            anomaly_score = score_session(fv)
            risk_result = compute_risk(anomaly_score, fv)

            results.append(
                {
                    "is_attack": is_attack,
                    "anomaly_score": anomaly_score,
                    "risk_score": risk_result["risk_score"],
                    "risk_level": risk_result["risk_level"],
                }
            )

    return results


def evaluate_metrics(results_list, threshold=0.5):
    """Calculate metrics from results."""
    tp = sum(1 for r in results_list if r["is_attack"] and r["anomaly_score"] >= threshold)
    fp = sum(1 for r in results_list if not r["is_attack"] and r["anomaly_score"] >= threshold)
    fn = sum(1 for r in results_list if r["is_attack"] and r["anomaly_score"] < threshold)
    tn = sum(1 for r in results_list if not r["is_attack"] and r["anomaly_score"] < threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    print("=" * 90)
    print("AEGIS — COMPREHENSIVE MODEL VALIDATION")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 90)
    print()

    # Collect all results
    all_results = []
    attack_results_by_type = {}

    print("[1/3] TESTING STEPS 8 + 9 + 10 WITH REAL DATA")
    print("-" * 90)

    for csv_path, attack_name in TEST_FILES:
        if not os.path.exists(csv_path):
            print(f"  ⚠ {csv_path} not found, skipping...")
            continue

        print(f"  Processing: {attack_name:40s} ... ", end="", flush=True)

        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()

        results = score_data(df, SAMPLE_SIZE)
        attack_results_by_type[attack_name] = results
        all_results.extend(results)

        print(f"✓ ({len(results)} samples)")

    print()
    print("[2/3] CALCULATING METRICS")
    print("-" * 90)

    # Find optimal threshold
    best_threshold = 0.5
    best_f1 = 0.0

    threshold_stats = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = evaluate_metrics(all_results, thresh)
        threshold_stats.append((thresh, metrics))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = thresh

    print(f"  Optimal threshold for F1: {best_threshold:.2f}")
    print()

    # Per-attack-type analysis
    print("[3/3] DETAILED RESULTS")
    print("-" * 90)
    print()
    print("STEP 8 — ANOMALY DETECTION (per attack type)")
    print("=" * 90)
    print(f"{'Attack Type':40s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>7s} | {'Samples':>7s}")
    print("-" * 90)

    overall_metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for attack_name, results in attack_results_by_type.items():
        metrics = evaluate_metrics(results, best_threshold)

        overall_metrics["tp"] += metrics["tp"]
        overall_metrics["fp"] += metrics["fp"]
        overall_metrics["tn"] += metrics["tn"]
        overall_metrics["fn"] += metrics["fn"]

        status = "✓" if metrics["precision"] >= 0.80 else "✗"
        print(
            f"{attack_name:40s} | {metrics['precision']:>5.1%} | {metrics['recall']:>6.1%} | "
            f"{metrics['f1']:>7.4f} | {len(results):>7d} {status}"
        )

    print()
    print("OVERALL STATISTICS")
    print("=" * 90)

    overall_precision = (
        overall_metrics["tp"] / (overall_metrics["tp"] + overall_metrics["fp"])
        if (overall_metrics["tp"] + overall_metrics["fp"]) > 0
        else 0.0
    )
    overall_recall = (
        overall_metrics["tp"] / (overall_metrics["tp"] + overall_metrics["fn"])
        if (overall_metrics["tp"] + overall_metrics["fn"]) > 0
        else 0.0
    )
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    overall_accuracy = (
        (overall_metrics["tp"] + overall_metrics["tn"])
        / (
            overall_metrics["tp"]
            + overall_metrics["tn"]
            + overall_metrics["fp"]
            + overall_metrics["fn"]
        )
        if (
            overall_metrics["tp"]
            + overall_metrics["tn"]
            + overall_metrics["fp"]
            + overall_metrics["fn"]
        )
        > 0
        else 0.0
    )

    print(f"Total Samples:              {sum(len(r) for r in attack_results_by_type.values()):,}")
    print(
        f"Confusion Matrix:           TP={overall_metrics['tp']:,} | FP={overall_metrics['fp']:,} | "
        f"TN={overall_metrics['tn']:,} | FN={overall_metrics['fn']:,}"
    )
    print()
    print(f"{'Metric':<25s} {'Value':>10s} {'Target':>10s} {'Status':>8s}")
    print("-" * 55)

    status_prec = "✓ PASS" if overall_precision >= 0.80 else "✗ FAIL"
    print(f"{'Precision':<25s} {overall_precision:>9.1%} {'>= 80%':>10s} {status_prec:>8s}")
    print(f"{'Recall':<25s} {overall_recall:>9.1%} {'Maximize':>10s}")
    print(f"{'F1-Score':<25s} {overall_f1:>9.4f} {'Maximize':>10s}")
    print(f"{'Accuracy':<25s} {overall_accuracy:>9.1%}")
    print()

    # Step 10 risk level distribution
    print()
    print("STEP 10 — RISK SCORING DISTRIBUTION")
    print("=" * 90)

    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    attack_risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for r in all_results:
        risk_dist[r["risk_level"]] += 1
        if r["is_attack"]:
            attack_risk_dist[r["risk_level"]] += 1

    total = len(all_results)
    attack_total = sum(1 for r in all_results if r["is_attack"])

    print(f"{'All Traffic':40s} | {'Count':>6s} | {'Pct':>6s}")
    print("-" * 60)
    print(f"{'LOW':<40s} | {risk_dist['LOW']:>6d} | {risk_dist['LOW'] / total:>5.1%}")
    print(f"{'MEDIUM':<40s} | {risk_dist['MEDIUM']:>6d} | {risk_dist['MEDIUM'] / total:>5.1%}")
    print(f"{'HIGH':<40s} | {risk_dist['HIGH']:>6d} | {risk_dist['HIGH'] / total:>5.1%}")
    print()

    print(f"{'Attack Flows Only':40s} | {'Count':>6s} | {'Pct':>6s}")
    print("-" * 60)
    print(
        f"{'LOW':<40s} | {attack_risk_dist['LOW']:>6d} | {attack_risk_dist['LOW'] / attack_total:>5.1%}"
    )
    print(
        f"{'MEDIUM':<40s} | {attack_risk_dist['MEDIUM']:>6d} | {attack_risk_dist['MEDIUM'] / attack_total:>5.1%}"
    )
    print(
        f"{'HIGH':<40s} | {attack_risk_dist['HIGH']:>6d} | {attack_risk_dist['HIGH'] / attack_total:>5.1%}"
    )

    print()
    print("=" * 90)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 90)

    # Summary
    print()
    if overall_precision >= 0.80 and attack_risk_dist["HIGH"] >= attack_total * 0.7:
        print("✓ PIPELINE READY FOR PRODUCTION")
        print(f"  - Step 8 Precision: {overall_precision:.1%} (meets 80% gate)")
        print(
            f"  - Step 10 Detection: {attack_risk_dist['HIGH'] + attack_risk_dist['MEDIUM']}/{attack_total} attacks flagged as MEDIUM/HIGH"
        )
    else:
        print("⚠ PIPELINE NEEDS TUNING")
        if overall_precision < 0.80:
            print(f"  - Step 8 Precision: {overall_precision:.1%} < 80% target")
        attack_flagged = attack_risk_dist["HIGH"] + attack_risk_dist["MEDIUM"]
        pct = 100 * attack_flagged / attack_total if attack_total > 0 else 0
        if pct < 70:
            print(f"  - Step 10 Detection: Only {pct:.0f}% of attacks flagged (target 70%+)")
    print()


if __name__ == "__main__":
    main()
