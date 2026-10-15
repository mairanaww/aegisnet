"""Analyze thresholds to find best precision."""

import joblib
import numpy as np
import pandas as pd

from anomaly_model import FEATURES_PATH, score_session

TEST_FILES = [
    ("data/Tuesday-WorkingHours.pcap_ISCX.csv", "Brute Force"),
    ("data/Wednesday-workingHours.pcap_ISCX.csv", "DoS"),
]

SAMPLE_SIZE = 500
feature_cols = joblib.load(FEATURES_PATH)
label_col = "Label"

all_benign = []
all_attack = []

for csv_path, attack_name in TEST_FILES:
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    benign = df[df[label_col].str.strip() == "BENIGN"].head(SAMPLE_SIZE)
    attack = df[df[label_col].str.strip() != "BENIGN"].head(SAMPLE_SIZE)

    for _, row in benign.iterrows():
        fv = {c: row[c] for c in feature_cols if c in row.index}
        score = score_session(fv, log_score=False)
        all_benign.append(score)

    for _, row in attack.iterrows():
        fv = {c: row[c] for c in feature_cols if c in row.index}
        score = score_session(fv, log_score=False)
        all_attack.append(score)

benign_scores = np.array(all_benign)
attack_scores = np.array(all_attack)

print("=" * 70)
print(f"COMPREHENSIVE ANALYSIS ({len(benign_scores)} benign, {len(attack_scores)} attack)")
print("=" * 70)

print(f"\nBENIGN (n={len(benign_scores)}):")
print(f"  mean={benign_scores.mean():.4f}, std={benign_scores.std():.4f}")
print(f"  min={benign_scores.min():.4f}, max={benign_scores.max():.4f}")
print(f"  25%={np.percentile(benign_scores, 25):.4f}")
print(f"  50%={np.percentile(benign_scores, 50):.4f}")
print(f"  75%={np.percentile(benign_scores, 75):.4f}")

print(f"\nATTACK (n={len(attack_scores)}):")
print(f"  mean={attack_scores.mean():.4f}, std={attack_scores.std():.4f}")
print(f"  min={attack_scores.min():.4f}, max={attack_scores.max():.4f}")
print(f"  25%={np.percentile(attack_scores, 25):.4f}")
print(f"  50%={np.percentile(attack_scores, 50):.4f}")
print(f"  75%={np.percentile(attack_scores, 75):.4f}")

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS (all thresholds from 0.0 to 1.0)")
print("=" * 70)
print("Threshold | TP   | FP   | FN   | TN   | Precision | Recall | F1")
print("-" * 70)

best_precision = None
best_config = None

for threshold in np.arange(0.0, 1.01, 0.05):
    tp = np.sum(attack_scores >= threshold)
    fp = np.sum(benign_scores >= threshold)
    fn = len(attack_scores) - tp
    tn = len(benign_scores) - fp

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    marker = ""
    if precision >= 0.80:
        marker = " <-- P>=80%"
        if best_precision is None or recall > best_config["recall"]:
            best_precision = precision
            best_config = {
                "t": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
            }

    print(
        f"  {threshold:.2f}   | {tp:4d} | {fp:4d} | {fn:4d} | {tn:4d} | {precision:8.1%} | {recall:6.1%} | {f1:.3f}{marker}"
    )

if best_config:
    print("\n" + "=" * 70)
    print("BEST CONFIG (Precision >= 80%)")
    print("=" * 70)
    print(f"Threshold: {best_config['t']:.2f}")
    print(f"Precision: {best_config['precision']:.1%}")
    print(f"Recall:    {best_config['recall']:.1%}")
    print(f"F1-Score:  {best_config['f1']:.3f}")
    print(f"TP={best_config['tp']}, FP={best_config['fp']}")
else:
    print("\n" + "=" * 70)
    print("NO CONFIGURATION ACHIEVES 80% PRECISION")
    print("=" * 70)

    # Find best F1 instead
    best_f1 = 0
    best_f1_config = None
    for threshold in np.arange(0.0, 1.01, 0.05):
        tp = np.sum(attack_scores >= threshold)
        fp = np.sum(benign_scores >= threshold)
        fn = len(attack_scores) - tp

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        if f1 > best_f1:
            best_f1 = f1
            best_f1_config = {
                "t": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
            }

    print("\nBest F1 Score instead:")
    print(f"Threshold: {best_f1_config['t']:.2f}")
    print(f"Precision: {best_f1_config['precision']:.1%}")
    print(f"Recall:    {best_f1_config['recall']:.1%}")
    print(f"F1-Score:  {best_f1_config['f1']:.3f}")
