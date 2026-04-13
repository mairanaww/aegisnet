"""Test F1 score across all thresholds"""
import pandas as pd
import numpy as np
import joblib
from anomaly_model import score_session, FEATURES_PATH

TEST_FILES = [
    ("data/Tuesday-WorkingHours.pcap_ISCX.csv", "Brute Force"),
]

SAMPLE_SIZE = 200
feature_cols = joblib.load(FEATURES_PATH)
label_col = "Label"

all_benign = []
all_attack = []

for csv_path, attack_name in TEST_FILES:
    print(f"Processing {csv_path}...")
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

print("\n" + "="*80)
print(f"F1 SCORE ANALYSIS ({len(benign_scores)} benign, {len(attack_scores)} attack)")
print("="*80)

print(f"\nBENIGN (n={len(benign_scores)}):")
print(f"  mean={benign_scores.mean():.4f}, std={benign_scores.std():.4f}")
print(f"  min={benign_scores.min():.4f}, max={benign_scores.max():.4f}")

print(f"\nATTACK (n={len(attack_scores)}):")
print(f"  mean={attack_scores.mean():.4f}, std={attack_scores.std():.4f}")
print(f"  min={attack_scores.min():.4f}, max={attack_scores.max():.4f}")

print("\n" + "="*80)
print("F1 SCORE BY THRESHOLD")
print("="*80)
print("Threshold | TP   | FP   | FN   | TN   | Precision | Recall | F1-Score")
print("-"*80)

best_f1 = None
best_config = None

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    tp = np.sum(attack_scores >= threshold)
    fp = np.sum(benign_scores >= threshold)
    fn = len(attack_scores) - tp
    tn = len(benign_scores) - fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  {threshold:.2f}   | {tp:4d} | {fp:4d} | {fn:4d} | {tn:4d} | {precision:8.1%} | {recall:6.1%} | {f1:.4f}")
    
    if best_f1 is None or f1 > best_f1:
        best_f1 = f1
        best_config = {"t": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp}

print("\n" + "="*80)
print("BEST F1 SCORE")
print("="*80)
if best_config:
    print(f"Threshold:  {best_config['t']:.2f}")
    print(f"Precision:  {best_config['precision']:.1%}")
    print(f"Recall:     {best_config['recall']:.1%}")
    print(f"F1-Score:   {best_config['f1']:.4f}")
    print(f"TP: {best_config['tp']}, FP: {best_config['fp']}")
