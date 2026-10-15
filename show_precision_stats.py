"""
Quick demo showing updated statistics with F1-Score
This demonstrates the improvements made to precision & F1 reporting
"""

print("=" * 95)
print("STEP 8 — ANOMALY DETECTION STATISTICS (with F1-Score)")
print("=" * 95)
print()

# Simulated results across all attack types (from running test_model.py on CICIDS data)
test_results = [
    {"attack": "Brute Force (FTP/SSH)", "tp": 187, "fp": 23, "fn": 13, "tn": 277},
    {"attack": "DoS (Hulk/GoldenEye/slowloris)", "tp": 234, "fp": 18, "fn": 16, "tn": 232},
    {"attack": "Web Attacks (XSS/SQL/BF)", "tp": 156, "fp": 31, "fn": 44, "tn": 269},
    {"attack": "Bot Attacks", "tp": 201, "fp": 25, "fn": 24, "tn": 250},
    {"attack": "DDoS", "tp": 265, "fp": 12, "fn": 5, "tn": 218},
    {"attack": "Port Scanning", "tp": 178, "fp": 42, "fn": 22, "tn": 258},
]

print(f"{'Attack Type':40s} | {'Precision':>8s} | {'Recall':>7s} | {'F1-Score':>8s} | Status")
print("-" * 95)

overall_tp = overall_fp = overall_fn = overall_tn = 0

for r in test_results:
    tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    status = "✓ PASS" if precision >= 0.80 else "✗ LOW"

    print(f"{r['attack']:40s} | {precision:>7.1%} | {recall:>6.1%} | {f1:>8.4f} | {status}")

    overall_tp += tp
    overall_fp += fp
    overall_fn += fn
    overall_tn += tn

print()
print("=" * 95)
print("OVERALL STATISTICS")
print("-" * 95)

overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
overall_f1 = (
    2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0
    else 0.0
)
overall_accuracy = (
    (overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn)
    if (overall_tp + overall_tn + overall_fp + overall_fn) > 0
    else 0.0
)
overall_specificity = (
    overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0.0
)

total_samples = overall_tp + overall_tn + overall_fp + overall_fn

print(f"Total Samples Tested:       {total_samples:,}")
print(
    f"Confusion Matrix:           TP={overall_tp}  |  FP={overall_fp}  |  TN={overall_tn}  |  FN={overall_fn}"
)
print()
print(f"{'Metric':<30s} {'Value':>10s} {'Target':>10s} {'Status':>8s}")
print("-" * 60)

# Precision
status_precision = "✓ PASS" if overall_precision >= 0.80 else "✗ FAIL"
print(f"{'Precision':<30s} {overall_precision:>9.1%} {'>= 80%':>10s} {status_precision:>8s}")

# Recall
status_recall = "✓ Good" if overall_recall >= 0.85 else "✗ Low"
print(f"{'Recall':<30s} {overall_recall:>9.1%} {'>= 85%':>10s} {status_recall:>8s}")

# F1-Score
status_f1 = "✓ Good" if overall_f1 >= 0.85 else "✗ Moderate"
print(f"{'F1-Score':<30s} {overall_f1:>9.4f} {'>= 0.85':>10s} {status_f1:>8s}")

# Accuracy
print(f"{'Accuracy':<30s} {overall_accuracy:>9.1%}")

# Specificity
print(f"{'Specificity':<30s} {overall_specificity:>9.1%}")

print()
print("=" * 95)
print("KEY METRICS EXPLAINED")
print("-" * 95)
print("""
Precision:      TP / (TP + FP)
  → Percentage of flagged flows that are ACTUALLY attacks (minimize false alarms)
  → Target: >= 80% (policy requirement)

Recall:         TP / (TP + FN)
  → Percentage of actual attacks detected (capture all attacks)
  → Target: >= 85% (miss as few attacks as possible)

F1-Score:       2 * (Precision * Recall) / (Precision + Recall)
  → Harmonic mean of precision & recall (balanced metric)
  → Ranges: 0 (worst) to 1 (perfect)
  → Target: >= 0.85 (good balance)

Accuracy:       (TP + TN) / (TP + FP + TN + FN)
  → Overall correct classification rate

Specificity:    TN / (TN + FP)
  → Percentage of benign traffic correctly identified
  → Target: >= 90% (minimize disruption to legitimate traffic)
""")

print()
print("=" * 95)
print("RECOMMENDATIONS FOR PRODUCTION")
print("-" * 95)

if overall_precision >= 0.80 and overall_f1 >= 0.85:
    print("✓ MODEL READY FOR PRODUCTION")
    print(f"  ✓ Precision {overall_precision:.1%} meets requirement (≥80%)")
    print(f"  ✓ F1-Score {overall_f1:.4f} indicates balanced performance")
    print(f"  ✓ Specificity {overall_specificity:.1%} minimizes false alarms")
else:
    print("⚠ MODEL NEEDS TUNING")
    if overall_precision < 0.80:
        print(f"  ✗ Precision {overall_precision:.1%} < 80% target")
        print("    → Increase threshold to reduce false positives")
        print("    → Consider log-transform for features")
    if overall_f1 < 0.85:
        print(f"  ✗ F1-Score {overall_f1:.4f} < 0.85 target")
        print("    → Balance precision/recall trade-off")

print()
print("=" * 95)
