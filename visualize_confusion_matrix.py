"""
Visualize Confusion Matrix for Step 8 Anomaly Detection
"""

import matplotlib.pyplot as plt
import numpy as np

# Current model metrics from show_precision_stats.py
TP = 1221
FP = 151
TN = 1504
FN = 124

# Build confusion matrix
# [[TN, FP],
#  [FN, TP]]
cm = np.array([[TN, FP], [FN, TP]])

# Calculate metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + FP + TN + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("=" * 80)
print("CONFUSION MATRIX VISUALIZATION")
print("=" * 80)
print("\nConfusion Matrix:")
print("                 Predicted Negative  |  Predicted Positive")
print(f"Actual Negative  |      {TN:4d}          |       {FP:4d}")
print(f"Actual Positive  |      {FN:4d}          |       {TP:4d}")

print("\n\nMetrics:")
print(f"  True Positives (TP):   {TP}  — Correctly identified attacks")
print(f"  False Positives (FP):  {FP}  — Benign flagged as attack")
print(f"  True Negatives (TN):   {TN}  — Correctly identified benign")
print(f"  False Negatives (FN):  {FN}  — Attacks missed")

print(f"\n  Precision:   {precision:.1%}  (TP / (TP + FP))")
print(f"  Recall:      {recall:.1%}  (TP / (TP + FN))")
print(f"  Specificity: {specificity:.1%}  (TN / (TN + FP))")
print(f"  F1-Score:    {f1:.4f}")
print(f"  Accuracy:    {accuracy:.1%}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix Heatmap (Counts) - manually drawn
ax1 = axes[0]
im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")

# Set ticks and labels
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["Negative\n(Benign)", "Positive\n(Attack)"], fontsize=11)
ax1.set_yticklabels(["Actual\nNegative", "Actual\nPositive"], fontsize=11)
ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("True Label", fontsize=12)
ax1.set_xlabel("Predicted Label", fontsize=12)

# Add text annotations
for i in range(2):
    for j in range(2):
        value = cm[i, j]
        text = ax1.text(
            j,
            i,
            f"{value}",
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=16,
            fontweight="bold",
        )

plt.colorbar(im1, ax=ax1, label="Count")

# 2. Normalized Heatmap (Percentages by row)
ax2 = axes[1]
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
im2 = ax2.imshow(cm_normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Negative\n(Benign)", "Positive\n(Attack)"], fontsize=11)
ax2.set_yticklabels(["Actual\nNegative", "Actual\nPositive"], fontsize=11)
ax2.set_title("Confusion Matrix (Normalized %)", fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("True Label", fontsize=12)
ax2.set_xlabel("Predicted Label", fontsize=12)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        pct = cm_normalized[i, j]
        text = ax2.text(
            j,
            i,
            f"{pct:.1%}",
            ha="center",
            va="center",
            color="white" if pct > 0.5 else "black",
            fontsize=16,
            fontweight="bold",
        )

plt.colorbar(im2, ax=ax2, label="Percentage")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: outputs/confusion_matrix.png")
plt.close(fig)  # Close figure instead of show()

# 3. Metrics comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ["Precision", "Recall", "Specificity", "Accuracy"]
values = [precision, recall, specificity, accuracy]
targets = [0.80, 0.85, 0.90, 0.85]  # Different targets for each
colors = ["#2ecc71" if v >= t else "#f39c12" for v, t in zip(values, targets)]

x_pos = np.arange(len(metrics))
bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2, width=0.6)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.1%}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim([0, 1.05])
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Add reference lines
ax.axhline(y=0.85, color="red", linestyle="--", linewidth=2, alpha=0.7, label="High Target (85%)")
ax.axhline(y=0.80, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Min Target (80%)")
ax.legend(fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("outputs/metrics_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Saved: outputs/metrics_comparison.png")
plt.close(fig)

# 4. Classification outcomes pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prediction accuracy pie
sizes_pred = [TP + TN, FP + FN]
labels_pred = [
    f"Correct\n{TP + TN}\n({(TP + TN) / (TP + FP + TN + FN):.1%})",
    f"Incorrect\n{FP + FN}\n({(FP + FN) / (TP + FP + TN + FN):.1%})",
]
colors_pred = ["#2ecc71", "#e74c3c"]
explode_pred = (0.05, 0.05)
axes[0].pie(
    sizes_pred,
    labels=labels_pred,
    colors=colors_pred,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10, "weight": "bold"},
    explode=explode_pred,
)
axes[0].set_title("Overall Classification Result", fontsize=12, fontweight="bold")

# Attack detection accuracy pie
sizes_attack = [TP, FN]
labels_attack = [f"Detected\n{TP}\n({recall:.1%})", f"Missed\n{FN}\n({(FN) / (TP + FN):.1%})"]
colors_attack = ["#2ecc71", "#e74c3c"]
explode_attack = (0.05, 0.05)
axes[1].pie(
    sizes_attack,
    labels=labels_attack,
    colors=colors_attack,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10, "weight": "bold"},
    explode=explode_attack,
)
axes[1].set_title("Attack Detection Rate (Recall)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/classification_outcomes.png", dpi=300, bbox_inches="tight")
print("✓ Saved: outputs/classification_outcomes.png")
plt.close(fig)

print("\n" + "=" * 80)
print("✓ All visualizations saved to outputs/")
print("=" * 80)
