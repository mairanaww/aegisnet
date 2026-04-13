"""
visualise_clusters.py
─────────────────────
Generates plots from clustered_flows.csv:
  1. 2-D PCA scatter — KMeans clusters
  2. 2-D PCA scatter — DBSCAN clusters
  3. Bar chart — cluster size distribution
  4. Heatmap  — feature importance per cluster
  5. Stacked bar — true attack types per cluster (if label available)

Run AFTER cluster_attackers.py:
  python visualise_clusters.py --output outputs/ --plots outputs/plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(output_dir: str):
    df = pd.read_csv(Path(output_dir) / "clustered_flows.csv", low_memory=False)
    report = json.loads((Path(output_dir) / "cluster_report.json").read_text())
    return df, report


def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {"kmeans_cluster", "dbscan_cluster", "attacker_profile",
            "true_label", " Label", "Label"}
    return [c for c in df.columns if c not in skip and df[c].dtype in (np.float64, np.int64, float, int)]


def pca_2d(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def plot_scatter(coords, labels, title, cmap_name, save_path, label_map=None):
    unique = sorted(set(labels))
    cmap = cm.get_cmap(cmap_name, len(unique))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, uid in enumerate(unique):
        mask = labels == uid
        name = label_map.get(uid, f"C{uid}") if label_map else str(uid)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=4, alpha=0.4, color=cmap(i), label=name)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.legend(markerscale=3, framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_dbscan_scatter(coords, labels, title, save_path):
    """Special handling for DBSCAN scatter with many clusters.
    Highlights noise points (-1) separately and groups clusters by cyclic colors."""
    unique = sorted(set(labels))
    has_noise = -1 in unique
    cluster_ids = [c for c in unique if c != -1]
    
    # Use cyclic colormap for actual clusters, gray for noise
    n_clusters = len(cluster_ids)
    cmap = cm.get_cmap("tab20c", max(3, n_clusters % 20) if n_clusters <= 20 else 20)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot clusters with cycling colors
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = cmap(i % cmap.N) if n_clusters > 20 else cmap(i)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=5, alpha=0.5, color=color, label=f"C{cid}" if i < 15 else "")
    
    # Plot noise points in gray, semi-transparent
    if has_noise:
        mask = labels == -1
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=5, alpha=0.2, color="gray", marker="x", label=f"Noise ({mask.sum()})")
    
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    
    # Show stats in legend
    legend_title = f"{n_clusters} clusters" + (f" + {mask.sum()} noise" if has_noise else "")
    ax.legend(markerscale=2, framealpha=0.85, fontsize=7, title=legend_title, 
              loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cluster_sizes(report, save_path):
    profiles = report["kmeans_profiles"]
    names  = [v["profile"] for v in profiles.values()]
    sizes  = [v["size"]    for v in profiles.values()]
    pcts   = [v["pct"]     for v in profiles.values()]

    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71",
              "#9b59b6", "#1abc9c", "#f39c12", "#d35400"][:len(names)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, sizes, color=colors, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + max(sizes)*0.01, bar.get_y() + bar.get_height()/2,
                f"{pct}%", va="center", fontsize=9)
    ax.set_xlabel("Number of flows")
    ax.set_title("Attacker Profile Distribution", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_heatmap(df, feature_cols, save_path, top_n=15):
    # Compute per-cluster mean for top-variance features
    variances = df[feature_cols].var(numeric_only=True).nlargest(top_n)
    top_feats = list(variances.index)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[top_feats].fillna(0))
    tmp = pd.DataFrame(scaled, columns=top_feats)
    tmp["cluster"] = df["attacker_profile"].values

    heatmap_data = tmp.groupby("cluster")[top_feats].mean()

    fig, ax = plt.subplots(figsize=(14, max(4, len(heatmap_data))))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels([f.strip()[:25] for f in top_feats], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index, fontsize=9)
    ax.set_title("Feature Means per Attacker Profile (Z-scored)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_behavioral_features(df, feature_cols, save_path):
    """Show key behavioral feature differences that distinguish the clusters."""
    # Select key differentiating features
    key_features = []
    for feat in ["Flow Packets/s", "Flow Duration", "SYN Flag Count", 
                 "Down/Up Ratio", "Fwd Packets/s", "Bwd Packets/s",
                 "ACK Flag Count", "Min Packet Length"]:
        if feat in feature_cols:
            key_features.append(feat)
    
    if not key_features:
        key_features = feature_cols[:5]  # fallback to first 5
    
    # Compute mean per cluster
    agg_data = df.groupby("attacker_profile")[key_features].mean()
    
    # Replace inf and handle extreme values
    agg_data = agg_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Normalize for visibility
    scaler = StandardScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(agg_data),
        index=agg_data.index,
        columns=agg_data.columns
    )
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(normalized.index))
    width = 0.11
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    
    for i, feat in enumerate(normalized.columns):
        ax.bar(x + i*width, normalized[feat], width, label=feat.strip()[:20], 
               color=colors[i % len(colors)], alpha=0.8)
    
    # Add background shading for each cluster
    cluster_colors = ["#f0f0f0", "#ffffff"]
    for i in range(len(normalized.index)):
        x_start = i - 0.5
        x_end = i + 0.5
        ax.axvspan(x_start, x_end, alpha=0.15, color=cluster_colors[i % 2], zorder=-1)
    
    # Add vertical dashed lines between clusters
    for i in range(1, len(normalized.index)):
        cluster_boundary = i - 0.5
        ax.axvline(x=cluster_boundary, color="black", linestyle="--", linewidth=2, alpha=0.5)
    
    ax.set_ylabel("Normalized Feature Value (Z-score)", fontsize=10)
    ax.set_xlabel("Attacker Profile", fontsize=10)
    ax.set_title("Behavioral Feature Characteristics per Cluster", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (len(normalized.columns)-1) / 2)
    ax.set_xticklabels(normalized.index, fontsize=10, fontweight="bold")
    # Move legend outside the plot to the right
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_attack_type_breakdown(df, report, save_path):
    if "true_label" not in df.columns:
        print("No true_label column — skipping attack-type breakdown chart.")
        return

    pivot = (df.groupby(["attacker_profile", "true_label"])
               .size()
               .unstack(fill_value=0))

    # Normalise to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_pct.columns)))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(pivot_pct))
    for i, col in enumerate(pivot_pct.columns):
        ax.bar(pivot_pct.index, pivot_pct[col], bottom=bottom,
               label=col.strip(), color=colors[i], edgecolor="white", linewidth=0.3)
        bottom += pivot_pct[col].values

    ax.set_ylabel("% of flows")
    ax.set_xlabel("Attacker Profile")
    ax.set_title("True Attack Types per Attacker Profile", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--plots",  default="outputs/plots/")
    args = parser.parse_args()

    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clustered data …")
    df, report = load_data(args.output)

    feature_cols = get_feature_cols(df)
    print(f"Feature columns for visualisation: {len(feature_cols)}")

    # --- Subsample for PCA speed ---
    SAMPLE = min(30_000, len(df))
    sample_df = df.sample(SAMPLE, random_state=42)

    X = sample_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    coords = pca_2d(X)

    # Build label_map: cluster_id → profile name
    label_map = {int(k): v["profile"] for k, v in report["kmeans_profiles"].items()}

    # 1. KMeans scatter
    plot_scatter(
        coords, sample_df["kmeans_cluster"].values,
        "KMeans — Attacker Behavior Clusters (PCA 2-D)",
        "tab10",
        plots_dir / "kmeans_scatter.png",
        label_map=label_map,
    )

    # 2. DBSCAN scatter (with improved handling for many clusters)
    plot_dbscan_scatter(
        coords, sample_df["dbscan_cluster"].values,
        "DBSCAN — Micro-Behavior Clusters (PCA 2-D)",
        plots_dir / "dbscan_scatter.png",
    )

    # 3. Profile sizes
    plot_cluster_sizes(report, plots_dir / "profile_distribution.png")

    # 4. Feature heatmap
    plot_feature_heatmap(df, feature_cols, plots_dir / "feature_heatmap.png")

    # 5. Behavioral characteristics per cluster
    plot_behavioral_features(df, feature_cols, plots_dir / "behavioral_characteristics.png")

    # 6. Attack-type breakdown
    plot_attack_type_breakdown(df, report, plots_dir / "attack_type_breakdown.png")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
