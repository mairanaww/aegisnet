"""
demo_synthetic.py
─────────────────
Generates a synthetic CICIDS-shaped dataset so you can test the full
pipeline WITHOUT placing any CSV files in data/.

Creates: data/synthetic_cicids_demo.csv

Then runs: cluster_attackers.py --data data/ --output outputs/ --models models/

Usage:
  python demo_synthetic.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
N_PER_CLASS = 5_000
rng = np.random.default_rng(42)


def make_scanner(n):
    d = {f: rng.normal(0, 1, n) for f in BEHAVIORAL_FEATURES}
    d[" Flow Duration"] = rng.exponential(100, n)  # very short
    d[" SYN Flag Count"] = rng.poisson(8, n).astype(float)  # many SYNs
    d[" Flow Packets/s"] = rng.normal(5000, 500, n)  # high rate
    d[" Flow Bytes/s"] = rng.normal(1000, 200, n)
    d[" Packet Length Variance"] = rng.normal(-1, 0.2, n)  # low variance
    d[LABEL_COL] = ["PortScan"] * n
    return pd.DataFrame(d)


def make_bot(n):
    d = {f: rng.normal(0, 1, n) for f in BEHAVIORAL_FEATURES}
    d[" Flow Packets/s"] = rng.normal(3000, 300, n)
    d[" ACK Flag Count"] = rng.poisson(6, n).astype(float)
    d[" Packet Length Variance"] = rng.normal(-2, 0.3, n)  # very low variance
    d[" Flow Duration"] = rng.normal(500, 50, n)
    d[LABEL_COL] = ["Bot"] * n
    return pd.DataFrame(d)


def make_human(n):
    d = {f: rng.normal(0, 1, n) for f in BEHAVIORAL_FEATURES}
    d[" Flow Duration"] = rng.normal(50_000, 10_000, n)  # long sessions
    d[" Packet Length Variance"] = rng.normal(3, 0.5, n)  # high variance
    d[" Flow Packets/s"] = rng.normal(50, 20, n)  # slow/interactive
    d[LABEL_COL] = rng.choice(
        ["Web Attack  Brute Force", "SSH-Patator", "Web Attack  XSS"], n
    ).tolist()
    return pd.DataFrame(d)


def make_insider(n):
    d = {f: rng.normal(0, 1, n) for f in BEHAVIORAL_FEATURES}
    d[" Flow Duration"] = rng.normal(200_000, 30_000, n)  # very long
    d[" Flow Packets/s"] = rng.normal(10, 5, n)  # very low rate
    d[" Total Fwd Packets"] = rng.normal(20, 5, n)
    d[LABEL_COL] = rng.choice(["Infiltration", "FTP-Patator"], n).tolist()
    return pd.DataFrame(d)


def main():
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    out = Path("data/synthetic_cicids_demo.csv")

    print("Generating synthetic CICIDS-style dataset …")
    df = pd.concat(
        [
            make_scanner(N_PER_CLASS),
            make_bot(N_PER_CLASS),
            make_human(N_PER_CLASS),
            make_insider(N_PER_CLASS),
        ],
        ignore_index=True,
    )

    # Clip negatives in count-like columns
    for col in [
        " SYN Flag Count",
        " ACK Flag Count",
        " Total Fwd Packets",
        " Total Backward Packets",
        " Flow Packets/s",
        " Flow Bytes/s",
    ]:
        df[col] = df[col].clip(lower=0)

    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")

    print("\nRunning cluster_attackers.py on synthetic data …\n")
    subprocess.run(
        [
            sys.executable,
            "cluster_attackers.py",
            "--data",
            "data/",
            "--output",
            "outputs/",
            "--models",
            "models/",
            "--clusters",
            "4",
        ],
        check=True,
    )

    print("\nRunning visualise_clusters.py …\n")
    subprocess.run(
        [
            sys.executable,
            "visualise_clusters.py",
            "--output",
            "outputs/",
            "--plots",
            "outputs/plots/",
        ],
        check=True,
    )

    print("\n✅  Demo complete!  Check outputs/ and outputs/plots/")


if __name__ == "__main__":
    main()
