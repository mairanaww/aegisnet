"""
AEGIS Step 8 — Train Script  
Run this ONCE to train and save the Isolation Forest model.

Uses ONLY Monday (pure 100% BENIGN baseline):
  - Monday-WorkingHours.pcap_ISCX.csv (529,918 rows - 100% BENIGN)
  
This ultra-clean training data maximizes attack detection capacity.

Usage:
    python train_step8.py          ← Step 8 only
    python train_all.py            ← Step 8 + Step 9 together (recommended)
"""

import os
import sys
import pandas as pd
from anomaly_model import train_model

DATA_DIR = "data"
TRAINING_FILES = [
    os.path.join(DATA_DIR, "Monday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Tuesday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Wednesday-workingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
]

if __name__ == "__main__":
    missing = [f for f in TRAINING_FILES if not os.path.exists(f)]
    if missing:
        print("=" * 60)
        print("ERROR: Training data not found!")
        print("Missing files:")
        for f in missing:
            print(f"  {f}")
        print()
        print("Fix: Ensure all TRAINING_FILES exist in data/")
        print("=" * 60)
        sys.exit(1)

    print("Starting Isolation Forest training...")
    print(f"Training files ({len(TRAINING_FILES)}):")
    for f in TRAINING_FILES:
        print(f"  [OK] {os.path.basename(f)}")
    print("-" * 60)
    train_model(TRAINING_FILES)
    print("-" * 60)
    print("Done! Artifacts written to models/")
    print("  models/isolation_forest.pkl")
    print("  models/scaler.pkl")
    print("  models/feature_columns.pkl")
    print("  data/normal_traffic.csv")

