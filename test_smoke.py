"""Quick smoke test to verify scoring is correct."""
import numpy as np
import pandas as pd
import joblib
from anomaly_model import score_session

feature_cols = joblib.load("models/feature_columns.pkl")

print("=" * 70)
print("SMOKE TEST — synthetic feature vectors")
print("=" * 70)

# All-zero vector → should look normal
normal_fv  = {col: 0.0 for col in feature_cols}
# Large values → should look anomalous
attack_fv  = {col: 1e6 for col in feature_cols}

score_normal = score_session(normal_fv)
score_attack = score_session(attack_fv)

print(f"Normal  vector score : {score_normal:.4f}  (expect < 0.5)")
print(f"Attack  vector score : {score_attack:.4f}  (expect > 0.5)")

if score_normal < 0.5 and score_attack > 0.5:
    print("Ordering correct     : YES [OK]")
else:
    print("Ordering correct     : NO [FAILED]")
    print(f"  Normal should be < 0.5, got {score_normal:.4f}")
    print(f"  Attack should be > 0.5, got {score_attack:.4f}")
