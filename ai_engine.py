"""
AEGIS — ai_engine.py
====================
Combined Step 8 + Step 9 + Step 10 entry point for the controller node.

What this does
--------------
  score_session(fv)       → float          (Step 8: anomaly score 0–1)
  classify_attacker(fv)   → str            (Step 9: attacker profile)
  compute_risk(score, fv) → dict           (Step 10: unified risk score)
  analyze_session(fv)     → dict           (all together + risk levels)

Controller node only ever needs to call analyze_session().

Usage
-----
  from ai_engine import analyze_session

  result = analyze_session(feature_vector)
  # {
  #   "anomaly_score":    0.82,
  #   "attacker_profile": "Human Attacker",
  #   "risk_band":        "HIGH"
  # }

Run order
---------
  1. python train_all.py          ← trains both models once
  2. python ai_engine.py          ← smoke test
  3. import from controller node
"""

import logging
import os
from datetime import datetime

# ── Step 8 import ──────────────────────────────────────────────────
from anomaly_model import score_session  # → float in [0, 1]

# ── Step 9 import ──────────────────────────────────────────────────
from cluster_attackers import predict_profile  # → {"cluster_id", "attacker_profile"}

# ── Step 10 import ─────────────────────────────────────────────────
from risk_scorer import compute_risk  # → unified risk score + level

# ── Logging ────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_engine.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RISK BAND MAPPING
# ══════════════════════════════════════════════════════════════════════════════


def _risk_band(score: float) -> str:
    """
    Maps anomaly_score → risk_band string used by the Policy Engine (Step 12).

    Score  | Band   | Policy Engine action
    -------|--------|---------------------
    < 0.3  | LOW    | Log only
    0.3–0.7| MEDIUM | Rate limit
    > 0.7  | HIGH   | Isolate real node + redirect to honeypot
    """
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    return "HIGH"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def analyze_session(feature_vector) -> dict:
    """
    Combined Step 8 + Step 9 + Step 10 call.  This is what the controller node imports.

    Parameters
    ----------
    feature_vector : dict  {"column_name": value, ...}
                   | list  [val1, val2, ...]  (same order as training features)
                   | pd.Series

    Returns
    -------
    dict:
        anomaly_score    (float, 0–1)      from Step 8 Isolation Forest
        attacker_profile (str)             from Step 9 K-Means clustering
            one of: "Scanner" | "Automated Bot" | "Human Attacker" | "Insider-like"
        risk_score       (float, 0–1)      from Step 10 Risk Scoring Engine
        risk_level       (str)             "LOW" | "MEDIUM" | "HIGH"
        risk_band        (str)             derived from anomaly_score (deprecated, kept for backward compatibility)
            one of: "LOW" | "MEDIUM" | "HIGH"

    Example
    -------
    >>> result = analyze_session(flow_feature_vector)
    >>> print(result)
    {
        "anomaly_score":    0.82,
        "attacker_profile": "Human Attacker",
        "risk_score":       0.803,
        "risk_level":       "HIGH",
        "risk_band":        "HIGH"
    }

    How the controller node uses this (Step 12 Policy Engine)
    ----------------------------------------------------------
    result = analyze_session(feature_vector)

    if result["risk_band"] == "HIGH":
        # → isolate real node (Step 13)
        # → redirect attacker to honeypot (Step 14)
        # → forensic capture (Step 15)

    # Fine-tune response based on profile:
    if result["attacker_profile"] == "Scanner":
        honeypot_depth = "shallow"   # fake ports only
    elif result["attacker_profile"] == "Human Attacker":
        honeypot_depth = "deep"      # fake filesystem + credentials
    elif result["attacker_profile"] == "Insider-like":
        honeypot_depth = "deep"      # decoy docs + forensic alert
    elif result["attacker_profile"] == "Automated Bot":
        honeypot_depth = "medium"    # collect IoCs
    """
    # ── Step 8: Anomaly score ──────────────────────────────────────
    anomaly_score = score_session(feature_vector)

    # ── Step 9: Attacker profile ───────────────────────────────────
    # predict_profile expects a dict; convert if needed
    if not isinstance(feature_vector, dict):
        try:
            import joblib
            import pandas as pd

            feature_cols = joblib.load("models/feature_columns.pkl")
            if isinstance(feature_vector, pd.Series):
                fv_dict = feature_vector.to_dict()
            else:
                arr = list(feature_vector)
                fv_dict = {col: arr[i] for i, col in enumerate(feature_cols) if i < len(arr)}
        except Exception:
            fv_dict = {}
    else:
        fv_dict = feature_vector

    cluster_result = predict_profile(fv_dict, models_dir="models")
    attacker_profile = cluster_result["attacker_profile"]
    cluster_id = cluster_result["cluster_id"]

    # ── Step 10: Risk Scoring ──────────────────────────────────────
    risk_result = compute_risk(anomaly_score, fv_dict)
    risk_score = risk_result["risk_score"]
    risk_level = risk_result["risk_level"]

    # ── Combine ────────────────────────────────────────────────────
    band = _risk_band(anomaly_score)  # legacy, kept for backward compatibility

    result = {
        "anomaly_score": round(anomaly_score, 4),
        "attacker_profile": attacker_profile,
        "risk_score": risk_score,  # Step 10: unified risk
        "risk_level": risk_level,  # Step 10: classification
        "risk_band": band,  # Step 8 legacy (deprecated)
        # extras the policy engine may want:
        "cluster_id": cluster_id,
        "timestamp": datetime.now().isoformat(),
        "risk_breakdown": risk_result.get("breakdown", {}),
    }

    logger.info(
        f"analyze_session | anomaly={anomaly_score:.3f} | risk_score={risk_score:.3f} ({risk_level}) | "
        f"profile={attacker_profile} | cluster={cluster_id}"
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (run directly: python ai_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("AEGIS — ai_engine.py smoke test")
    print("=" * 65)
    print()

    # Simulated feature vector (replace with real Step 7 output)
    # Keys match CICIDS 2017 MachineLearningCVE column names
    test_vector = {
        " Flow Duration": 500,
        " Total Fwd Packets": 10,
        " Total Backward Packets": 5,
        "Total Length of Fwd Packets": 1200,
        " Total Length of Bwd Packets": 600,
        " Fwd Packet Length Mean": 120.0,
        " Bwd Packet Length Mean": 60.0,
        " Flow Bytes/s": 1000.0,
        " Flow Packets/s": 4500.0,
        " Flow IAT Mean": 100.0,
        " SYN Flag Count": 8,
        " ACK Flag Count": 6,
        " PSH Flag Count": 3,
        " FIN Flag Count": 0,
        " RST Flag Count": 0,
        " Packet Length Variance": -1.5,
    }

    print("Input feature vector (subset shown):")
    for k, v in list(test_vector.items())[:6]:
        print(f"  {k.strip():35s}: {v}")
    print("  ...")
    print()

    result = analyze_session(test_vector)

    print("Result from analyze_session():")
    print(f"  anomaly_score    : {result['anomaly_score']}")
    print(f"  attacker_profile : {result['attacker_profile']}")
    print(f"  risk_score       : {result['risk_score']}")
    print(f"  risk_level       : {result['risk_level']}")
    print(f"  cluster_id       : {result['cluster_id']}")
    print()
    print("✓ OK — ai_engine is wired up correctly (Steps 8, 9, 10).")
    print("Run train_all.py first if you see FileNotFoundError.")
