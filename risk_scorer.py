"""
AEGIS - Step 10: Risk Scoring Engine
Person 5 - Policy Engine

Combines multiple threat signals into a unified risk score.

What it does
============
Weighted combination of:
  • anomaly_score from Step 8 (structural network anomaly)
  • privilege escalation indicator (failed auth attempts, flag patterns)
  • lateral movement indicator (port scanning, multi-destination behavior)

Output: risk_score [0,1] + risk_level {LOW|MEDIUM|HIGH} + diagnostic breakdown

Exports
=======
compute_risk(anomaly_score: float, feature_vector: dict) -> dict
  - risk_score        : float [0, 1]
  - risk_level        : 'LOW' | 'MEDIUM' | 'HIGH'
  - breakdown         : dict with component scores for auditing
  - timestamp         : ISO timestamp of evaluation
  - source            : source IP (if available in feature_vector)
"""

import os
import json
import logging
import numpy as np
from datetime import datetime

# ── Logging setup 
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/risk_scoring.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Config
CONFIG_PATH = "risk_scorer_config.json"


def _load_config():
    """Load tunable weights and thresholds from config."""
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Config not found at {CONFIG_PATH}. Using hardcoded defaults.")
        return {
            "weights": {"anomaly_score": 0.5, "privilege_escalation": 0.3, "lateral_movement": 0.2},
            "thresholds": {"low": 0.3, "medium": 0.7, "high": 1.0},
            "normalization_limits": {
                "syn_flag_count_max": 100,
                "fin_flag_count_max": 50,
                "ack_flag_count_max": 200,
                "pst_flag_count_max": 50,
                "urg_flag_count_max": 20,
            }
        }
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            return data.get("risk_scoring", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}. Using defaults.")
        return {}


CONFIG = _load_config()


def _get_nested(d, key, default=0):
    """Safely get nested dictionary value."""
    keys = key.split(".")
    val = d
    for k in keys:
        val = val.get(k, default) if isinstance(val, dict) else default
    return val


def _extract_privilege_escalation_score(feature_vector: dict) -> float:
    """
    Estimate privilege escalation from network behavior indicators.
    
    Proxy signals:
      - High PST (PSH) flag count → aggressive data transmission pattern
      - High FIN flag count → repeated connection attempts to different targets
      - Unusual ACK patterns → manipulation of connection state
    
    Returns normalized score [0, 1].
    """
    # Extract flag counts (handle both leading/trailing spaces and variations)
    pst_count = feature_vector.get(" PSH Flag Count", 0)
    fin_count = feature_vector.get(" FIN Flag Count", 0)
    ack_count = feature_vector.get(" ACK Flag Count", 0)
    
    # Normalization limits from config
    pst_max = _get_nested(CONFIG, "normalization_limits.pst_flag_count_max", 50)
    fin_max = _get_nested(CONFIG, "normalization_limits.fin_flag_count_max", 50)
    ack_max = _get_nested(CONFIG, "normalization_limits.ack_flag_count_max", 200)
    
    # Normalize each to [0, 1]
    pst_norm = min(1.0, pst_count / pst_max) if pst_max > 0 else 0
    fin_norm = min(1.0, fin_count / fin_max) if fin_max > 0 else 0
    ack_norm = min(1.0, ack_count / ack_max) if ack_max > 0 else 0
    
    # Weighted average (PST + FIN more indicative than ACK)
    priv_esc_score = 0.4 * pst_norm + 0.4 * fin_norm + 0.2 * ack_norm
    
    return min(1.0, priv_esc_score)


def _extract_lateral_movement_score(feature_vector: dict) -> float:
    """
    Estimate lateral movement / scanning behavior from network indicators.
    
    Proxy signals:
      - High SYN flag count → port scanning
      - High RST flag count → blocked connections / reconnaissance
      - Low data volume + many packets → scanning pattern
    
    Returns normalized score [0, 1].
    """
    syn_count = feature_vector.get(" SYN Flag Count", 0)
    rst_count = feature_vector.get(" RST Flag Count", 0)
    total_packets = (
        feature_vector.get(" Total Fwd Packets", 0) +
        feature_vector.get(" Total Backward Packets", 0)
    )
    
    # Normalization limits from config
    syn_max = _get_nested(CONFIG, "normalization_limits.syn_flag_count_max", 100)
    
    # Normalize
    syn_norm = min(1.0, syn_count / syn_max) if syn_max > 0 else 0
    rst_norm = min(1.0, rst_count / (syn_max * 0.5)) if syn_max > 0 else 0
    
    # Scanning often manifests as high syn-to-packet ratio
    packet_ratio = 0
    if total_packets > 0:
        # If >20% of packets are SYN, suggests scanning
        packet_ratio = min(1.0, (syn_count / total_packets) * 5) if total_packets > 0 else 0
    
    lateral_score = 0.5 * syn_norm + 0.3 * rst_norm + 0.2 * packet_ratio
    
    return min(1.0, lateral_score)


def _classify_risk(risk_score: float, config: dict = None) -> str:
    """
    Classify risk_score into LOW | MEDIUM | HIGH.
    
    Thresholds (tunable from config):
      score < 0.3 → LOW    (routine rate-limiting, logging)
      0.3 ≤ score < 0.7 → MEDIUM (stronger rate-limiting, MFA challenge)
      score ≥ 0.7 → HIGH   (isolate + redirect to honeypot)
    """
    if config is None:
        config = CONFIG
    
    thresh = _get_nested(config, "thresholds", {})
    low_thresh = thresh.get("low", 0.3)
    med_thresh = thresh.get("medium", 0.7)
    
    if risk_score < low_thresh:
        return "LOW"
    elif risk_score < med_thresh:
        return "MEDIUM"
    else:
        return "HIGH"


def compute_risk(anomaly_score: float, feature_vector: dict) -> dict:
    """
    Compute unified risk from multiple threat signals.
    
    Parameters
    ----------
    anomaly_score : float
        Anomaly score from Step 8 Isolation Forest [0, 1]
    feature_vector : dict
        Feature dict from Step 7, keys like " SYN Flag Count", " PSH Flag Count", etc.
    
    Returns
    -------
    dict:
        risk_score          : float [0, 1]
        risk_level          : str 'LOW' | 'MEDIUM' | 'HIGH'
        breakdown           : dict {
            'anomaly_component':             score contribution
            'privilege_escalation_component': score contribution
            'lateral_movement_component':    score contribution
        }
        timestamp           : ISO string
        source_ip           : str (if in feature_vector) or 'UNKNOWN'
    
    Example
    -------
    >>> result = compute_risk(0.82, feature_vector)
    >>> print(result['risk_level'])
    'HIGH'
    """
    # Load config
    cfg = _load_config()
    weights = _get_nested(cfg, "weights", {})
    w_anom = weights.get("anomaly_score", 0.5)
    w_priv = weights.get("privilege_escalation", 0.3)
    w_lateral = weights.get("lateral_movement", 0.2)
    
    # Clamp anomaly score just in case
    anomaly_score = np.clip(anomaly_score, 0, 1)
    
    # Extract sub-signals
    priv_esc_score = _extract_privilege_escalation_score(feature_vector)
    lateral_score = _extract_lateral_movement_score(feature_vector)
    
    # Weighted combination
    risk_score = (
        w_anom * anomaly_score +
        w_priv * priv_esc_score +
        w_lateral * lateral_score
    )
    risk_score = np.clip(risk_score, 0, 1)
    
    # Classify
    risk_level = _classify_risk(risk_score, cfg)
    
    # Prepare result
    source_ip = feature_vector.get(" Source IP", "UNKNOWN")
    
    result = {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "breakdown": {
            "anomaly_component": round(w_anom * anomaly_score, 4),
            "privilege_escalation_component": round(w_priv * priv_esc_score, 4),
            "lateral_movement_component": round(w_lateral * lateral_score, 4),
        },
        "timestamp": datetime.now().isoformat(),
        "source_ip": str(source_ip),
    }
    
    # Audit log
    logger.info(
        f"risk_score={result['risk_score']:.3f} | risk_level={risk_level} | "
        f"anomaly={anomaly_score:.3f} | priv_esc={priv_esc_score:.3f} | "
        f"lateral={lateral_score:.3f} | source_ip={source_ip}"
    )
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (run directly: python risk_scorer.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("AEGIS — risk_scorer.py smoke test (Step 10)")
    print("=" * 70)
    print()
    
    # Test vectors with different threat profiles
    test_cases = [
        {
            "name": "Low risk (benign traffic)",
            "anomaly_score": 0.1,
            "vector": {
                " Source IP": "192.168.1.100",
                " SYN Flag Count": 2,
                " FIN Flag Count": 1,
                " ACK Flag Count": 20,
                " PSH Flag Count": 5,
                " RST Flag Count": 0,
                " Total Fwd Packets": 50,
                " Total Backward Packets": 45,
            }
        },
        {
            "name": "Medium risk (suspicious activity)",
            "anomaly_score": 0.45,
            "vector": {
                " Source IP": "203.0.113.42",
                " SYN Flag Count": 30,
                " FIN Flag Count": 15,
                " ACK Flag Count": 80,
                " PSH Flag Count": 20,
                " RST Flag Count": 5,
                " Total Fwd Packets": 200,
                " Total Backward Packets": 50,
            }
        },
        {
            "name": "High risk (active attack)",
            "anomaly_score": 0.85,
            "vector": {
                " Source IP": "198.51.100.99",
                " SYN Flag Count": 95,
                " FIN Flag Count": 40,
                " ACK Flag Count": 180,
                " PSH Flag Count": 45,
                " RST Flag Count": 20,
                " Total Fwd Packets": 500,
                " Total Backward Packets": 150,
            }
        },
    ]
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"  anomaly_score: {test['anomaly_score']}")
        
        result = compute_risk(test['anomaly_score'], test['vector'])
        
        print(f"  → risk_score: {result['risk_score']}")
        print(f"  → risk_level: {result['risk_level']}")
        print(f"  → breakdown:")
        for key, val in result['breakdown'].items():
            print(f"      {key}: {val}")
        print()
    
    print("✓ risk_scorer is wired up correctly.")
