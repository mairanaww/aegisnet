"""
Unit tests for the AegisNet Risk Scoring Engine (risk_scorer.py).

Validates threshold classification, feature extraction heuristics
(privilege escalation and lateral movement), and composite risk calculation.
"""

import pytest

from risk_scorer import (
    _classify_risk,
    _extract_lateral_movement_score,
    _extract_privilege_escalation_score,
    compute_risk,
)


def test_classify_risk_low():
    """Verify that scores below 0.3 are classified as LOW."""
    assert _classify_risk(0.0) == "LOW"
    assert _classify_risk(0.15) == "LOW"
    assert _classify_risk(0.29) == "LOW"


def test_classify_risk_medium():
    """Verify that scores from 0.3 up to (not including) 0.7 are classified as MEDIUM."""
    assert _classify_risk(0.3) == "MEDIUM"
    assert _classify_risk(0.5) == "MEDIUM"
    assert _classify_risk(0.69) == "MEDIUM"


def test_classify_risk_high():
    """Verify that scores 0.7 and above are classified as HIGH."""
    assert _classify_risk(0.7) == "HIGH"
    assert _classify_risk(0.85) == "HIGH"
    assert _classify_risk(1.0) == "HIGH"


def test_classify_risk_boundary_low():
    """Verify boundary condition at score = 0.3: returns MEDIUM, not LOW."""
    assert _classify_risk(0.3) == "MEDIUM"
    assert _classify_risk(0.2999) == "LOW"


def test_classify_risk_boundary_high():
    """Verify boundary condition at score = 0.7: returns HIGH, not MEDIUM."""
    assert _classify_risk(0.7) == "HIGH"
    assert _classify_risk(0.6999) == "MEDIUM"


def test_compute_risk_returns_required_keys(sample_flow_features):
    """Verify compute_risk returns all required output keys and nested breakdown keys."""
    result = compute_risk(0.5, sample_flow_features)

    required_keys = {"risk_score", "risk_level", "breakdown", "timestamp", "source_ip"}
    assert required_keys.issubset(result.keys())

    breakdown_keys = {
        "anomaly_component",
        "privilege_escalation_component",
        "lateral_movement_component",
    }
    assert breakdown_keys.issubset(result["breakdown"].keys())


def test_compute_risk_score_range(sample_flow_features):
    """Verify computed risk_score is bounded within [0, 1]."""
    result = compute_risk(0.5, sample_flow_features)
    assert 0.0 <= result["risk_score"] <= 1.0


def test_compute_risk_low_anomaly(sample_flow_features):
    """Verify low anomaly score (0.1) paired with benign features yields LOW risk."""
    result = compute_risk(0.1, sample_flow_features)
    assert result["risk_level"] == "LOW"
    assert result["risk_score"] < 0.3


def test_compute_risk_high_anomaly(malicious_flow_features):
    """Verify high anomaly score (0.9) paired with attack features yields HIGH risk."""
    result = compute_risk(0.9, malicious_flow_features)
    assert result["risk_level"] == "HIGH"
    assert result["risk_score"] >= 0.7


def test_compute_risk_breakdown_sums(sample_flow_features):
    """Verify the breakdown component contributions sum up to the total risk_score."""
    result = compute_risk(0.55, sample_flow_features)
    breakdown_sum = sum(result["breakdown"].values())
    assert breakdown_sum == pytest.approx(result["risk_score"], abs=1e-3)


def test_privilege_escalation_zero():
    """Verify privilege escalation score is 0 when all relevant flags are 0."""
    fv = {
        " PSH Flag Count": 0,
        " FIN Flag Count": 0,
        " ACK Flag Count": 0,
    }
    score = _extract_privilege_escalation_score(fv)
    assert score == 0.0


def test_privilege_escalation_max():
    """Verify privilege escalation score approaches 1 when flags reach or exceed normalization limits."""
    fv = {
        " PSH Flag Count": 100,  # limit is 50
        " FIN Flag Count": 100,  # limit is 50
        " ACK Flag Count": 300,  # limit is 200
    }
    score = _extract_privilege_escalation_score(fv)
    assert pytest.approx(score, abs=1e-5) == 1.0


def test_lateral_movement_zero():
    """Verify lateral movement score is 0 when all relevant counters are 0."""
    fv = {
        " SYN Flag Count": 0,
        " RST Flag Count": 0,
        " Total Fwd Packets": 0,
        " Total Backward Packets": 0,
    }
    score = _extract_lateral_movement_score(fv)
    assert score == 0.0


def test_lateral_movement_high_syn():
    """Verify high SYN count with port scan pattern produces a high lateral movement score."""
    fv = {
        " SYN Flag Count": 150,
        " RST Flag Count": 40,
        " Total Fwd Packets": 150,
        " Total Backward Packets": 10,
    }
    score = _extract_lateral_movement_score(fv)
    assert score >= 0.8
    assert score <= 1.0


def test_compute_risk_source_ip_passthrough(sample_flow_features):
    """Verify source IP in feature vector is passed through to output result."""
    sample_flow_features[" Source IP"] = "192.168.10.45"
    result = compute_risk(0.2, sample_flow_features)
    assert result["source_ip"] == "192.168.10.45"

    # Test default fallback when missing
    res_unknown = compute_risk(0.2, {})
    assert res_unknown["source_ip"] == "UNKNOWN"
