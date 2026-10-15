"""
Unit tests for the AegisNet Anomaly Detection Engine (anomaly_model.py).

Tests score scaling, risk band classification, signed log transformations,
DataFrame cleaning, and inference scoring with Isolation Forest.
"""

import os
import numpy as np
import pandas as pd
import pytest

from anomaly_model import (
    _raw_to_score,
    _classify_risk,
    _signed_log1p,
    _clean_df,
    score_session,
)

MODEL_EXISTS = os.path.exists("models/isolation_forest.pkl")


def test_raw_to_score_normal_range():
    """Verify that raw score 0 with symmetric lo=-1, hi=1 scales to exactly 0.5."""
    score = _raw_to_score(0.0, lo=-1.0, hi=1.0)
    assert score == 0.5


def test_raw_to_score_high_anomaly():
    """Verify that very negative raw score (anomaly in IF) gives score near 1."""
    score = _raw_to_score(-10.0, lo=-1.0, hi=1.0)
    assert score >= 0.95
    assert score <= 1.0


def test_raw_to_score_low_anomaly():
    """Verify that very positive raw score (normal in IF) gives score near 0."""
    score = _raw_to_score(10.0, lo=-1.0, hi=1.0)
    assert score <= 0.05
    assert score >= 0.0


def test_raw_to_score_clamps_to_0_1():
    """Verify score is clamped within [0, 1] for extreme raw values."""
    assert 0.0 <= _raw_to_score(-1e6, lo=-1.0, hi=1.0) <= 1.0
    assert 0.0 <= _raw_to_score(1e6, lo=-1.0, hi=1.0) <= 1.0
    assert _raw_to_score(-1e6, lo=-1.0, hi=1.0) == 1.0
    assert _raw_to_score(1e6, lo=-1.0, hi=1.0) == 0.0


def test_classify_risk_levels():
    """Verify risk classification thresholds: LOW (<0.3), MEDIUM (0.3-0.7), HIGH (>=0.7)."""
    assert _classify_risk(0.0) == "LOW"
    assert _classify_risk(0.29) == "LOW"
    assert _classify_risk(0.3) == "MEDIUM"
    assert _classify_risk(0.5) == "MEDIUM"
    assert _classify_risk(0.699) == "MEDIUM"
    assert _classify_risk(0.7) == "HIGH"
    assert _classify_risk(1.0) == "HIGH"


def test_signed_log1p_positive():
    """Verify signed log1p maps positive values to log1p(x)."""
    x = np.array([1.0, 10.0, 100.0])
    res = _signed_log1p(x)
    assert np.all(res > 0)
    assert np.allclose(res, np.log1p(x))


def test_signed_log1p_negative():
    """Verify signed log1p preserves negative sign and applies log1p(|x|)."""
    x = np.array([-1.0, -10.0, -100.0])
    res = _signed_log1p(x)
    assert np.all(res < 0)
    assert np.allclose(res, -np.log1p(np.abs(x)))


def test_signed_log1p_zero():
    """Verify signed log1p maps zero to zero."""
    x = np.array([0.0])
    res = _signed_log1p(x)
    assert res[0] == 0.0


def test_clean_df_handles_inf():
    """Verify DataFrame with positive and negative infinity is sanitized to 0.0."""
    df = pd.DataFrame({
        "feat1": [1.0, np.inf, -np.inf, 4.0],
        "feat2": [5.0, 6.0, 7.0, 8.0],
    })
    cleaned = _clean_df(df, ["feat1", "feat2"])
    assert not np.isinf(cleaned).any()
    assert not np.isnan(cleaned).any()
    assert cleaned[1, 0] == 0.0
    assert cleaned[2, 0] == 0.0


def test_clean_df_handles_nan():
    """Verify DataFrame with NaN values gets filled with 0.0."""
    df = pd.DataFrame({
        "feat1": [1.0, np.nan, 3.0],
        "feat2": [np.nan, 5.0, 6.0],
    })
    cleaned = _clean_df(df, ["feat1", "feat2"])
    assert not np.isnan(cleaned).any()
    assert cleaned[1, 0] == 0.0
    assert cleaned[0, 1] == 0.0


@pytest.mark.skipif(not MODEL_EXISTS, reason="Models not trained")
def test_score_session_returns_float(sample_flow_features):
    """Verify score_session returns a float anomaly score."""
    score = score_session(sample_flow_features, log_score=False)
    assert isinstance(score, float)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Models not trained")
def test_score_session_range(sample_flow_features):
    """Verify score_session output is bounded within [0, 1]."""
    score = score_session(sample_flow_features, log_score=False)
    assert 0.0 <= score <= 1.0


@pytest.mark.skipif(not MODEL_EXISTS, reason="Models not trained")
def test_score_session_dict_input(sample_flow_features):
    """Verify score_session accepts dictionary feature vectors."""
    score = score_session(sample_flow_features, log_score=False)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.skipif(not MODEL_EXISTS, reason="Models not trained")
def test_score_session_list_input(sample_feature_vector_list):
    """Verify score_session accepts list format feature vectors."""
    score = score_session(sample_feature_vector_list, log_score=False)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
