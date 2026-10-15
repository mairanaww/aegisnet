"""
Pytest configuration and shared fixtures for the AegisNet test suite.

Provides representative network flow feature vectors for benign and malicious traffic,
supporting testing of anomaly detection, behavioral clustering, and risk scoring.
"""

import os
import sys
from typing import Any, Dict, List
import pytest

# Ensure project root is at the head of sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_flow_features() -> Dict[str, Any]:
    """
    Returns a realistic CICIDS-2017 feature vector representing normal/benign network traffic.
    Uses CICIDS standard column naming convention with leading spaces.
    """
    return {
        " Source IP": "192.168.1.100",
        " Destination IP": "10.0.0.5",
        " Flow Duration": 120000,
        " Total Fwd Packets": 10,
        " Total Backward Packets": 8,
        "Total Length of Fwd Packets": 1500,
        " Total Length of Bwd Packets": 2400,
        " Fwd Packet Length Max": 300,
        " Fwd Packet Length Min": 40,
        " Fwd Packet Length Mean": 150.0,
        " Bwd Packet Length Max": 600,
        " Bwd Packet Length Min": 40,
        " Bwd Packet Length Mean": 300.0,
        " Flow Bytes/s": 32500.0,
        " Flow Packets/s": 150.0,
        " Flow IAT Mean": 6500.0,
        " FIN Flag Count": 1,
        " SYN Flag Count": 2,
        " RST Flag Count": 0,
        " PSH Flag Count": 2,
        " ACK Flag Count": 18,
        " URG Flag Count": 0,
        " Average Packet Size": 216.6,
        " Packet Length Variance": 2500.0,
    }


@pytest.fixture
def malicious_flow_features() -> Dict[str, Any]:
    """
    Returns a feature vector representing an attack flow with high SYN count,
    high packet rate, elevated RST count, and high flag manipulation.
    """
    return {
        " Source IP": "198.51.100.99",
        " Destination IP": "10.0.0.5",
        " Flow Duration": 350,
        " Total Fwd Packets": 150,
        " Total Backward Packets": 10,
        "Total Length of Fwd Packets": 6000,
        " Total Length of Bwd Packets": 400,
        " Fwd Packet Length Max": 40,
        " Fwd Packet Length Min": 40,
        " Fwd Packet Length Mean": 40.0,
        " Bwd Packet Length Max": 40,
        " Bwd Packet Length Min": 40,
        " Bwd Packet Length Mean": 40.0,
        " Flow Bytes/s": 18285714.0,
        " Flow Packets/s": 457142.8,
        " Flow IAT Mean": 2.3,
        " FIN Flag Count": 60,
        " SYN Flag Count": 150,
        " RST Flag Count": 30,
        " PSH Flag Count": 60,
        " ACK Flag Count": 250,
        " URG Flag Count": 10,
        " Average Packet Size": 40.0,
        " Packet Length Variance": 0.0,
    }


@pytest.fixture
def sample_feature_vector_list(sample_flow_features: Dict[str, Any]) -> List[Any]:
    """
    Returns a list version (values only) of sample_flow_features for testing list input format.
    """
    return list(sample_flow_features.values())
