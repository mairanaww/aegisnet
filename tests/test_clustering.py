"""
Unit tests for AegisNet Attacker Clustering Engine (cluster_attackers.py).

Validates profile assignment from cluster centroids, K-Means clustering execution,
and DBSCAN clustering on behavioral feature representations.
"""

import numpy as np
import pytest

from cluster_attackers import (
    assign_profile_names_from_centroids,
    run_dbscan,
    run_kmeans,
)

EXPECTED_PROFILES = {"Scanner", "Automated Bot", "Human Attacker", "Insider-like"}
FEATURE_NAMES = [
    "Flow Packets/s",
    "SYN Flag Count",
    "Flow Duration",
    "Packet Length Variance",
    "ACK Flag Count",
    "Total Backward Packets",
    "f7",
    "f8",
    "f9",
    "f10",
]


@pytest.fixture
def synthetic_clustering_data():
    """Generates deterministic random matrix with 100 samples and 10 features."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 10)


@pytest.fixture
def synthetic_centroids():
    """Generates deterministic 4-cluster centroid matrix for 10 features."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 10)


def test_assign_profiles_returns_all_clusters(synthetic_centroids):
    """Verify assign_profile_names_from_centroids maps all 4 clusters."""
    profiles = assign_profile_names_from_centroids(synthetic_centroids, FEATURE_NAMES)
    assert len(profiles) == 4
    assert set(profiles.keys()) == {0, 1, 2, 3}


def test_assign_profiles_unique_names(synthetic_centroids):
    """Verify every assigned cluster receives a distinct, unique profile name."""
    profiles = assign_profile_names_from_centroids(synthetic_centroids, FEATURE_NAMES)
    assert len(set(profiles.values())) == 4


def test_assign_profiles_known_names(synthetic_centroids):
    """Verify assigned profile names belong strictly to the expected profile taxonomy."""
    profiles = assign_profile_names_from_centroids(synthetic_centroids, FEATURE_NAMES)
    assert set(profiles.values()) == EXPECTED_PROFILES


def test_run_kmeans_returns_correct_shape(synthetic_clustering_data):
    """Verify run_kmeans produces a cluster label array matching input sample length."""
    km, labels, score = run_kmeans(synthetic_clustering_data, n_clusters=4)
    assert labels.shape == (synthetic_clustering_data.shape[0],)
    assert len(labels) == 100


def test_run_kmeans_correct_n_clusters(synthetic_clustering_data):
    """Verify run_kmeans partitions data into exactly the requested number of clusters."""
    km, labels, score = run_kmeans(synthetic_clustering_data, n_clusters=4)
    assert km.n_clusters == 4
    assert len(set(labels)) == 4


def test_run_dbscan_returns_labels(synthetic_clustering_data):
    """Verify run_dbscan returns a valid NumPy label array matching input sample count."""
    db, labels, score, pca = run_dbscan(synthetic_clustering_data, eps=0.8, min_samples=10)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (synthetic_clustering_data.shape[0],)
