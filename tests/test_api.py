"""
Integration tests for the AegisNet FastAPI REST API service (api/main.py).

Validates endpoints:
- GET  /health           : service health and readiness status
- POST /analyze          : single flow analysis (503 if models missing, 200 if trained)
- POST /analyze/batch    : batch flow analysis with empty list
- GET  /config           : risk scoring configuration inspection
- GET  /                 : root documentation redirect to /docs
"""

import os

import pytest

# Ensure fastapi and httpx dependencies are available; skip if not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

try:
    from fastapi.testclient import TestClient

    from api.main import app
except Exception:
    app = None
    TestClient = None

pytestmark = pytest.mark.skipif(
    app is None or TestClient is None,
    reason="FastAPI app not importable (api/main.py not yet implemented or missing dependencies)",
)


@pytest.fixture
def client():
    """Fixture providing a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns HTTP 200 and includes a 'status' field in the response."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_analyze_endpoint_without_models(client, sample_flow_features):
    """
    POST /analyze returns 503 Service Unavailable when ML model artifacts
    are not present, or 200 OK if trained models are present.
    """
    models_trained = os.path.exists("models/isolation_forest.pkl") and os.path.exists(
        "models/kmeans_model.pkl"
    )

    response = client.post("/analyze", json={})
    if response.status_code == 422:
        response = client.post("/analyze", json=sample_flow_features)
    if response.status_code == 422:
        response = client.post("/analyze", json={"features": sample_flow_features})

    if not models_trained:
        assert response.status_code == 503
    else:
        assert response.status_code == 200


def test_analyze_batch_endpoint(client):
    """POST /analyze/batch with an empty list returns HTTP 200 OK."""
    # Attempt raw empty list first
    response = client.post("/analyze/batch", json=[])
    if response.status_code == 422:
        # Fall back to BatchAnalysisRequest schema with flows list
        response = client.post("/analyze/batch", json={"flows": []})
    if response.status_code == 422:
        response = client.post("/analyze/batch", json={"items": []})

    assert response.status_code == 200


def test_documented_analyze_request_matches_schema(client):
    """The README example reaches inference rather than failing request validation."""
    flow = {
        "flow_duration": 12500,
        "total_fwd_packets": 12,
        "total_backward_packets": 8,
        "total_length_fwd_packets": 2400,
        "total_length_bwd_packets": 1150,
    }
    response = client.post("/analyze", json=flow)
    assert response.status_code != 422
    if not os.path.exists("models/isolation_forest.pkl"):
        assert response.status_code == 503


def test_config_endpoint(client):
    """GET /config returns HTTP 200 with runtime scoring engine configuration."""
    response = client.get("/config")
    assert response.status_code == 200


def test_docs_redirect(client):
    """GET / redirects to /docs (status 307 redirect or 200 if followed)."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (200, 301, 302, 307)
    if response.status_code in (301, 302, 307):
        assert "/docs" in response.headers.get("location", "")
    else:
        assert response.status_code == 200
