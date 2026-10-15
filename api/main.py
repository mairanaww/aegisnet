"""
AegisNet API — Main Application Entrypoint
FastAPI application providing REST endpoints for AI-powered network intrusion detection,
flow analysis, behavior clustering, risk scoring, model retraining, and configuration management.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Union

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.models import (
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    ErrorResponse,
    FlowFeatures,
    HealthResponse,
    RetrainRequest,
    RetrainResponse,
)

# ── Logging setup ────────────────────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "api.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("aegisnet.api")

# ── Paths and Model Artifact Configuration ──────────────────────────────────
CONFIG_PATH = os.path.join(PROJECT_ROOT, "risk_scorer_config.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

REQUIRED_MODEL_FILES = [
    os.path.join(MODELS_DIR, "isolation_forest.pkl"),
    os.path.join(MODELS_DIR, "scaler.pkl"),
    os.path.join(MODELS_DIR, "feature_columns.pkl"),
    os.path.join(MODELS_DIR, "score_stats.pkl"),
    os.path.join(MODELS_DIR, "kmeans_model.pkl"),
    os.path.join(MODELS_DIR, "cluster_scaler.pkl"),
    os.path.join(MODELS_DIR, "profile_map.pkl"),
]

START_TIME = time.time()


def check_models_exist() -> bool:
    """Check if all required trained model files exist on disk."""
    return all(os.path.exists(path) for path in REQUIRED_MODEL_FILES)


def get_missing_models() -> List[str]:
    """Return a list of missing model file names."""
    return [os.path.basename(path) for path in REQUIRED_MODEL_FILES if not os.path.exists(path)]


# ── FastAPI Application ──────────────────────────────────────────────────────
app = FastAPI(
    title="AegisNet API",
    description=(
        "AI-powered network intrusion detection and response API. "
        "Provides real-time network flow anomaly scoring, attacker behavioral profiling, "
        "and dynamic risk assessment."
    ),
    version="1.0.0",
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle Events ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """Startup event handler: record server start time and check model status."""
    global START_TIME
    START_TIME = time.time()
    logger.info("Starting AegisNet API v1.0.0...")

    if check_models_exist():
        logger.info("All required AegisNet model artifacts are present on disk.")
        try:
            import anomaly_model
            anomaly_model._load_artifacts()
            logger.info("Model artifacts loaded successfully into cache.")
        except Exception as exc:
            logger.warning("Could not pre-cache model artifacts at startup: %s", exc)
    else:
        missing = get_missing_models()
        logger.warning(
            "AegisNet models are not trained yet! Missing artifacts: %s. "
            "Inference endpoints (/analyze, /analyze/batch) will return 503 until models are trained.",
            missing,
        )


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get(
    "/",
    include_in_schema=False,
    summary="Redirect to interactive API documentation",
)
async def root() -> RedirectResponse:
    """Redirect root endpoint to interactive API documentation (/docs)."""
    return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="System health and model readiness check",
    responses={
        200: {"model": HealthResponse, "description": "System health information"},
    },
)
async def health() -> HealthResponse:
    """
    Check API health and status of underlying detection and clustering models.
    """
    models_loaded = check_models_exist()
    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        version="1.0.0",
        models_loaded=models_loaded,
        uptime_seconds=uptime,
    )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze a single network flow session",
    responses={
        200: {"model": AnalysisResponse, "description": "Analysis result"},
        503: {"model": ErrorResponse, "description": "Models not trained or unavailable"},
        500: {"model": ErrorResponse, "description": "Internal error during analysis"},
    },
)
async def analyze(flow: FlowFeatures) -> AnalysisResponse:
    """
    Analyze a network flow session:
    1. Score anomaly using Isolation Forest.
    2. Classify attacker profile using behavioral clustering.
    3. Compute combined risk score and classification level.
    """
    if not check_models_exist():
        missing = get_missing_models()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detection models are not trained yet. Missing artifacts: {missing}. Call /retrain or train models first.",
        )

    try:
        from ai_engine import analyze_session

        fv = flow.to_cicids_dict()
        result = analyze_session(fv)
        return AnalysisResponse(**result)
    except FileNotFoundError as fnf:
        logger.error("Model file missing during inference: %s", fnf)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model files not available: {str(fnf)}",
        )
    except Exception as exc:
        logger.exception("Error processing flow analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(exc)}",
        )


@app.post(
    "/analyze/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze a batch of network flow sessions",
    responses={
        200: {"model": BatchAnalysisResponse, "description": "Batch analysis results"},
        503: {"model": ErrorResponse, "description": "Models not trained or unavailable"},
        500: {"model": ErrorResponse, "description": "Internal error during batch analysis"},
    },
)
async def analyze_batch(
    batch: Union[BatchAnalysisRequest, List[FlowFeatures]] = Body(...)
) -> BatchAnalysisResponse:
    """
    Analyze multiple network flows in a single batch request and return timing.
    Supports either a BatchAnalysisRequest object ({"flows": [...]}) or a raw list of FlowFeatures ([...]).
    """
    flows = batch.flows if isinstance(batch, BatchAnalysisRequest) else batch
    if not flows:
        return BatchAnalysisResponse(results=[], processing_time_ms=0.0)

    if not check_models_exist():
        missing = get_missing_models()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detection models are not trained yet. Missing artifacts: {missing}. Call /retrain or train models first.",
        )

    start_time = time.perf_counter()
    results: List[AnalysisResponse] = []

    try:
        from ai_engine import analyze_session

        for flow in flows:
            fv = flow.to_cicids_dict()
            res = analyze_session(fv)
            results.append(AnalysisResponse(**res))
    except FileNotFoundError as fnf:
        logger.error("Model file missing during batch analysis: %s", fnf)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model files not available: {str(fnf)}",
        )
    except Exception as exc:
        logger.exception("Error during batch flow analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(exc)}",
        )

    duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return BatchAnalysisResponse(
        results=results,
        processing_time_ms=duration_ms,
    )


@app.post(
    "/retrain",
    response_model=RetrainResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Management"],
    summary="Trigger model retraining cycle",
    responses={
        200: {"model": RetrainResponse, "description": "Retrain execution status"},
    },
)
async def retrain(request: RetrainRequest = RetrainRequest()) -> RetrainResponse:
    """
    Trigger an automated retraining cycle using honeypot attack data and baseline traffic:
    - Retrains Isolation Forest model.
    - Retrains K-Means behavioral clustering.
    - Validates against benign baselines before swapping models.
    """
    try:
        from retrain import retrain_cycle, RETRAIN_METRICS

        success = retrain_cycle(
            dry_run=request.dry_run,
            min_samples=request.min_samples,
        )

        metrics = None
        if os.path.exists(RETRAIN_METRICS):
            try:
                with open(RETRAIN_METRICS, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        metrics = json.loads(lines[-1])
            except Exception as e:
                logger.warning("Could not parse latest retrain metrics from log: %s", e)

        if success:
            msg = (
                "Retrain cycle completed successfully."
                if not request.dry_run
                else "Dry-run retrain cycle completed successfully without updating active models."
            )
        else:
            msg = "Retrain cycle finished without updating models (e.g. insufficient attack samples or validation failed)."

        return RetrainResponse(
            success=success,
            message=msg,
            metrics=metrics,
        )
    except Exception as exc:
        logger.exception("Unexpected failure during model retraining: %s", exc)
        return RetrainResponse(
            success=False,
            message=f"Retrain cycle encountered an error: {str(exc)}",
            metrics=None,
        )


@app.get(
    "/config",
    tags=["Configuration"],
    summary="Get current risk scoring configuration",
    responses={
        200: {"description": "Current configuration JSON"},
        404: {"model": ErrorResponse, "description": "Configuration file not found"},
        500: {"model": ErrorResponse, "description": "Error reading configuration"},
    },
)
async def get_config() -> Dict[str, Any]:
    """
    Retrieve contents of risk_scorer_config.json.
    """
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration file not found at {CONFIG_PATH}",
        )

    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        logger.exception("Failed to read configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read configuration: {str(exc)}",
        )


@app.put(
    "/config",
    tags=["Configuration"],
    summary="Update and reload risk scoring configuration",
    responses={
        200: {"description": "Configuration updated and reloaded"},
        400: {"model": ErrorResponse, "description": "Invalid configuration payload"},
        500: {"model": ErrorResponse, "description": "Error updating configuration"},
    },
)
async def update_config(new_config: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Accept a new configuration dictionary, write it to risk_scorer_config.json,
    and reload the risk_scorer module configuration in-memory.
    """
    if not isinstance(new_config, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must be a JSON dictionary.",
        )

    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(new_config, f, indent=2)

        # Reload configuration in risk_scorer module
        try:
            import risk_scorer
            risk_scorer.CONFIG = risk_scorer._load_config()
            logger.info("risk_scorer module configuration reloaded successfully.")
        except Exception as e:
            logger.warning("Could not reload risk_scorer in-memory configuration: %s", e)

        return {
            "status": "success",
            "message": "Configuration updated and reloaded successfully.",
            "config": new_config,
        }
    except Exception as exc:
        logger.exception("Failed updating configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(exc)}",
        )
