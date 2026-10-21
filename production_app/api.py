"""FastAPI application for serving the Breast Cancer classification pipeline.

Exposes endpoints for service health verification and real-time model inference.
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from production_app.config import get_config
from production_app.logging_config import get_logger, setup_logging
from production_app.model_service import get_model, is_model_loaded, predict
from production_app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    Attempts to load the serialized pipeline artifact once into memory at
    startup to avoid repeated disk reads during inference requests.
    """
    cfg = get_config()
    setup_logging(cfg.log_level)
    logger.info("Initializing Breast Cancer Wisconsin Classification API...")

    try:
        get_model()
        logger.info("Model pipeline loaded and ready for inference.")
    except Exception as exc:
        logger.error(
            "Model artifact could not be loaded at startup: %s. "
            "Service will report unhealthy until a valid model is available.",
            exc,
        )

    yield

    logger.info("Shutting down Breast Cancer Wisconsin Classification API.")


app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "Production REST API serving a trained scikit-learn pipeline for the "
        "Breast Cancer Wisconsin Diagnostic dataset. Provides validated inference "
        "and health monitoring endpoints without retraining."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


def _sanitize_validation_error(obj: Any) -> Any:
    """Recursively sanitize error objects so non-finite floats serialize cleanly."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_validation_error(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_validation_error(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic request validation errors with sanitized JSON serialization.

    Ensures that non-finite numbers (NaN, Inf) provided by clients produce
    clean HTTP 422 JSON responses without serialization failures.
    """
    sanitized = [_sanitize_validation_error(err) for err in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": sanitized},
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root path to interactive OpenAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description=(
        "Reports the operational health of the API service and the readiness "
        "of the underlying machine learning model artifact."
    ),
    responses={
        200: {
            "description": "Service is healthy; model artifact is loaded and ready.",
            "model": HealthResponse,
        },
        503: {
            "description": "Service is unhealthy; model artifact is unavailable.",
            "model": HealthResponse,
        },
    },
)
def health_check() -> JSONResponse:
    """Verify service health and model artifact readiness.

    Returns HTTP 200 if the model pipeline is loaded and ready for inference.
    Returns HTTP 503 if the model artifact cannot be found or loaded.
    """
    cfg = get_config()
    model_ready = is_model_loaded()

    if not model_ready:
        try:
            get_model()
            model_ready = True
        except Exception as exc:
            logger.warning("Health check failed - model is unavailable: %s", exc)
            model_ready = False

    if model_ready:
        response_data = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path=str(cfg.model_path),
            detail="Model artifact loaded and ready for inference.",
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data.model_dump(),
        )

    response_data = HealthResponse(
        status="unhealthy",
        model_loaded=False,
        model_path=str(cfg.model_path),
        detail="Model artifact is unavailable or cannot be loaded.",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response_data.model_dump(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Tumor Diagnosis",
    description=(
        "Accepts a single observation of 30 diagnostic cell nuclei features, "
        "validates input constraints, and returns the predicted diagnostic class, "
        "readable label, and class probabilities."
    ),
    responses={
        200: {
            "description": "Successful diagnostic prediction.",
            "model": PredictionResponse,
        },
        422: {
            "description": (
                "Validation error: missing, non-numeric, or non-finite features."
            )
        },
        503: {"description": "Model artifact is unavailable or uninitialized."},
    },
)
def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Perform real-time diagnostic classification on a single sample.

    Args:
        payload: Validated observation matching the 30 cell feature schema.

    Returns:
        Structured prediction response with class label and probabilities.

    Raises:
        HTTPException(503): If the model artifact cannot be loaded.
        HTTPException(500): If an internal inference error occurs.
    """
    try:
        pipeline = get_model()
    except Exception as exc:
        logger.error("Inference request failed - model unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model service is currently unavailable. "
                "Ensure the model artifact is trained and accessible."
            ),
        ) from None

    try:
        # Convert validated Pydantic model to strictly ordered DataFrame
        input_df = payload.to_dataframe()

        # Delegate inference to the existing model service
        raw_result: dict[str, Any] = predict(pipeline=pipeline, input_data=input_df)

        pred_class = int(raw_result["predictions"][0])
        class_label = str(raw_result["class_labels"][0])

        probabilities_dict: dict[str, float] | None = None
        if "probabilities" in raw_result and raw_result["probabilities"]:
            cfg = get_config()
            raw_probs = raw_result["probabilities"][0]
            classes = getattr(pipeline, "classes_", [0, 1])
            probabilities_dict = {
                cfg.class_names.get(int(cls_id), str(cls_id)): round(float(prob), 6)
                for cls_id, prob in zip(classes, raw_probs)
            }

        return PredictionResponse(
            predicted_class=pred_class,
            class_label=class_label,
            probabilities=probabilities_dict,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during inference execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing inference on the input features.",
        ) from None
