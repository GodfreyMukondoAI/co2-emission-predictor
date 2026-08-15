"""
Production-grade API routes for the CO₂ Emission Predictor.

This module exposes the public REST API endpoints consumed by the
React frontend and other API clients.

Available endpoints:

    GET  /api/health
        Check API availability.

    GET  /api/model
        Return dynamically generated trained-model metadata,
        evaluation metrics, and dataset statistics.

    POST /api/predict
        Generate a CO₂ emission prediction.

Important:
    Model metrics and dataset statistics are NEVER hardcoded in this
    router. They are obtained from the machine-learning model management
    layer and preprocessing layer.

The router is mounted by the main FastAPI application.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.ml.model import get_model_metadata
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import prediction_service


logger = logging.getLogger(__name__)


# ============================================================
# ROUTER CONFIGURATION
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["CO₂ Prediction"],
)


# ============================================================
# HEALTH CHECK
# ============================================================


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check API health",
    description=(
        "Returns the current availability status of the "
        "CO₂ Emission Predictor API."
    ),
)
def health_check() -> dict[str, str]:
    """
    Check whether the prediction API is available.

    Returns:
        A dictionary containing the API health status.
    """

    return {
        "status": "healthy",
        "service": "CO₂ Emission Predictor",
    }


# ============================================================
# MODEL INFORMATION
# ============================================================


@router.get(
    "/model",
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description=(
        "Returns dynamically generated metadata about the trained "
        "machine-learning model, including its features, target, "
        "evaluation metrics, and dataset statistics."
    ),
)
def model_info() -> dict[str, Any]:
    """
    Return dynamically generated information about the trained model.

    The values returned by this endpoint are NOT hardcoded.

    Model metrics are loaded from the persisted metrics generated
    during model training.

    Dataset statistics are calculated from the actual cleaned
    training dataset.

    Returns:
        Dictionary containing:

        - model information
        - model version
        - algorithm
        - features
        - target
        - target unit
        - evaluation metrics
        - evaluation information
        - dataset statistics

    Raises:
        HTTPException:
            If model metadata cannot be generated.
    """

    try:
        metadata = get_model_metadata()

    except Exception as exc:
        logger.exception(
            "Failed to retrieve CO₂ model metadata."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to retrieve machine-learning model "
                "information."
            ),
        ) from exc

    return metadata


# ============================================================
# CO₂ PREDICTION
# ============================================================


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict vehicle CO₂ emissions",
    description=(
        "Generates an estimated CO₂ emission value using "
        "engine size and combined fuel consumption."
    ),
)
def predict_co2(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict CO₂ emissions for a vehicle.

    Args:
        request:
            Validated vehicle information containing engine
            size and combined fuel consumption.

    Returns:
        PredictionResponse containing:

        - predicted CO₂ emissions
        - engine size
        - fuel consumption

    Raises:
        HTTPException:
            422 if the supplied prediction inputs are invalid.

            500 if the prediction service fails unexpectedly.
    """

    try:
        prediction = prediction_service.predict(
            engine_size=request.engine_size,
            fuel_consumption_mpg=request.fuel_consumption_mpg,
        )

    except ValueError as exc:
        logger.warning(
            "Invalid prediction request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            (
                "CO₂ prediction failed | "
                "engine_size=%s | "
                "fuel_consumption_mpg=%s"
            ),
            request.engine_size,
            request.fuel_consumption_mpg,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to generate the CO₂ prediction. "
                "Please try again later."
            ),
        ) from exc

    return PredictionResponse(
        predicted_co2=round(
            prediction,
            2,
        ),
        engine_size=request.engine_size,
        fuel_consumption_mpg=request.fuel_consumption_mpg,
    )

