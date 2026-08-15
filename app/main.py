"""
Main application for the CO₂ Emission Predictor.

This module creates and configures the FastAPI application.

The API provides:

- REST API endpoints
- Interactive Swagger documentation
- Health monitoring
- Machine-learning CO₂ prediction
- Model metadata and evaluation metrics
- Dataset metadata and statistics
- CORS support for the React frontend
- Application startup and shutdown logging
- Production-oriented application configuration
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dataset import router as dataset_router
from app.api.routes import router


# ============================================================
# APPLICATION METADATA
# ============================================================

APP_TITLE = "CO₂ Emission Predictor API"
APP_DESCRIPTION = """
A production-oriented machine-learning REST API for predicting
vehicle CO₂ emissions.

The API provides:

- Vehicle CO₂ emission predictions
- Machine-learning model metadata
- Model evaluation metrics
- Dataset metadata and statistics
- Health monitoring
- Interactive API documentation
"""

APP_VERSION = "1.0.0"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# React/Vite development servers that are allowed to
# communicate with this FastAPI application.

ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    Manage FastAPI application startup and shutdown.

    Args:
        application:
            The FastAPI application instance.

    Yields:
        Control to the running FastAPI application.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting CO₂ Emission Predictor API..."
    )

    logger.info(
        "Application version: %s",
        APP_VERSION,
    )

    logger.info(
        "Machine-learning prediction service is ready."
    )

    logger.info(
        "Dataset metadata service is ready."
    )

    logger.info(
        "API documentation available at /docs"
    )

    logger.info(
        "============================================================"
    )

    try:
        yield

    finally:
        logger.info(
            "Shutting down CO₂ Emission Predictor API..."
        )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

# Main machine-learning router:
#
# GET  /api/health
# GET  /api/model
# POST /api/predict

app.include_router(router)


# Dataset router:
#
# GET  /dataset/metadata

app.include_router(dataset_router)


# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get(
    "/",
    tags=["System"],
    summary="API information",
    response_model=dict[str, str],
)
def root() -> dict[str, str]:
    """
    Return basic information about the CO₂ Emission Predictor API.

    Returns:
        A dictionary containing the API name, version,
        documentation locations, and available endpoints.
    """

    return {
        "message": (
            "Welcome to the CO₂ Emission Predictor API."
        ),
        "version": APP_VERSION,
        "documentation": "/docs",
        "alternative_documentation": "/redoc",
        "health": "/api/health",
        "model": "/api/model",
        "prediction_endpoint": "/api/predict",
        "dataset_metadata": "/dataset/metadata",
    }