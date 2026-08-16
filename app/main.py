"""
===============================================================================
CO₂ EMISSION PREDICTOR API
===============================================================================

Main application entry point for the CO₂ Emission Predictor.

Responsibilities
----------------
- Create and configure the FastAPI application
- Register machine-learning and dataset routers
- Configure production-oriented CORS
- Provide API metadata
- Provide root/system information
- Configure application lifecycle logging
- Support local development and deployed frontends
- Expose Swagger and ReDoc documentation

Primary endpoints
-----------------
GET  /
GET  /api/health
GET  /api/model
POST /api/predict

GET  /dataset/health
GET  /dataset/metadata
GET  /dataset/records

Environment variables
---------------------
CORS_ORIGINS
    Comma-separated list of browser origins allowed to access this API.

APP_ENV
    Application environment.

Example:

CORS_ORIGINS=https://your-frontend.onrender.com,http://localhost:5173

The backend URL itself must NOT be placed in CORS_ORIGINS.

===============================================================================
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dataset import router as dataset_router
from app.api.routes import router


# =============================================================================
# APPLICATION METADATA
# =============================================================================

APP_TITLE = "CO₂ Emission Predictor API"

APP_DESCRIPTION = """
Production-oriented machine-learning REST API for predicting
vehicle CO₂ emissions.

The API provides:

- Vehicle CO₂ emission predictions
- Machine-learning model metadata
- Model evaluation information
- Dataset health monitoring
- Dataset metadata and statistics
- Actual dataset records
- Pagination and dataset search
- Interactive API documentation
"""

APP_VERSION = "1.0.0"


# =============================================================================
# LOGGING
# =============================================================================

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


# =============================================================================
# CORS CONFIGURATION
# =============================================================================

def get_cors_origins() -> list[str]:
    """
    Build the list of allowed browser origins.

    Production frontend origins should be supplied through CORS_ORIGINS.

    Example:

        CORS_ORIGINS=https://example.com,https://www.example.com

    Local Vite development and preview origins are included automatically.
    """

    configured_origins = os.getenv(
        "CORS_ORIGINS",
        "",
    )

    origins: list[str] = []

    # -------------------------------------------------------------------------
    # Environment-configured origins
    # -------------------------------------------------------------------------

    for origin in configured_origins.split(","):

        normalized = (
            origin
            .strip()
            .rstrip("/")
        )

        if normalized and normalized not in origins:
            origins.append(normalized)

    # -------------------------------------------------------------------------
    # Local development / preview origins
    # -------------------------------------------------------------------------

    local_origins = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    for origin in local_origins:

        if origin not in origins:
            origins.append(origin)

    return origins


ALLOWED_ORIGINS = get_cors_origins()


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting %s",
        APP_TITLE,
    )

    logger.info(
        "Application version: %s",
        APP_VERSION,
    )

    logger.info(
        "Environment: %s",
        os.getenv(
            "APP_ENV",
            "production",
        ),
    )

    logger.info(
        "Machine-learning API service initialized."
    )

    logger.info(
        "Dataset API service initialized."
    )

    logger.info(
        "CORS configured for %d browser origin(s).",
        len(ALLOWED_ORIGINS),
    )

    logger.info(
        "Available API documentation: /docs"
    )

    logger.info(
        "Alternative API documentation: /redoc"
    )

    logger.info(
        "Dataset endpoints:"
    )

    logger.info(
        "  GET /dataset/health"
    )

    logger.info(
        "  GET /dataset/metadata"
    )

    logger.info(
        "  GET /dataset/records"
    )

    logger.info(
        "============================================================"
    )

    try:

        yield

    finally:

        logger.info(
            "Shutting down %s...",
            APP_TITLE,
        )

        logger.info(
            "Application shutdown completed."
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=ALLOWED_ORIGINS,

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
    ],

    max_age=3600,
)


# =============================================================================
# API ROUTERS
# =============================================================================

# -----------------------------------------------------------------------------
# Machine-learning API
# -----------------------------------------------------------------------------
#
# Provided by app.api.routes
#
# Expected:
#
# GET  /api/health
# GET  /api/model
# POST /api/predict
#
# -----------------------------------------------------------------------------

app.include_router(
    router
)


# -----------------------------------------------------------------------------
# Dataset API
# -----------------------------------------------------------------------------
#
# Provided by app.api.dataset
#
# dataset.py defines:
#
#     prefix="/dataset"
#
# Therefore the following endpoints become:
#
# GET /dataset/health
# GET /dataset/metadata
# GET /dataset/records
#
# -----------------------------------------------------------------------------

app.include_router(
    dataset_router
)


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get(
    "/",
    tags=["System"],
    summary="API information",
    response_model=dict[str, str],
)
def root() -> dict[str, str]:
    """
    Return basic information about the CO₂ Emission Predictor API.
    """

    return {

        "message": (
            "Welcome to the CO₂ Emission Predictor API."
        ),

        "name": APP_TITLE,

        "version": APP_VERSION,

        "status": "operational",

        "documentation": "/docs",

        "alternative_documentation": "/redoc",

        "health": "/api/health",

        "model": "/api/model",

        "prediction_endpoint": "/api/predict",

        "dataset_health": "/dataset/health",

        "dataset_metadata": "/dataset/metadata",

        "dataset_records": "/dataset/records",
    }