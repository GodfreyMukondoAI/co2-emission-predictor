
"""
============================================================================
CO₂ EMISSION PREDICTOR API
============================================================================

Main application entry point for the CO₂ Emission Predictor.

Responsibilities
----------------
- Create and configure the FastAPI application
- Register API routers
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
GET  /dataset/metadata

Environment variables
---------------------
CORS_ORIGINS

Comma-separated list of browser origins allowed to access this API.

Example:

CORS_ORIGINS=https://your-frontend.onrender.com,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173

The backend URL itself must NOT be placed in CORS_ORIGINS.

============================================================================
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


# ============================================================================
# APPLICATION METADATA
# ============================================================================

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


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

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


# ============================================================================
# CORS CONFIGURATION
# ============================================================================

def get_cors_origins() -> list[str]:
    """
    Build the list of browser origins allowed to access the API.

    Production origins should be configured through the CORS_ORIGINS
    environment variable.

    Local Vite development and preview origins are included so that
    the deployed API can also be tested from a local frontend.

    Returns:
        A normalized list of allowed browser origins.
    """

    configured_origins = os.getenv(
        "CORS_ORIGINS",
        "",
    )

    origins: list[str] = []

    # ------------------------------------------------------------------------
    # Read origins from environment
    # ------------------------------------------------------------------------

    for origin in configured_origins.split(","):
        normalized = origin.strip().rstrip("/")

        if normalized and normalized not in origins:
            origins.append(normalized)

    # ------------------------------------------------------------------------
    # Local Vite development/preview origins
    # ------------------------------------------------------------------------
    #
    # Development:
    #   http://localhost:5173
    #   http://127.0.0.1:5173
    #
    # Vite preview:
    #   http://localhost:4173
    #   http://127.0.0.1:4173
    #
    # Additional development ports are retained for flexibility.
    # ------------------------------------------------------------------------

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


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

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
        "Machine-learning prediction service is ready."
    )

    logger.info(
        "Dataset metadata service is ready."
    )

    logger.info(
        "CORS configured for %d browser origin(s).",
        len(ALLOWED_ORIGINS),
    )

    logger.info(
        "API documentation available at /docs"
    )

    logger.info(
        "Alternative API documentation available at /redoc"
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


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,

    # Explicitly allowed browser origins.
    allow_origins=ALLOWED_ORIGINS,

    # No authentication cookies are currently required by this API.
    #
    # Keeping this False is safer for the current public ML API.
    # If cookie-based authentication is introduced later, this should
    # be reconsidered together with explicit origins.
    allow_credentials=False,

    # Explicit HTTP methods used by the API.
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    # Headers required by the frontend/API client.
    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
    ],

    # Cache successful CORS preflight responses.
    max_age=3600,
)


# ============================================================================
# API ROUTES
# ============================================================================

# Main machine-learning router.
#
# Expected endpoints:
#
# GET  /api/health
# GET  /api/model
# POST /api/predict

app.include_router(router)


# Dataset router.
#
# Expected endpoint:
#
# GET /dataset/metadata

app.include_router(dataset_router)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

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
        Dictionary containing API metadata and endpoint locations.
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
        "dataset_metadata": "/dataset/metadata",
    }

