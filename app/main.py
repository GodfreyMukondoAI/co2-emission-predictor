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
- Allow the deployed Render frontend to access the API
- Support local Vite development
- Provide API metadata
- Provide root/system information
- Configure application lifecycle logging
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

    Production example:

        CORS_ORIGINS=https://co2-emission-predictor-frontend.onrender.com

    Multiple origins:

        CORS_ORIGINS=https://example.com,https://www.example.com

    IMPORTANT:
        The backend URL must NOT be placed in CORS_ORIGINS.

APP_ENV
    Application environment.

    Example:

        APP_ENV=production

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
- Dataset pagination
- Dataset search
- Interactive API documentation
"""

APP_VERSION = "1.0.0"

DEFAULT_APP_ENV = "production"


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
# ENVIRONMENT HELPERS
# =============================================================================

def get_app_environment() -> str:
    """
    Return the configured application environment.

    Defaults to production.
    """

    environment = os.getenv(
        "APP_ENV",
        DEFAULT_APP_ENV,
    ).strip()

    return environment or DEFAULT_APP_ENV


# =============================================================================
# CORS CONFIGURATION
# =============================================================================

# These origins are safe for local development and local Vite preview.
#
# The deployed Render frontend is also included as a built-in production
# fallback so that the API remains accessible even if CORS_ORIGINS was
# accidentally omitted from the Render environment variables.
#
# CORS_ORIGINS should still be configured in Render because environment-based
# configuration is preferred for production deployments.

DEFAULT_CORS_ORIGINS = [
    # -------------------------------------------------------------------------
    # Production Render frontend
    # -------------------------------------------------------------------------
    "https://co2-emission-predictor-frontend.onrender.com",

    # -------------------------------------------------------------------------
    # Local Vite development
    # -------------------------------------------------------------------------
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",

    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",

    # -------------------------------------------------------------------------
    # Vite production preview
    # -------------------------------------------------------------------------
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def normalize_origin(
    origin: str,
) -> str:
    """
    Normalize a browser origin.

    Removes surrounding whitespace and trailing slashes.

    Example:

        https://example.com/

    becomes:

        https://example.com
    """

    return (
        origin
        .strip()
        .rstrip("/")
    )


def get_cors_origins() -> list[str]:
    """
    Build the final list of allowed CORS origins.

    Origins supplied through CORS_ORIGINS are combined with the safe
    built-in development and production origins.

    Example:

        CORS_ORIGINS=https://example.com,https://www.example.com

    Results in:

        [
            "https://example.com",
            "https://www.example.com",
            ...
        ]

    The backend URL itself must never be used as a browser origin.
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

        normalized = normalize_origin(
            origin
        )

        if (
            normalized
            and normalized not in origins
        ):
            origins.append(
                normalized
            )

    # -------------------------------------------------------------------------
    # Built-in safe origins
    # -------------------------------------------------------------------------

    for origin in DEFAULT_CORS_ORIGINS:

        normalized = normalize_origin(
            origin
        )

        if (
            normalized
            and normalized not in origins
        ):
            origins.append(
                normalized
            )

    return origins


ALLOWED_ORIGINS = get_cors_origins()


# =============================================================================
# CORS VALIDATION
# =============================================================================

def validate_cors_configuration(
    origins: list[str],
) -> None:
    """
    Validate the CORS configuration at startup.

    This intentionally rejects wildcard origins.

    Production APIs should explicitly identify trusted browser origins instead
    of using:

        *

    This is especially important when the API is eventually extended with
    authenticated requests.
    """

    if not origins:
        raise RuntimeError(
            "No CORS origins are configured. "
            "Configure CORS_ORIGINS or provide safe default origins."
        )

    for origin in origins:

        if origin == "*":
            raise RuntimeError(
                "Wildcard CORS origin '*' is not allowed. "
                "Configure explicit frontend origins."
            )

        if not (
            origin.startswith(
                "http://"
            )
            or origin.startswith(
                "https://"
            )
        ):
            raise RuntimeError(
                "Invalid CORS origin configured: "
                f"{origin!r}. "
                "Origins must start with http:// or https://."
            )


validate_cors_configuration(
    ALLOWED_ORIGINS
)


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

    environment = get_app_environment()

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
        environment,
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

    # -------------------------------------------------------------------------
    # Log each configured origin.
    #
    # This is extremely useful when debugging Render CORS deployments.
    # -------------------------------------------------------------------------

    for origin in ALLOWED_ORIGINS:

        logger.info(
            "CORS allowed origin: %s",
            origin,
        )

    logger.info(
        "Available API documentation: /docs"
    )

    logger.info(
        "Alternative API documentation: /redoc"
    )

    logger.info(
        "Machine-learning endpoints:"
    )

    logger.info(
        "  GET /api/health"
    )

    logger.info(
        "  GET /api/model"
    )

    logger.info(
        "  POST /api/predict"
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

    # -------------------------------------------------------------------------
    # Explicit trusted browser origins
    # -------------------------------------------------------------------------
    allow_origins=ALLOWED_ORIGINS,

    # -------------------------------------------------------------------------
    # We currently do not use browser cookies for the prediction API.
    #
    # Keeping this False avoids unnecessarily broad credential behavior.
    # -------------------------------------------------------------------------
    allow_credentials=False,

    # -------------------------------------------------------------------------
    # Supported HTTP methods
    # -------------------------------------------------------------------------
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    # -------------------------------------------------------------------------
    # Supported request headers
    # -------------------------------------------------------------------------
    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
    ],

    # -------------------------------------------------------------------------
    # Browser preflight cache
    # -------------------------------------------------------------------------
    max_age=3600,
)


# =============================================================================
# API ROUTERS
# =============================================================================

# -----------------------------------------------------------------------------
# MACHINE-LEARNING API
# -----------------------------------------------------------------------------
#
# Provided by:
#
#     app.api.routes
#
# Expected endpoints:
#
#     GET  /api/health
#     GET  /api/model
#     POST /api/predict
#
# -----------------------------------------------------------------------------

app.include_router(
    router
)


# -----------------------------------------------------------------------------
# DATASET API
# -----------------------------------------------------------------------------
#
# Provided by:
#
#     app.api.dataset
#
# Expected endpoints:
#
#     GET /dataset/health
#     GET /dataset/metadata
#     GET /dataset/records
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
        "environment": get_app_environment(),
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