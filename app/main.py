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
- Provide CORS diagnostics
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

System diagnostics
------------------
GET  /api/cors

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

from fastapi import FastAPI, Request
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
- CORS-protected browser access
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
    ).strip().lower()

    return environment or DEFAULT_APP_ENV


# =============================================================================
# CORS CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Production frontend
# -----------------------------------------------------------------------------
#
# This is your current deployed Render frontend.
#
# Keeping this as a built-in trusted origin means the API remains accessible
# even if CORS_ORIGINS is accidentally missing from Render.
#
# CORS_ORIGINS should STILL be configured in Render because environment-based
# configuration is preferred for production.
# -----------------------------------------------------------------------------

PRODUCTION_FRONTEND_ORIGIN = (
    "https://co2-emission-predictor-frontend.onrender.com"
)


# -----------------------------------------------------------------------------
# Safe local development origins
# -----------------------------------------------------------------------------

LOCAL_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",

    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",

    # Vite preview
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


# -----------------------------------------------------------------------------
# Built-in default origins
# -----------------------------------------------------------------------------

DEFAULT_CORS_ORIGINS = [
    PRODUCTION_FRONTEND_ORIGIN,
    *LOCAL_CORS_ORIGINS,
]


# =============================================================================
# CORS HELPERS
# =============================================================================

def normalize_origin(
    origin: str,
) -> str:
    """
    Normalize a browser origin.

    Removes surrounding whitespace and trailing slash.

    Example:

        https://example.com/

    becomes:

        https://example.com
    """

    return origin.strip().rstrip("/")


def get_cors_origins() -> list[str]:
    """
    Build the final list of trusted CORS origins.

    Origins supplied through CORS_ORIGINS are combined with the built-in
    production and local development origins.

    Example:

        CORS_ORIGINS=https://example.com,https://www.example.com

    Results in a unique list containing:

        https://example.com
        https://www.example.com
        production Render frontend
        local development origins

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

        normalized = normalize_origin(origin)

        if normalized and normalized not in origins:
            origins.append(normalized)

    # -------------------------------------------------------------------------
    # Built-in trusted origins
    # -------------------------------------------------------------------------

    for origin in DEFAULT_CORS_ORIGINS:

        normalized = normalize_origin(origin)

        if normalized and normalized not in origins:
            origins.append(normalized)

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

    Wildcard origins are deliberately rejected.

    Production APIs should explicitly identify trusted browser origins instead
    of using:

        *

    This is especially important if authentication or credentials are added
    in the future.
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
            origin.startswith("http://")
            or origin.startswith("https://")
        ):
            raise RuntimeError(
                "Invalid CORS origin configured: "
                f"{origin!r}. "
                "Origins must start with http:// or https://."
            )

        if " " in origin:
            raise RuntimeError(
                "Invalid CORS origin contains whitespace: "
                f"{origin!r}"
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
    # Log every allowed origin.
    #
    # This is intentionally useful when debugging Render deployments.
    # -------------------------------------------------------------------------

    for origin in ALLOWED_ORIGINS:

        logger.info(
            "CORS allowed origin: %s",
            origin,
        )

    logger.info(
        "API documentation: /docs"
    )

    logger.info(
        "Alternative API documentation: /redoc"
    )

    logger.info(
        "System endpoints:"
    )

    logger.info(
        "  GET /"
    )

    logger.info(
        "  GET /api/cors"
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

    # -------------------------------------------------------------------------
    # Swagger
    # -------------------------------------------------------------------------
    docs_url="/docs",

    # -------------------------------------------------------------------------
    # ReDoc
    # -------------------------------------------------------------------------
    redoc_url="/redoc",

    # -------------------------------------------------------------------------
    # OpenAPI schema
    # -------------------------------------------------------------------------
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
#
# IMPORTANT:
#
# CORSMiddleware must be registered before the application starts handling
# browser requests.
#
# Explicit origins are used instead of "*".
#
# This configuration allows:
#
#   GET
#   POST
#   OPTIONS
#
# and the headers normally required by the frontend.
# =============================================================================

app.add_middleware(
    CORSMiddleware,

    # -------------------------------------------------------------------------
    # Explicit trusted browser origins
    # -------------------------------------------------------------------------
    allow_origins=ALLOWED_ORIGINS,

    # -------------------------------------------------------------------------
    # Credentials are currently disabled.
    #
    # The prediction API does not currently require browser cookies.
    # -------------------------------------------------------------------------
    allow_credentials=False,

    # -------------------------------------------------------------------------
    # HTTP methods
    # -------------------------------------------------------------------------
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    # -------------------------------------------------------------------------
    # Request headers
    # -------------------------------------------------------------------------
    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
        "Origin",
    ],

    # -------------------------------------------------------------------------
    # Cache browser preflight responses.
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


# =============================================================================
# CORS DIAGNOSTIC ENDPOINT
# =============================================================================

@app.get(
    "/api/cors",
    tags=["System"],
    summary="CORS configuration diagnostics",
)
def cors_diagnostics(
    request: Request,
) -> dict[str, object]:
    """
    Return CORS diagnostic information.

    This endpoint is particularly useful when troubleshooting browser
    requests from the deployed Render frontend.

    Example request:

        GET /api/cors

    with:

        Origin:
        https://co2-emission-predictor-frontend.onrender.com

    The endpoint reports:

    - The request Origin
    - Whether that origin is configured as trusted
    - The configured trusted origins

    It does NOT expose environment secrets.
    """

    request_origin = request.headers.get(
        "origin"
    )

    normalized_request_origin = (
        normalize_origin(request_origin)
        if request_origin
        else None
    )

    origin_allowed = (
        normalized_request_origin in ALLOWED_ORIGINS
        if normalized_request_origin
        else False
    )

    return {
        "status": "ok",

        "request_origin": normalized_request_origin,

        "origin_allowed": origin_allowed,

        "production_frontend": (
            PRODUCTION_FRONTEND_ORIGIN
        ),

        "allowed_origins": ALLOWED_ORIGINS,

        "cors_origins_environment_configured": bool(
            os.getenv("CORS_ORIGINS", "").strip()
        ),
    }


# =============================================================================
# APPLICATION HEALTH FALLBACK
# =============================================================================

@app.get(
    "/api/system/health",
    tags=["System"],
    summary="System health check",
)
def system_health() -> dict[str, object]:
    """
    Return basic API process health.

    The machine-learning-specific health endpoint remains:

        /api/health

    This endpoint provides an independent application-level health check.
    """

    return {
        "status": "healthy",

        "service": APP_TITLE,

        "version": APP_VERSION,

        "environment": get_app_environment(),

        "cors_configured": bool(
            ALLOWED_ORIGINS
        ),
    }