"""
===============================================================================
CO₂ EMISSION PREDICTOR - DATASET API
===============================================================================

Production-ready FastAPI routes for inspecting the dataset used by the
CO₂-emission machine-learning prediction pipeline.

Endpoint
--------
GET /dataset/metadata

The endpoint dynamically reads the configured CSV dataset and generates
statistics from the actual data.

Canonical model
---------------
Features:
    - ENGINESIZE
    - FUELCONSUMPTION_COMB_MPG

Target:
    - CO2EMISSIONS

Important
---------
Dataset statistics are NEVER hardcoded.

Environment variables
---------------------
CO2_DATASET_DIRECTORY
    Optional dataset directory.
    Defaults to <project-root>/data

CO2_DATASET_FILENAMES
    Optional comma-separated list of supported filenames.

    Example:
        CO2_DATASET_FILENAMES=FuelConsumption.csv,co2_emissions.csv

CO2_DATASET_CACHE_TTL_SECONDS
    Optional cache lifetime.

    Defaults to 300 seconds.

The cache is automatically invalidated when the dataset file's modification
time or size changes.

===============================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

FEATURE_COLUMNS: tuple[str, ...] = (
    "ENGINESIZE",
    "FUELCONSUMPTION_COMB_MPG",
)

TARGET_COLUMN = "CO2EMISSIONS"

TARGET_UNIT = "g/km"

MODEL_NAME = "Multiple Linear Regression"

MODEL_DESCRIPTION = (
    "A multiple linear regression model that uses engine size "
    "and combined fuel consumption in MPG to predict vehicle "
    "CO₂ emissions in grams per kilometre."
)


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# Expected structure:
#
# project-root/
# ├── app/
# │   └── routes/
# │       └── dataset.py
# └── data/

PROJECT_ROOT = CURRENT_FILE.parents[2]

DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"


def get_data_directory() -> Path:
    """
    Return the configured dataset directory.

    The path can be overridden using CO2_DATASET_DIRECTORY.
    """

    configured_directory = os.getenv(
        "CO2_DATASET_DIRECTORY"
    )

    if configured_directory:
        return Path(configured_directory).expanduser().resolve()

    return DEFAULT_DATA_DIRECTORY


def get_cache_ttl_seconds() -> int:
    """
    Return the dataset metadata cache lifetime.

    Defaults to 300 seconds.

    Invalid or negative values fall back to the default.
    """

    raw_value = os.getenv(
        "CO2_DATASET_CACHE_TTL_SECONDS",
        "300",
    )

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            (
                "Invalid CO2_DATASET_CACHE_TTL_SECONDS=%r. "
                "Using default value of 300 seconds."
            ),
            raw_value,
        )
        return 300

    if value < 0:
        logger.warning(
            (
                "Negative CO2_DATASET_CACHE_TTL_SECONDS=%r. "
                "Using default value of 300 seconds."
            ),
            raw_value,
        )
        return 300

    return value


# =============================================================================
# SUPPORTED DATASET FILENAMES
# =============================================================================

DEFAULT_DATASET_FILENAMES: tuple[str, ...] = (
    "FuelConsumption.csv",
    "fuel_consumption.csv",
    "CO2 Emissions.csv",
    "co2_emissions.csv",
    "co2_emission.csv",
    "CO2_Emissions.csv",
)


def get_dataset_filenames() -> tuple[str, ...]:
    """
    Return configured dataset filenames.

    Environment variable example:

        CO2_DATASET_FILENAMES=FuelConsumption.csv,co2_emissions.csv
    """

    configured = os.getenv(
        "CO2_DATASET_FILENAMES"
    )

    if not configured:
        return DEFAULT_DATASET_FILENAMES

    filenames = tuple(
        filename.strip()
        for filename in configured.split(",")
        if filename.strip()
    )

    if not filenames:
        logger.warning(
            "CO2_DATASET_FILENAMES was empty. Using defaults."
        )
        return DEFAULT_DATASET_FILENAMES

    return filenames


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class DatasetFeature(BaseModel):
    """Description of a model feature or target column."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    column: str
    role: str
    unit: str
    description: str


class DatasetStatistics(BaseModel):
    """Descriptive statistics for a numeric dataset column."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class DatasetStatisticsCard(BaseModel):
    """Human-readable statistic used by the frontend."""

    label: str
    value: str
    description: str
    type: str


class DatasetModelInformation(BaseModel):
    """Machine-learning model metadata."""

    name: str
    description: str
    features: list[str]
    target: str
    targetUnit: str


class DatasetIntegrity(BaseModel):
    """Dataset file integrity and storage information."""

    fileSizeBytes: int
    sha256: str
    modifiedAt: str | None = None


class DatasetMetadataResponse(BaseModel):
    """
    Complete dataset metadata response.
    """

    datasetName: str
    title: str
    description: str

    recordCount: int
    columnCount: int
    columnNames: list[str]

    missingValues: int

    validRecordCount: int
    invalidRecordCount: int

    featureCount: int
    targetCount: int

    validationPercentage: float

    features: list[DatasetFeature]

    featureStatistics: dict[str, DatasetStatistics]

    targetStatistics: DatasetStatistics

    statistics: list[DatasetStatisticsCard]

    model: DatasetModelInformation

    lastUpdated: str | None = None

    integrity: DatasetIntegrity


# =============================================================================
# CACHE
# =============================================================================

_CACHE_LOCK = Lock()

_cached_metadata: DatasetMetadataResponse | None = None

_cached_dataset_path: Path | None = None

_cached_mtime_ns: int | None = None

_cached_file_size: int | None = None

_cached_at: float | None = None


def clear_dataset_cache() -> None:
    """
    Clear the in-process dataset metadata cache.
    """

    global \
        _cached_metadata, \
        _cached_dataset_path, \
        _cached_mtime_ns, \
        _cached_file_size, \
        _cached_at

    with _CACHE_LOCK:
        _cached_metadata = None
        _cached_dataset_path = None
        _cached_mtime_ns = None
        _cached_file_size = None
        _cached_at = None

    logger.info("CO₂ dataset metadata cache cleared.")


def get_dataset_signature(
    dataset_path: Path,
) -> tuple[int, int]:
    """
    Return a lightweight dataset signature.

    The signature consists of:
        - modification time in nanoseconds
        - file size in bytes
    """

    try:
        stat = dataset_path.stat()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The CO₂ emissions dataset could not be found.",
        ) from exc
    except OSError as exc:
        logger.exception(
            "Unable to inspect dataset file."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The CO₂ emissions dataset could not be accessed.",
        ) from exc

    return stat.st_mtime_ns, stat.st_size


def is_cache_valid(
    dataset_path: Path,
    mtime_ns: int,
    file_size: int,
) -> bool:
    """
    Determine whether cached metadata can be reused.
    """

    if _cached_metadata is None:
        return False

    if _cached_dataset_path != dataset_path:
        return False

    if _cached_mtime_ns != mtime_ns:
        return False

    if _cached_file_size != file_size:
        return False

    if _cached_at is None:
        return False

    import time

    age = time.monotonic() - _cached_at

    return age <= get_cache_ttl_seconds()


def get_cached_metadata(
    dataset_path: Path,
    mtime_ns: int,
    file_size: int,
) -> DatasetMetadataResponse | None:
    """
    Return cached metadata when it is still valid.
    """

    with _CACHE_LOCK:
        if is_cache_valid(
            dataset_path,
            mtime_ns,
            file_size,
        ):
            return _cached_metadata

    return None


def store_cached_metadata(
    dataset_path: Path,
    mtime_ns: int,
    file_size: int,
    metadata: DatasetMetadataResponse,
) -> None:
    """
    Store generated metadata in the process-local cache.
    """

    global \
        _cached_metadata, \
        _cached_dataset_path, \
        _cached_mtime_ns, \
        _cached_file_size, \
        _cached_at

    import time

    with _CACHE_LOCK:
        _cached_metadata = metadata
        _cached_dataset_path = dataset_path
        _cached_mtime_ns = mtime_ns
        _cached_file_size = file_size
        _cached_at = time.monotonic()


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def find_dataset() -> Path | None:
    """
    Locate the supported CO₂ emissions dataset.

    Preferred filenames are checked first. A case-insensitive fallback
    is then performed.
    """

    data_directory = get_data_directory()

    logger.debug(
        "Searching for CO₂ dataset in configured directory."
    )

    if not data_directory.exists():
        logger.error(
            "Configured dataset directory does not exist."
        )
        return None

    if not data_directory.is_dir():
        logger.error(
            "Configured dataset path is not a directory."
        )
        return None

    filenames = get_dataset_filenames()

    # -------------------------------------------------------------------------
    # Preferred filenames
    # -------------------------------------------------------------------------

    for filename in filenames:
        dataset_path = data_directory / filename

        if dataset_path.is_file():
            logger.info(
                "CO₂ dataset found: %s",
                dataset_path.name,
            )
            return dataset_path

    # -------------------------------------------------------------------------
    # Case-insensitive fallback
    # -------------------------------------------------------------------------

    try:
        available_files = sorted(
            (
                path
                for path in data_directory.iterdir()
                if path.is_file()
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        logger.exception(
            "Unable to inspect configured dataset directory."
        )
        return None

    expected_names = {
        filename.lower()
        for filename in filenames
    }

    for file_path in available_files:
        if file_path.name.lower() in expected_names:
            logger.info(
                "CO₂ dataset found using case-insensitive lookup: %s",
                file_path.name,
            )
            return file_path

    logger.error(
        "No supported CO₂ dataset found."
    )

    return None


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_name(column: Any) -> str:
    """
    Normalize a dataframe column name.

    Removes UTF-8 BOM characters and surrounding whitespace.
    """

    return (
        str(column)
        .replace("\ufeff", "")
        .strip()
    )


def normalize_dataframe_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a copy with normalized column names.

    Raises
    ------
    ValueError
        If normalization creates duplicate columns.
    """

    normalized = dataframe.copy()

    normalized.columns = [
        normalize_column_name(column)
        for column in normalized.columns
    ]

    duplicated_columns = (
        normalized.columns[
            normalized.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated_columns:
        raise ValueError(
            "Dataset contains duplicate column names after normalization: "
            + ", ".join(
                map(
                    str,
                    duplicated_columns,
                )
            )
        )

    return normalized


# =============================================================================
# REQUIRED COLUMN VALIDATION
# =============================================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the columns required by the ML model.
    """

    required_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if not missing_columns:
        return

    logger.error(
        "Dataset is missing required model columns: %s",
        missing_columns,
    )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "The configured dataset is missing one or more "
            "required model columns."
        ),
    )


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

def convert_model_columns_to_numeric(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert model-required columns to numeric values.

    Invalid values become NaN and are subsequently treated as invalid
    records.
    """

    result = dataframe[
        [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_statistics(
    series: pd.Series,
) -> dict[str, float | None]:
    """
    Calculate safe descriptive statistics.

    NaN and infinite values are excluded.

    If no valid numeric values exist, None is returned for each statistic.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = numeric.to_numpy(
        dtype=np.float64,
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    standard_deviation = (
        float(
            np.std(
                finite_values,
                ddof=1,
            )
        )
        if finite_values.size > 1
        else 0.0
    )

    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "std": standard_deviation,
    }


# =============================================================================
# DATASET INTEGRITY
# =============================================================================

def calculate_sha256(
    dataset_path: Path,
) -> str:
    """
    Calculate the SHA-256 checksum of the dataset.

    The checksum provides a reproducible fingerprint of the exact dataset
    currently being served.
    """

    digest = hashlib.sha256()

    try:
        with dataset_path.open(
            "rb"
        ) as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The CO₂ emissions dataset could not be found.",
        ) from exc

    except OSError as exc:
        logger.exception(
            "Unable to calculate dataset checksum."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to verify dataset integrity.",
        ) from exc

    return digest.hexdigest()


def build_dataset_integrity(
    dataset_path: Path,
) -> DatasetIntegrity:
    """
    Build dataset file integrity information.
    """

    try:
        stat = dataset_path.stat()

        modified_at = (
            pd.Timestamp
            .fromtimestamp(
                stat.st_mtime,
                tz="UTC",
            )
            .isoformat()
        )

        return DatasetIntegrity(
            fileSizeBytes=int(stat.st_size),
            sha256=calculate_sha256(
                dataset_path
            ),
            modifiedAt=modified_at,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The CO₂ emissions dataset could not be found.",
        ) from exc

    except OSError as exc:
        logger.exception(
            "Unable to inspect dataset integrity."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to inspect dataset integrity.",
        ) from exc


# =============================================================================
# FORMATTERS
# =============================================================================

def format_statistic(
    value: float | None,
    suffix: str = "",
) -> str:
    """
    Format a statistic safely for frontend display.
    """

    if value is None or not math.isfinite(value):
        return "N/A"

    return f"{value:.2f}{suffix}"


# =============================================================================
# DATASET METADATA BUILDER
# =============================================================================

def build_dataset_metadata(
    dataframe: pd.DataFrame,
    dataset_path: Path,
) -> DatasetMetadataResponse:
    """
    Build complete metadata from the actual dataset.

    No dataset statistics are hardcoded.
    """

    dataframe = normalize_dataframe_columns(
        dataframe
    )

    validate_required_columns(
        dataframe
    )

    total_records = int(
        len(dataframe)
    )

    total_columns = int(
        len(dataframe.columns)
    )

    if total_records == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The configured CO₂ emissions dataset "
                "contains no records."
            ),
        )

    column_names = [
        str(column)
        for column in dataframe.columns
    ]

    # -------------------------------------------------------------------------
    # Model dataframe
    # -------------------------------------------------------------------------

    model_dataframe = (
        convert_model_columns_to_numeric(
            dataframe
        )
    )

    numeric_array = model_dataframe.to_numpy(
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Valid records
    # -------------------------------------------------------------------------

    valid_mask = (
        model_dataframe.notna().all(axis=1)
        & np.isfinite(numeric_array).all(axis=1)
    )

    valid_record_count = int(
        valid_mask.sum()
    )

    invalid_record_count = (
        total_records - valid_record_count
    )

    validation_percentage = (
        valid_record_count / total_records
    ) * 100.0

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    missing_values = int(
        dataframe.isna()
        .sum()
        .sum()
    )

    # -------------------------------------------------------------------------
    # Feature statistics
    # -------------------------------------------------------------------------

    feature_statistics: dict[
        str,
        DatasetStatistics,
    ] = {}

    for column in FEATURE_COLUMNS:
        feature_statistics[column] = (
            DatasetStatistics(
                **calculate_statistics(
                    model_dataframe[column]
                )
            )
        )

    # -------------------------------------------------------------------------
    # Target statistics
    # -------------------------------------------------------------------------

    target_statistics = DatasetStatistics(
        **calculate_statistics(
            model_dataframe[TARGET_COLUMN]
        )
    )

    # -------------------------------------------------------------------------
    # Feature descriptions
    # -------------------------------------------------------------------------

    feature_descriptions = {
        "ENGINESIZE": DatasetFeature(
            column="ENGINESIZE",
            role="Feature",
            unit="L",
            description=(
                "Engine displacement measured in litres."
            ),
        ),
        "FUELCONSUMPTION_COMB_MPG": DatasetFeature(
            column="FUELCONSUMPTION_COMB_MPG",
            role="Feature",
            unit="MPG",
            description=(
                "Combined fuel consumption measured "
                "in miles per gallon."
            ),
        ),
        "CO2EMISSIONS": DatasetFeature(
            column="CO2EMISSIONS",
            role="Target",
            unit=TARGET_UNIT,
            description=(
                "Vehicle carbon-dioxide emissions "
                "measured in grams per kilometre."
            ),
        ),
    }

    features = [
        feature_descriptions[column]
        for column in (
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        )
    ]

    # -------------------------------------------------------------------------
    # Statistics cards
    # -------------------------------------------------------------------------

    engine_stats = feature_statistics[
        "ENGINESIZE"
    ]

    fuel_stats = feature_statistics[
        "FUELCONSUMPTION_COMB_MPG"
    ]

    statistics = [
        DatasetStatisticsCard(
            label="Total Records",
            value=str(total_records),
            description=(
                "Total rows available in the dataset."
            ),
            type="records",
        ),
        DatasetStatisticsCard(
            label="Engine Size",
            value=format_statistic(
                engine_stats.mean,
                " L",
            ),
            description=(
                "Average engine displacement."
            ),
            type="engine-size",
        ),
        DatasetStatisticsCard(
            label="Fuel Consumption",
            value=format_statistic(
                fuel_stats.mean,
                " MPG",
            ),
            description=(
                "Average combined fuel consumption."
            ),
            type="fuel-consumption",
        ),
        DatasetStatisticsCard(
            label="CO₂ Emissions",
            value=format_statistic(
                target_statistics.mean,
                f" {TARGET_UNIT}",
            ),
            description=(
                "Average vehicle CO₂ emissions."
            ),
            type="target",
        ),
    ]

    # -------------------------------------------------------------------------
    # Model information
    # -------------------------------------------------------------------------

    model = DatasetModelInformation(
        name=MODEL_NAME,
        description=MODEL_DESCRIPTION,
        features=list(FEATURE_COLUMNS),
        target=TARGET_COLUMN,
        targetUnit=TARGET_UNIT,
    )

    # -------------------------------------------------------------------------
    # Dataset timestamp
    # -------------------------------------------------------------------------

    try:
        modified_timestamp = dataset_path.stat().st_mtime

        last_updated = (
            pd.Timestamp
            .fromtimestamp(
                modified_timestamp,
                tz="UTC",
            )
            .isoformat()
        )

    except OSError:
        logger.warning(
            "Unable to determine dataset modification timestamp."
        )
        last_updated = None

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    integrity = build_dataset_integrity(
        dataset_path
    )

    # -------------------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------------------

    return DatasetMetadataResponse(
        datasetName=dataset_path.stem,
        title="CO₂ Emissions Dataset",
        description=(
            "Dataset used by the CO₂ Emission Predictor "
            "machine-learning pipeline."
        ),
        recordCount=total_records,
        columnCount=total_columns,
        columnNames=column_names,
        missingValues=missing_values,
        validRecordCount=valid_record_count,
        invalidRecordCount=invalid_record_count,
        featureCount=len(FEATURE_COLUMNS),
        targetCount=1,
        validationPercentage=float(
            validation_percentage
        ),
        features=features,
        featureStatistics=feature_statistics,
        targetStatistics=target_statistics,
        statistics=statistics,
        model=model,
        lastUpdated=last_updated,
        integrity=integrity,
    )


# =============================================================================
# CSV LOADER
# =============================================================================

def load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    """
    Load the configured CSV dataset.

    UTF-8 with BOM is preferred. Latin-1 is supported as a fallback for
    legacy datasets.
    """

    try:
        return pd.read_csv(
            dataset_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

    except UnicodeDecodeError:
        logger.warning(
            (
                "UTF-8 decoding failed. "
                "Retrying dataset using latin-1."
            )
        )

        try:
            return pd.read_csv(
                dataset_path,
                encoding="latin-1",
                low_memory=False,
            )

        except UnicodeDecodeError as exc:
            logger.exception(
                "Dataset encoding is unsupported."
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "The CO₂ emissions dataset uses an "
                    "unsupported text encoding."
                ),
            ) from exc

    except FileNotFoundError as exc:
        logger.exception(
            "Dataset file disappeared before it could be loaded."
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "The CO₂ emissions dataset could not be found."
            ),
        ) from exc

    except PermissionError as exc:
        logger.exception(
            "Permission denied while reading dataset."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset could not be read."
            ),
        ) from exc

    except pd.errors.EmptyDataError as exc:
        logger.exception(
            "Dataset file is empty."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset is empty."
            ),
        ) from exc

    except pd.errors.ParserError as exc:
        logger.exception(
            "Dataset CSV parsing failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset could not be parsed."
            ),
        ) from exc

    except OSError as exc:
        logger.exception(
            "Unable to access dataset file."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset could not be accessed."
            ),
        ) from exc


# =============================================================================
# DATASET HEALTH
# =============================================================================

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check CO₂ dataset availability",
)
def get_dataset_health() -> dict[str, Any]:
    """
    Check whether the configured dataset exists and contains the required
    model columns.

    This endpoint is intentionally lightweight and can be used by deployment
    health checks.
    """

    dataset_path = find_dataset()

    if dataset_path is None:
        return {
            "status": "unavailable",
            "datasetAvailable": False,
            "modelColumnsAvailable": False,
        }

    try:
        dataframe = load_dataset(
            dataset_path
        )

        dataframe = normalize_dataframe_columns(
            dataframe
        )

        required_columns = {
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        }

        available_columns = set(
            dataframe.columns
        )

        model_columns_available = (
            required_columns
            .issubset(available_columns)
        )

        return {
            "status": (
                "healthy"
                if model_columns_available
                else "invalid"
            ),
            "datasetAvailable": True,
            "modelColumnsAvailable": (
                model_columns_available
            ),
            "recordCount": int(
                len(dataframe)
            ),
        }

    except Exception:
        logger.exception(
            "Dataset health check failed."
        )

        return {
            "status": "error",
            "datasetAvailable": True,
            "modelColumnsAvailable": False,
        }


# =============================================================================
# API ENDPOINT
# =============================================================================

@router.get(
    "/metadata",
    response_model=DatasetMetadataResponse,
    response_model_exclude_none=False,
    status_code=status.HTTP_200_OK,
    summary="Get CO₂ emissions dataset metadata",
    description=(
        "Returns dynamically generated metadata and statistics from "
        "the actual CO₂ emissions CSV dataset."
    ),
)
def get_dataset_metadata() -> DatasetMetadataResponse:
    """
    Return metadata generated from the actual CSV dataset.

    The endpoint never uses hardcoded dataset statistics.
    """

    logger.info(
        "Dataset metadata request received."
    )

    # -------------------------------------------------------------------------
    # Locate dataset
    # -------------------------------------------------------------------------

    dataset_path = find_dataset()

    if dataset_path is None:
        logger.error(
            "CO₂ emissions dataset was not found."
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "The CO₂ emissions dataset could not be found."
            ),
        )

    # -------------------------------------------------------------------------
    # Obtain lightweight file signature
    # -------------------------------------------------------------------------

    mtime_ns, file_size = (
        get_dataset_signature(
            dataset_path
        )
    )

    # -------------------------------------------------------------------------
    # Cache lookup
    # -------------------------------------------------------------------------

    cached_metadata = get_cached_metadata(
        dataset_path,
        mtime_ns,
        file_size,
    )

    if cached_metadata is not None:
        logger.debug(
            "Returning cached CO₂ dataset metadata."
        )

        return cached_metadata

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    dataframe = load_dataset(
        dataset_path
    )

    # -------------------------------------------------------------------------
    # Generate metadata
    # -------------------------------------------------------------------------

    try:
        metadata = build_dataset_metadata(
            dataframe=dataframe,
            dataset_path=dataset_path,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.exception(
            "Dataset structural validation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset failed "
                "structural validation."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error while generating dataset metadata."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to generate CO₂ dataset metadata."
            ),
        ) from exc

    # -------------------------------------------------------------------------
    # Cache result
    # -------------------------------------------------------------------------

    store_cached_metadata(
        dataset_path=dataset_path,
        mtime_ns=mtime_ns,
        file_size=file_size,
        metadata=metadata,
    )

    logger.info(
        (
            "Dataset metadata generated successfully | "
            "dataset=%s | records=%d | valid=%d | invalid=%d"
        ),
        dataset_path.name,
        metadata.recordCount,
        metadata.validRecordCount,
        metadata.invalidRecordCount,
    )

    return metadata

