"""
============================================================================
CO₂ EMISSION PREDICTOR
============================================================================

Dataset API Routes
------------------

Production-ready dataset service.

Responsibilities
-----------------
- Locate the canonical FuelConsumption.csv dataset reliably.
- Never depend on the process working directory.
- Support an optional DATASET_PATH environment override.
- Validate the dataset schema.
- Validate dataset values.
- Normalize column names safely.
- Calculate dataset metadata.
- Calculate feature statistics.
- Calculate target statistics.
- Return model information.
- Return dataset records for the frontend.
- Cache the loaded dataset safely.
- Detect dataset file changes automatically.
- Provide deterministic API responses.
- Return JSON-safe values.
- Provide meaningful HTTP error responses.
- Remain compatible with local development and Render deployment.

Canonical dataset
-----------------
The project currently contains:

project-root/
├── app/
│   └── routes/
│       └── dataset.py
├── data/
│   ├── CO2 Emissions.csv
│   ├── co2_emission.csv
│   ├── co2_emissions.csv
│   ├── fuel_consumption.csv
│   └── FuelConsumption.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
└── ...

The canonical dataset for this API is:

    data/FuelConsumption.csv

The API does NOT depend on:
    FuelConsumptionCo2.csv

Required dataset columns
------------------------
ENGINESIZE
FUELCONSUMPTION_COMB_MPG
CO2EMISSIONS

API endpoints
-------------
GET /dataset/metadata
GET /dataset/records

Example
-------
GET /dataset/metadata

GET /dataset/records?offset=0&limit=100

============================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

"""
dataset.py is expected to live at:

project-root/
└── app/
    └── routes/
        └── dataset.py

Therefore:

parents[0] -> routes
parents[1] -> app
parents[2] -> project-root
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# This is the canonical file in the user's current project.
CANONICAL_DATASET_FILENAME = "FuelConsumption.csv"

CANONICAL_DATASET_PATH = (
    DATA_DIRECTORY / CANONICAL_DATASET_FILENAME
)


# ============================================================================
# DATASET SCHEMA
# ============================================================================

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ENGINESIZE",
    "FUELCONSUMPTION_COMB_MPG",
    "CO2EMISSIONS",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "ENGINESIZE",
    "FUELCONSUMPTION_COMB_MPG",
)

TARGET_COLUMN = "CO2EMISSIONS"

TARGET_UNIT = "g/km"


# ============================================================================
# DATASET DESCRIPTION
# ============================================================================

DATASET_TITLE = (
    "Vehicle Fuel Consumption and CO₂ Emissions"
)

DATASET_DESCRIPTION = (
    "Vehicle fuel-consumption dataset used by the CO₂ emission "
    "prediction machine-learning pipeline."
)


# ============================================================================
# MODEL INFORMATION
# ============================================================================

MODEL_NAME = "Multiple Linear Regression"

MODEL_DESCRIPTION = (
    "A multiple linear regression model that estimates vehicle "
    "CO₂ emissions from engine size and combined fuel consumption."
)


# ============================================================================
# API PAGINATION
# ============================================================================

DEFAULT_PAGE_SIZE = 100

MAX_PAGE_SIZE = 1000


# ============================================================================
# OPTIONAL ENVIRONMENT CONFIGURATION
# ============================================================================

DATASET_PATH_ENVIRONMENT_VARIABLE = "DATASET_PATH"


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def _configured_dataset_path() -> Path | None:
    """
    Resolve an optional DATASET_PATH environment variable.

    Supported values:

        DATASET_PATH=data/FuelConsumption.csv

    or:

        DATASET_PATH=/absolute/path/to/FuelConsumption.csv

    Relative paths are resolved against PROJECT_ROOT rather than the
    process working directory.
    """

    configured = os.getenv(
        DATASET_PATH_ENVIRONMENT_VARIABLE
    )

    if configured is None:
        return None

    configured = configured.strip()

    if not configured:
        return None

    path = Path(configured).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _find_dataset_path() -> Path:
    """
    Resolve the dataset path deterministically.

    Resolution order:

    1. DATASET_PATH environment variable.
    2. Canonical data/FuelConsumption.csv.

    We intentionally DO NOT automatically select an arbitrary CSV from
    the data directory because the project contains multiple CSV files.

    This prevents production deployments from accidentally loading the
    wrong dataset.
    """

    # ----------------------------------------------------------------------
    # Environment override
    # ----------------------------------------------------------------------

    configured_path = _configured_dataset_path()

    if configured_path is not None:
        if not configured_path.is_file():
            raise FileNotFoundError(
                "Configured dataset file does not exist: "
                f"{configured_path}"
            )

        if configured_path.suffix.lower() != ".csv":
            raise ValueError(
                "Configured dataset file must have a .csv extension."
            )

        return configured_path

    # ----------------------------------------------------------------------
    # Canonical dataset
    # ----------------------------------------------------------------------

    if CANONICAL_DATASET_PATH.is_file():
        return CANONICAL_DATASET_PATH.resolve()

    # ----------------------------------------------------------------------
    # Helpful diagnostic information
    # ----------------------------------------------------------------------

    available_csv_files: list[str] = []

    if DATA_DIRECTORY.exists():
        available_csv_files = sorted(
            file.name
            for file in DATA_DIRECTORY.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".csv"
        )

    available_message = (
        ", ".join(available_csv_files)
        if available_csv_files
        else "none"
    )

    raise FileNotFoundError(
        "The canonical dataset file could not be found. "
        f"Expected: {CANONICAL_DATASET_PATH}. "
        f"Available CSV files: {available_message}."
    )


# ============================================================================
# CACHE
# ============================================================================

_DATASET_CACHE: pd.DataFrame | None = None

_DATASET_PATH_CACHE: Path | None = None

_DATASET_SIGNATURE_CACHE: str | None = None

_CACHE_LOCK = Lock()


# ============================================================================
# DATASET FILE SIGNATURE
# ============================================================================

def _file_signature(
    path: Path,
) -> str:
    """
    Generate a deterministic file signature.

    The signature allows the service to detect changes to the dataset
    without continuously re-reading the CSV file.

    The signature includes:

    - absolute path
    - file size
    - modification timestamp
    """

    try:
        stat = path.stat()
    except OSError as exc:
        raise FileNotFoundError(
            f"Unable to inspect dataset file: {path}"
        ) from exc

    raw = (
        f"{path.resolve()}|"
        f"{stat.st_size}|"
        f"{stat.st_mtime_ns}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def _normalize_column_name(
    column: Any,
) -> str:
    """
    Normalize a dataframe column name.

    Examples:

        " ENGINESIZE " -> "ENGINESIZE"

        "fuelconsumption_comb_mpg"
            -> "FUELCONSUMPTION_COMB_MPG"

    A UTF-8 BOM is also removed.
    """

    return (
        str(column)
        .replace("\ufeff", "")
        .strip()
        .upper()
    )


def _normalize_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dataframe column names.

    Dataset values themselves are not modified here.
    """

    dataframe = dataframe.copy()

    normalized_columns = [
        _normalize_column_name(column)
        for column in dataframe.columns
    ]

    # Prevent duplicate columns after normalization.
    if len(normalized_columns) != len(
        set(normalized_columns)
    ):
        duplicates = sorted(
            {
                column
                for column in normalized_columns
                if normalized_columns.count(column) > 1
            }
        )

        raise ValueError(
            "Dataset schema validation failed because column "
            f"names become duplicated after normalization: {duplicates}"
        )

    dataframe.columns = normalized_columns

    return dataframe


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def _validate_schema(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the structural dataset schema.
    """

    if dataframe.empty:
        raise ValueError(
            "Dataset validation failed because the dataset contains "
            "no records."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset schema validation failed. "
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )


# ============================================================================
# NUMERIC CONVERSION
# ============================================================================

def _coerce_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert model-required columns to numeric values.

    Values that cannot be converted become NaN and are subsequently
    classified as invalid records.
    """

    dataframe = dataframe.copy()

    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


# ============================================================================
# NUMERIC VALUE VALIDATION
# ============================================================================

def _validate_numeric_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate values required by the prediction pipeline.

    Missing values are allowed at this stage because they are represented
    as invalid records in the dataset quality metadata.

    Physically invalid values are rejected because silently correcting
    them would change the source dataset.
    """

    # ----------------------------------------------------------------------
    # Replace infinities first.
    # ----------------------------------------------------------------------

    dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # ----------------------------------------------------------------------
    # Engine size
    # ----------------------------------------------------------------------

    engine_size = dataframe[
        "ENGINESIZE"
    ].dropna()

    if (
        not engine_size.empty
        and engine_size.lt(0).any()
    ):
        raise ValueError(
            "Dataset contains negative ENGINESIZE values."
        )

    # ----------------------------------------------------------------------
    # Combined fuel consumption
    # ----------------------------------------------------------------------

    fuel_consumption = dataframe[
        "FUELCONSUMPTION_COMB_MPG"
    ].dropna()

    if (
        not fuel_consumption.empty
        and fuel_consumption.le(0).any()
    ):
        raise ValueError(
            "Dataset contains non-positive "
            "FUELCONSUMPTION_COMB_MPG values."
        )

    # ----------------------------------------------------------------------
    # CO₂ emissions
    # ----------------------------------------------------------------------

    emissions = dataframe[
        TARGET_COLUMN
    ].dropna()

    if (
        not emissions.empty
        and emissions.lt(0).any()
    ):
        raise ValueError(
            "Dataset contains negative CO2EMISSIONS values."
        )


# ============================================================================
# DATASET LOADER
# ============================================================================

def _load_dataset() -> pd.DataFrame:
    """
    Load, normalize and validate the dataset.

    The dataframe is cached until the dataset file changes.

    Thread safety:
        A process-local lock protects cache initialization and refresh.

    Important:
        The returned dataframe is always copied so API consumers cannot
        mutate the cached dataframe accidentally.
    """

    global _DATASET_CACHE
    global _DATASET_PATH_CACHE
    global _DATASET_SIGNATURE_CACHE

    with _CACHE_LOCK:

        # ------------------------------------------------------------------
        # Resolve dataset path.
        # ------------------------------------------------------------------

        path = _find_dataset_path()

        # ------------------------------------------------------------------
        # Generate current file signature.
        # ------------------------------------------------------------------

        signature = _file_signature(path)

        # ------------------------------------------------------------------
        # Return cached dataframe when unchanged.
        # ------------------------------------------------------------------

        if (
            _DATASET_CACHE is not None
            and _DATASET_PATH_CACHE == path
            and _DATASET_SIGNATURE_CACHE == signature
        ):
            return _DATASET_CACHE.copy()

        logger.info(
            "Loading dataset from %s",
            path,
        )

        # ------------------------------------------------------------------
        # Read CSV.
        # ------------------------------------------------------------------

        try:
            dataframe = pd.read_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
            )

        except UnicodeDecodeError:
            try:
                dataframe = pd.read_csv(
                    path,
                    encoding="latin-1",
                    low_memory=False,
                )

            except Exception as exc:
                logger.exception(
                    "Unable to read dataset using fallback encoding: %s",
                    path,
                )

                raise RuntimeError(
                    "The dataset file could not be read."
                ) from exc

        except pd.errors.EmptyDataError as exc:
            logger.error(
                "Dataset file is empty: %s",
                path,
            )

            raise ValueError(
                "The dataset file is empty."
            ) from exc

        except pd.errors.ParserError as exc:
            logger.error(
                "Dataset CSV parsing failed: %s",
                path,
            )

            raise ValueError(
                "The dataset CSV file could not be parsed."
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error while reading dataset: %s",
                path,
            )

            raise RuntimeError(
                "The dataset file could not be read."
            ) from exc

        # ------------------------------------------------------------------
        # Normalize schema.
        # ------------------------------------------------------------------

        dataframe = _normalize_dataset(
            dataframe
        )

        # ------------------------------------------------------------------
        # Validate schema.
        # ------------------------------------------------------------------

        _validate_schema(
            dataframe
        )

        # ------------------------------------------------------------------
        # Convert model columns to numeric.
        # ------------------------------------------------------------------

        dataframe = _coerce_numeric_columns(
            dataframe
        )

        # ------------------------------------------------------------------
        # Validate numeric values.
        # ------------------------------------------------------------------

        _validate_numeric_values(
            dataframe
        )

        # ------------------------------------------------------------------
        # Cache validated dataframe.
        # ------------------------------------------------------------------

        _DATASET_CACHE = dataframe.copy()

        _DATASET_PATH_CACHE = path

        _DATASET_SIGNATURE_CACHE = signature

        logger.info(
            "Dataset loaded successfully: "
            "%s rows, %s columns",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe.copy()


# ============================================================================
# DATA QUALITY
# ============================================================================

def _get_valid_mask(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Return a boolean mask identifying records that contain valid values
    for every required model column.
    """

    return dataframe[
        list(REQUIRED_COLUMNS)
    ].notna().all(axis=1)


def _get_valid_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only records that are complete for the prediction pipeline.
    """

    mask = _get_valid_mask(
        dataframe
    )

    return dataframe.loc[
        mask
    ].copy()


# ============================================================================
# NUMERIC STATISTICS
# ============================================================================

def _numeric_statistics(
    series: pd.Series,
) -> dict[str, float]:
    """
    Calculate deterministic descriptive statistics.

    Returned fields:

    - min
    - max
    - mean
    - median
    - std
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    std = (
        float(values.std())
        if len(values) > 1
        else 0.0
    )

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(std),
    }


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class DatasetFeature(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    column: str = Field(
        min_length=1
    )

    role: Literal[
        "Feature",
        "Target",
    ]

    unit: str

    description: str


class DatasetStatistic(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    label: str = Field(
        min_length=1
    )

    value: str = Field(
        min_length=1
    )

    description: str

    type: str | None = None


class DatasetNumericStatistics(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    min: float
    max: float
    mean: float
    median: float
    std: float


class DatasetTargetStatistics(
    DatasetNumericStatistics
):
    column: str = Field(
        min_length=1
    )


class DatasetModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    name: str = Field(
        min_length=1
    )

    description: str

    features: list[str] = Field(
        min_length=1
    )

    target: str = Field(
        min_length=1
    )

    targetUnit: str = Field(
        min_length=1
    )


class DatasetMetadataResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

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

    featureStatistics: dict[
        str,
        DatasetNumericStatistics,
    ]

    targetStatistics: DatasetTargetStatistics

    statistics: list[DatasetStatistic]

    model: DatasetModel

    lastUpdated: str | None = None


class DatasetRecordsResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    datasetName: str

    totalRecords: int

    returnedRecords: int

    offset: int

    limit: int

    columns: list[str]

    records: list[dict[str, Any]]


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

FEATURE_DEFINITIONS: tuple[
    DatasetFeature,
    ...
] = (
    DatasetFeature(
        column="ENGINESIZE",
        role="Feature",
        unit="L",
        description=(
            "Vehicle engine displacement in litres."
        ),
    ),
    DatasetFeature(
        column="FUELCONSUMPTION_COMB_MPG",
        role="Feature",
        unit="MPG",
        description=(
            "Combined fuel consumption measured in "
            "miles per gallon."
        ),
    ),
    DatasetFeature(
        column="CO2EMISSIONS",
        role="Target",
        unit=TARGET_UNIT,
        description=(
            "Vehicle carbon dioxide emissions measured "
            "in grams per kilometre."
        ),
    ),
)


# ============================================================================
# STATISTICS BUILDER
# ============================================================================

def _build_statistics(
    dataframe: pd.DataFrame,
    valid_dataframe: pd.DataFrame,
) -> list[DatasetStatistic]:
    """
    Build the four overview statistics displayed by DatasetPage.tsx.
    """

    # ----------------------------------------------------------------------
    # Empty valid dataset safeguard.
    # ----------------------------------------------------------------------

    if valid_dataframe.empty:
        engine_size_value = "—"

        fuel_consumption_value = "—"

        emissions_value = "—"

    else:
        engine_size_value = (
            f"{valid_dataframe['ENGINESIZE'].mean():.2f} L"
        )

        fuel_consumption_value = (
            f"{valid_dataframe['FUELCONSUMPTION_COMB_MPG'].mean():.2f} MPG"
        )

        emissions_value = (
            f"{valid_dataframe[TARGET_COLUMN].mean():.2f} "
            f"{TARGET_UNIT}"
        )

    return [
        DatasetStatistic(
            label="Records",
            value=f"{len(dataframe):,}",
            description=(
                "Total records available in the dataset."
            ),
            type="records",
        ),
        DatasetStatistic(
            label="Engine Size",
            value=engine_size_value,
            description=(
                "Average engine size across valid records."
            ),
            type="engine-size",
        ),
        DatasetStatistic(
            label="Fuel Consumption",
            value=fuel_consumption_value,
            description=(
                "Average combined fuel consumption across "
                "valid records."
            ),
            type="fuel-consumption",
        ),
        DatasetStatistic(
            label="CO₂ Emissions",
            value=emissions_value,
            description=(
                "Average CO₂ emissions across valid records."
            ),
            type="target",
        ),
    ]


# ============================================================================
# METADATA BUILDER
# ============================================================================

def _build_metadata(
    dataframe: pd.DataFrame,
    dataset_path: Path,
) -> DatasetMetadataResponse:
    """
    Build the complete dataset metadata response.
    """

    record_count = len(
        dataframe
    )

    column_count = len(
        dataframe.columns
    )

    # ----------------------------------------------------------------------
    # Valid records.
    # ----------------------------------------------------------------------

    valid_mask = _get_valid_mask(
        dataframe
    )

    valid_dataframe = dataframe.loc[
        valid_mask
    ].copy()

    valid_record_count = len(
        valid_dataframe
    )

    invalid_record_count = (
        record_count
        - valid_record_count
    )

    # ----------------------------------------------------------------------
    # Validation percentage.
    # ----------------------------------------------------------------------

    validation_percentage = (
        (
            valid_record_count
            / record_count
        )
        * 100.0
        if record_count > 0
        else 0.0
    )

    # ----------------------------------------------------------------------
    # Missing values.
    # ----------------------------------------------------------------------

    missing_values = int(
        dataframe[
            list(REQUIRED_COLUMNS)
        ]
        .isna()
        .sum()
        .sum()
    )

    # ----------------------------------------------------------------------
    # Feature statistics.
    # ----------------------------------------------------------------------

    feature_statistics = {
        column: DatasetNumericStatistics(
            **_numeric_statistics(
                valid_dataframe[column]
            )
        )
        for column in FEATURE_COLUMNS
    }

    # ----------------------------------------------------------------------
    # Target statistics.
    # ----------------------------------------------------------------------

    target_statistics = (
        DatasetTargetStatistics(
            column=TARGET_COLUMN,
            **_numeric_statistics(
                valid_dataframe[
                    TARGET_COLUMN
                ]
            ),
        )
    )

    # ----------------------------------------------------------------------
    # Dataset modification timestamp.
    # ----------------------------------------------------------------------

    try:
        modified_time = (
            dataset_path.stat().st_mtime
        )

        last_updated = (
            datetime.fromtimestamp(
                modified_time,
                tz=timezone.utc,
            ).isoformat()
        )

    except OSError:
        last_updated = None

    # ----------------------------------------------------------------------
    # Response.
    # ----------------------------------------------------------------------

    return DatasetMetadataResponse(
        datasetName=dataset_path.stem,

        title=DATASET_TITLE,

        description=DATASET_DESCRIPTION,

        recordCount=record_count,

        columnCount=column_count,

        columnNames=[
            str(column)
            for column in dataframe.columns
        ],

        missingValues=missing_values,

        validRecordCount=valid_record_count,

        invalidRecordCount=invalid_record_count,

        featureCount=len(
            FEATURE_COLUMNS
        ),

        targetCount=1,

        validationPercentage=round(
            validation_percentage,
            4,
        ),

        features=list(
            FEATURE_DEFINITIONS
        ),

        featureStatistics=feature_statistics,

        targetStatistics=target_statistics,

        statistics=_build_statistics(
            dataframe,
            valid_dataframe,
        ),

        model=DatasetModel(
            name=MODEL_NAME,

            description=MODEL_DESCRIPTION,

            features=list(
                FEATURE_COLUMNS
            ),

            target=TARGET_COLUMN,

            targetUnit=TARGET_UNIT,
        ),

        lastUpdated=last_updated,
    )


# ============================================================================
# JSON-SAFE VALUE
# ============================================================================

def _json_safe_value(
    value: Any,
) -> Any:
    """
    Convert pandas / NumPy values into JSON-safe Python primitives.

    Rules:

    NaN / NaT / None
        -> None

    NumPy integers
        -> int

    NumPy floats
        -> float, or None for non-finite values

    NumPy booleans
        -> bool

    pandas Timestamp / datetime
        -> ISO-8601 string

    Everything else
        -> unchanged
    """

    # ----------------------------------------------------------------------
    # None.
    # ----------------------------------------------------------------------

    if value is None:
        return None

    # ----------------------------------------------------------------------
    # pandas / NumPy missing values.
    # ----------------------------------------------------------------------

    try:
        missing = pd.isna(value)

        if isinstance(
            missing,
            (bool, np.bool_),
        ) and bool(missing):
            return None

    except (TypeError, ValueError):
        pass

    # ----------------------------------------------------------------------
    # NumPy integer.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    # ----------------------------------------------------------------------
    # NumPy floating point.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        np.floating,
    ):
        numeric_value = float(value)

        return (
            numeric_value
            if math.isfinite(
                numeric_value
            )
            else None
        )

    # ----------------------------------------------------------------------
    # Native floating point.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(value)
            else None
        )

    # ----------------------------------------------------------------------
    # NumPy boolean.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    # ----------------------------------------------------------------------
    # Datetime values.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        (datetime, pd.Timestamp),
    ):
        return value.isoformat()

    # ----------------------------------------------------------------------
    # Native integer / boolean / string.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        (int, bool, str),
    ):
        return value

    # ----------------------------------------------------------------------
    # NumPy scalar fallback.
    # ----------------------------------------------------------------------

    if isinstance(
        value,
        np.generic,
    ):
        try:
            converted = value.item()

            return _json_safe_value(
                converted
            )

        except Exception:
            return str(value)

    # ----------------------------------------------------------------------
    # Final fallback.
    # ----------------------------------------------------------------------

    return value


# ============================================================================
# DATASET ERROR HELPERS
# ============================================================================

def _dataset_not_found_error(
    exc: Exception,
) -> HTTPException:
    """
    Build a consistent dataset-not-found HTTP response.
    """

    logger.error(
        "Dataset file not found: %s",
        exc,
    )

    return HTTPException(
        status_code=503,
        detail=(
            "The dataset file could not be found on the server. "
            "The API expects the canonical dataset at "
            "data/FuelConsumption.csv."
        ),
        headers={
            "X-Error-Code": "DATASET_NOT_FOUND"
        },
    )


def _dataset_invalid_error(
    exc: Exception,
) -> HTTPException:
    """
    Build a consistent dataset-validation HTTP response.
    """

    logger.error(
        "Dataset validation failed: %s",
        exc,
    )

    return HTTPException(
        status_code=500,
        detail=(
            "The dataset failed server-side validation."
        ),
        headers={
            "X-Error-Code": "DATASET_INVALID"
        },
    )


# ============================================================================
# GET /dataset/metadata
# ============================================================================

@router.get(
    "/metadata",
    response_model=DatasetMetadataResponse,
    summary="Get dataset metadata",
    description=(
        "Returns dataset metadata, statistics, feature definitions, "
        "model information and data-quality information."
    ),
)
async def get_dataset_metadata() -> DatasetMetadataResponse:
    """
    Return metadata for the canonical CO₂ emissions dataset.
    """

    try:
        # ------------------------------------------------------------------
        # Load validated dataset.
        # ------------------------------------------------------------------

        dataframe = _load_dataset()

        # ------------------------------------------------------------------
        # Resolve path independently.
        #
        # This avoids relying solely on mutable global cache state.
        # ------------------------------------------------------------------

        dataset_path = _find_dataset_path()

        # ------------------------------------------------------------------
        # Build response.
        # ------------------------------------------------------------------

        return _build_metadata(
            dataframe,
            dataset_path,
        )

    except FileNotFoundError as exc:
        raise _dataset_not_found_error(
            exc
        ) from exc

    except ValueError as exc:
        raise _dataset_invalid_error(
            exc
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected dataset metadata error."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load dataset metadata."
            ),
            headers={
                "X-Error-Code": "DATASET_METADATA_ERROR"
            },
        ) from exc


# ============================================================================
# GET /dataset/records
# ============================================================================

@router.get(
    "/records",
    response_model=DatasetRecordsResponse,
    summary="Get dataset records",
    description=(
        "Returns paginated records from the validated dataset."
    ),
)
async def get_dataset_records(
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip.",
    ),
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Maximum number of records to return.",
    ),
) -> DatasetRecordsResponse:
    """
    Return a paginated view of the canonical dataset.
    """

    try:
        # ------------------------------------------------------------------
        # Load validated dataset.
        # ------------------------------------------------------------------

        dataframe = _load_dataset()

        total_records = len(
            dataframe
        )

        # ------------------------------------------------------------------
        # Slice requested page.
        # ------------------------------------------------------------------

        page = dataframe.iloc[
            offset : offset + limit
        ]

        # ------------------------------------------------------------------
        # Convert rows to JSON-safe records.
        # ------------------------------------------------------------------

        raw_records = page.to_dict(
            orient="records"
        )

        records: list[
            dict[str, Any]
        ] = []

        for row in raw_records:
            safe_record = {
                str(key): _json_safe_value(
                    value
                )
                for key, value in row.items()
            }

            records.append(
                safe_record
            )

        # ------------------------------------------------------------------
        # Resolve dataset path.
        # ------------------------------------------------------------------

        dataset_path = _find_dataset_path()

        # ------------------------------------------------------------------
        # Build response.
        # ------------------------------------------------------------------

        return DatasetRecordsResponse(
            datasetName=dataset_path.stem,

            totalRecords=total_records,

            returnedRecords=len(
                records
            ),

            offset=offset,

            limit=limit,

            columns=[
                str(column)
                for column in dataframe.columns
            ],

            records=records,
        )

    except FileNotFoundError as exc:
        raise _dataset_not_found_error(
            exc
        ) from exc

    except ValueError as exc:
        raise _dataset_invalid_error(
            exc
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected dataset records error."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load dataset records."
            ),
            headers={
                "X-Error-Code": "DATASET_RECORDS_ERROR"
            },
        ) from exc