"""
===============================================================================
CO₂ EMISSION PREDICTOR - DATASET API
===============================================================================

Production-ready FastAPI routes for inspecting the actual CO₂ emissions
dataset used by the machine-learning pipeline.

Endpoint
--------
GET /dataset/metadata

Dataset location
----------------
<project-root>/data/

The API dynamically reads the CSV file. Dataset statistics are never
hardcoded.

Canonical model
---------------
Features:
    - ENGINESIZE
    - FUELCONSUMPTION_COMB_MPG

Target:
    - CO2EMISSIONS

===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status


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
# PROJECT PATHS
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# Expected:
#
# co2-emission-predictor/
# ├── app/
# │   └── routes/
# │       └── dataset.py
# └── data/
#
# parents[0] -> routes
# parents[1] -> app
# parents[2] -> project root

PROJECT_ROOT = CURRENT_FILE.parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"


# =============================================================================
# SUPPORTED DATASET FILE NAMES
# =============================================================================

DATASET_FILENAMES: tuple[str, ...] = (
    "FuelConsumption.csv",
    "fuel_consumption.csv",
    "CO2 Emissions.csv",
    "co2_emissions.csv",
    "co2_emission.csv",
    "CO2_Emissions.csv",
)


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def find_dataset() -> Path | None:
    """
    Locate the CO₂ emissions dataset.

    The function first checks the supported filenames and then performs
    a case-insensitive fallback search.

    Returns
    -------
    Path | None
        The dataset path when found, otherwise None.
    """

    logger.debug(
        "Searching for dataset in: %s",
        DATA_DIRECTORY,
    )

    if not DATA_DIRECTORY.exists():
        logger.error(
            "Dataset directory does not exist: %s",
            DATA_DIRECTORY,
        )
        return None

    if not DATA_DIRECTORY.is_dir():
        logger.error(
            "Dataset path is not a directory: %s",
            DATA_DIRECTORY,
        )
        return None

    # -------------------------------------------------------------------------
    # Preferred filenames
    # -------------------------------------------------------------------------

    for filename in DATASET_FILENAMES:
        dataset_path = DATA_DIRECTORY / filename

        if dataset_path.is_file():
            logger.info(
                "CO₂ dataset found: %s",
                dataset_path,
            )

            return dataset_path

    # -------------------------------------------------------------------------
    # Case-insensitive fallback
    # -------------------------------------------------------------------------

    try:
        available_files = list(DATA_DIRECTORY.iterdir())
    except OSError:
        logger.exception(
            "Unable to inspect dataset directory: %s",
            DATA_DIRECTORY,
        )
        return None

    expected_names = {
        filename.lower()
        for filename in DATASET_FILENAMES
    }

    for file_path in available_files:
        if not file_path.is_file():
            continue

        if file_path.name.lower() in expected_names:
            logger.info(
                "CO₂ dataset found using case-insensitive lookup: %s",
                file_path,
            )

            return file_path

    logger.error(
        "No supported CO₂ dataset found in %s",
        DATA_DIRECTORY,
    )

    return None


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_name(column: Any) -> str:
    """
    Normalize a CSV column name.

    Removes BOM characters and surrounding whitespace.
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
    Return a copy of the dataframe with normalized column names.
    """

    dataframe = dataframe.copy()

    dataframe.columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    return dataframe


# =============================================================================
# REQUIRED COLUMN VALIDATION
# =============================================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that all model-required columns exist.
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
        "Dataset missing required columns: %s",
        missing_columns,
    )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "The dataset is missing required model columns: "
            + ", ".join(missing_columns)
        ),
    )


# =============================================================================
# NUMERIC STATISTICS
# =============================================================================

def calculate_statistics(
    series: pd.Series,
) -> dict[str, float]:
    """
    Calculate descriptive statistics for a numeric series.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    values = numeric.to_numpy(
        dtype=np.float64,
    )

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        ),
    }


# =============================================================================
# DATASET METADATA BUILDER
# =============================================================================

def build_dataset_metadata(
    dataframe: pd.DataFrame,
    dataset_path: Path,
) -> dict[str, Any]:
    """
    Build the complete dataset metadata response.

    All dataset statistics are calculated from the actual CSV file.
    """

    dataframe = normalize_dataframe_columns(
        dataframe,
    )

    validate_required_columns(
        dataframe,
    )

    # -------------------------------------------------------------------------
    # Basic dataset information
    # -------------------------------------------------------------------------

    total_records = int(len(dataframe))
    total_columns = int(len(dataframe.columns))

    column_names = [
        str(column)
        for column in dataframe.columns
    ]

    # -------------------------------------------------------------------------
    # Model dataframe
    # -------------------------------------------------------------------------

    model_dataframe = dataframe[
        [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    for column in model_dataframe.columns:
        model_dataframe[column] = pd.to_numeric(
            model_dataframe[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Valid records
    # -------------------------------------------------------------------------

    numeric_array = model_dataframe.to_numpy(
        dtype=np.float64,
    )

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
        (
            valid_record_count
            / total_records
        )
        * 100.0
        if total_records > 0
        else 0.0
    )

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
        dict[str, float],
    ] = {}

    for column in FEATURE_COLUMNS:
        feature_statistics[column] = calculate_statistics(
            model_dataframe[column],
        )

    # -------------------------------------------------------------------------
    # Target statistics
    # -------------------------------------------------------------------------

    target_statistics = calculate_statistics(
        model_dataframe[TARGET_COLUMN],
    )

    # -------------------------------------------------------------------------
    # Feature descriptions
    # -------------------------------------------------------------------------

    feature_descriptions = {
        "ENGINESIZE": {
            "column": "ENGINESIZE",
            "role": "Feature",
            "unit": "L",
            "description": (
                "Engine displacement measured in litres."
            ),
        },
        "FUELCONSUMPTION_COMB_MPG": {
            "column": "FUELCONSUMPTION_COMB_MPG",
            "role": "Feature",
            "unit": "MPG",
            "description": (
                "Combined fuel consumption measured "
                "in miles per gallon."
            ),
        },
        "CO2EMISSIONS": {
            "column": "CO2EMISSIONS",
            "role": "Target",
            "unit": TARGET_UNIT,
            "description": (
                "Vehicle carbon-dioxide emissions "
                "measured in grams per kilometre."
            ),
        },
    }

    features = [
        feature_descriptions[column]
        for column in [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
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
        {
            "label": "Total Records",
            "value": str(total_records),
            "description": (
                "Total rows available in the dataset."
            ),
            "type": "records",
        },
        {
            "label": "Engine Size",
            "value": (
                f"{engine_stats['mean']:.2f} L"
            ),
            "description": (
                "Average engine displacement."
            ),
            "type": "engine-size",
        },
        {
            "label": "Fuel Consumption",
            "value": (
                f"{fuel_stats['mean']:.2f} MPG"
            ),
            "description": (
                "Average combined fuel consumption."
            ),
            "type": "fuel-consumption",
        },
        {
            "label": "CO₂ Emissions",
            "value": (
                f"{target_statistics['mean']:.2f} "
                f"{TARGET_UNIT}"
            ),
            "description": (
                "Average vehicle CO₂ emissions."
            ),
            "type": "target",
        },
    ]

    # -------------------------------------------------------------------------
    # Model information
    #
    # IMPORTANT:
    # The frontend expects name and description.
    # We also expose features, target and targetUnit.
    # -------------------------------------------------------------------------

    model = {
        "name": MODEL_NAME,
        "description": MODEL_DESCRIPTION,
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "targetUnit": TARGET_UNIT,
    }

    # -------------------------------------------------------------------------
    # Last modified timestamp
    # -------------------------------------------------------------------------

    try:
        modified_timestamp = dataset_path.stat().st_mtime

        last_updated = pd.Timestamp.fromtimestamp(
            modified_timestamp,
            tz="UTC",
        ).isoformat()

    except OSError:
        last_updated = None

    # -------------------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------------------

    return {
        "datasetName": dataset_path.stem,

        "title": "CO₂ Emissions Dataset",

        "description": (
            "Dataset used by the CO₂ Emission Predictor "
            "machine-learning pipeline."
        ),

        "recordCount": total_records,

        "columnCount": total_columns,

        "columnNames": column_names,

        "missingValues": missing_values,

        "validRecordCount": valid_record_count,

        "invalidRecordCount": invalid_record_count,

        "featureCount": len(FEATURE_COLUMNS),

        "targetCount": 1,

        "validationPercentage": float(
            validation_percentage
        ),

        "features": features,

        "featureStatistics": feature_statistics,

        "targetStatistics": {
            "column": TARGET_COLUMN,
            **target_statistics,
        },

        "statistics": statistics,

        "model": model,

        "lastUpdated": last_updated,
    }


# =============================================================================
# API ENDPOINT
# =============================================================================

@router.get(
    "/metadata",
    status_code=status.HTTP_200_OK,
    summary="Get CO₂ emissions dataset metadata",
)
def get_dataset_metadata() -> dict[str, Any]:
    """
    Return dynamically generated metadata from the actual CSV dataset.
    """

    logger.info(
        "Dataset metadata request received.",
    )

    # -------------------------------------------------------------------------
    # Find dataset
    # -------------------------------------------------------------------------

    dataset_path = find_dataset()

    if dataset_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "CO₂ emissions dataset could not be found. "
                "Please place FuelConsumption.csv inside: "
                f"{DATA_DIRECTORY}"
            ),
        )

    # -------------------------------------------------------------------------
    # Read CSV
    # -------------------------------------------------------------------------

    try:
        dataframe = pd.read_csv(
            dataset_path,
        )

    except FileNotFoundError as exc:
        logger.exception(
            "Dataset file disappeared: %s",
            dataset_path,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "The CO₂ emissions dataset could not be found."
            ),
        ) from exc

    except pd.errors.EmptyDataError as exc:
        logger.exception(
            "Dataset is empty: %s",
            dataset_path,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset is empty."
            ),
        ) from exc

    except pd.errors.ParserError as exc:
        logger.exception(
            "Dataset parsing failed: %s",
            dataset_path,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset could not be parsed."
            ),
        ) from exc

    except UnicodeDecodeError as exc:
        logger.exception(
            "Dataset encoding could not be decoded: %s",
            dataset_path,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The CO₂ emissions dataset contains an unsupported "
                "text encoding."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected dataset loading error.",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to read the CO₂ emissions dataset."
            ),
        ) from exc

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

    except Exception as exc:
        logger.exception(
            "Failed to generate dataset metadata.",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to generate CO₂ dataset metadata."
            ),
        ) from exc

    logger.info(
        (
            "Dataset metadata generated successfully | "
            "dataset=%s | records=%d | valid=%d | invalid=%d"
        ),
        dataset_path.name,
        metadata["recordCount"],
        metadata["validRecordCount"],
        metadata["invalidRecordCount"],
    )

    return metadata