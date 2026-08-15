"""
Production-grade data preprocessing for the CO₂ Emission Predictor.

Responsibilities
----------------
This module is responsible for:

- Loading the CO₂ emissions dataset.
- Validating the dataset schema.
- Cleaning missing, invalid, NaN, and infinite values.
- Preparing model features and target values.
- Providing reusable dataset statistics.
- Keeping feature definitions centralized.
- Ensuring training and inference use the exact same feature order.
- Providing metadata helpers for the application.

Model features
--------------
- ENGINESIZE
- FUELCONSUMPTION_COMB_MPG

Target
------
- CO2EMISSIONS

Important
---------
The preprocessing pipeline is shared by the training and inference
systems. Feature order must therefore remain consistent.

The trained model is intended for prediction purposes only and must
not be interpreted as an official vehicle-emissions certification,
regulatory measurement, laboratory result, or legal emissions
determination.
"""

from __future__ import annotations

import logging
import os
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DEFAULT_DATA_URL: Final[str] = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/"
    "labs/Module%202/data/FuelConsumptionCo2.csv"
)

"""
Dataset URL.

The environment variable CO2_DATASET_URL can be used to provide a
different dataset without changing source code.

Example:

CO2_DATASET_URL=https://example.com/dataset.csv
"""

DATA_URL: Final[str] = os.getenv(
    "CO2_DATASET_URL",
    DEFAULT_DATA_URL,
).strip()

if not DATA_URL:
    raise RuntimeError(
        "CO2_DATASET_URL cannot be empty."
    )


# ============================================================================
# MODEL SCHEMA
# ============================================================================

"""
Canonical model features.

IMPORTANT:
The order of these columns is part of the model contract.

The same order is used for:

1. Dataset preparation.
2. Model training.
3. Feature scaling.
4. Model inference.
5. API metadata.

Do not reorder these columns unless the model is retrained.
"""

FEATURE_COLUMNS: Final[list[str]] = [
    "ENGINESIZE",
    "FUELCONSUMPTION_COMB_MPG",
]

TARGET_COLUMN: Final[str] = "CO2EMISSIONS"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

MINIMUM_TRAINING_ROWS: Final[int] = 10


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """
    Load the CO₂ emissions dataset.

    The configured dataset source is loaded and its schema is
    validated before returning it.

    Returns:
        Raw dataset as a pandas DataFrame.

    Raises:
        RuntimeError:
            If the dataset cannot be downloaded or parsed.

        ValueError:
            If the dataset is empty or has an invalid schema.
    """

    logger.info(
        "Loading CO₂ emissions dataset."
    )

    logger.debug(
        "Dataset source: %s",
        DATA_URL,
    )

    try:
        data = pd.read_csv(
            DATA_URL,
        )

    except Exception as exc:
        logger.exception(
            "Unable to load CO₂ emissions dataset."
        )

        raise RuntimeError(
            "Could not load the CO₂ emissions dataset."
        ) from exc

    if data is None:
        raise RuntimeError(
            "Dataset loader returned no data."
        )

    if data.empty:
        raise ValueError(
            "The CO₂ emissions dataset is empty."
        )

    validate_schema(data)

    logger.info(
        "Dataset loaded successfully | rows=%d | columns=%d",
        len(data),
        len(data.columns),
    )

    return data


# ============================================================================
# DATASET SCHEMA VALIDATION
# ============================================================================

def validate_schema(
    data: pd.DataFrame,
) -> None:
    """
    Validate that the dataset contains all required columns.

    Args:
        data:
            Dataset to validate.

    Raises:
        TypeError:
            If data is not a pandas DataFrame.

        ValueError:
            If required columns are missing.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Dataset must be provided as a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        logger.error(
            "Dataset is missing required columns: %s",
            missing_columns,
        )

        raise ValueError(
            "The dataset is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and prepare the dataset for machine learning.

    Processing includes:

    1. Schema validation.
    2. Selecting only required columns.
    3. Converting values to numeric values.
    4. Removing missing values.
    5. Removing NaN and infinite values.
    6. Resetting the DataFrame index.
    7. Converting model columns to float64.
    8. Validating the final dataset size.

    Args:
        data:
            Raw dataset.

    Returns:
        Cleaned DataFrame containing model features and target.

    Raises:
        TypeError:
            If data is not a pandas DataFrame.

        ValueError:
            If no valid records remain or the dataset is too small.
    """

    validate_schema(data)

    logger.info(
        "Preparing dataset for machine-learning training."
    )

    required_columns = list(
        REQUIRED_COLUMNS
    )

    cleaned_data = data.loc[
        :,
        required_columns,
    ].copy()

    original_rows = len(
        cleaned_data
    )

    # ------------------------------------------------------------------------
    # NUMERIC CONVERSION
    # ------------------------------------------------------------------------

    for column in required_columns:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------------
    # REMOVE MISSING VALUES
    # ------------------------------------------------------------------------

    before_dropna = len(
        cleaned_data
    )

    cleaned_data = cleaned_data.dropna(
        subset=required_columns,
    )

    missing_rows_removed = (
        before_dropna
        - len(cleaned_data)
    )

    if missing_rows_removed > 0:
        logger.warning(
            "Removed %d rows containing missing or "
            "non-numeric values.",
            missing_rows_removed,
        )

    # ------------------------------------------------------------------------
    # REMOVE NaN / INFINITE VALUES
    # ------------------------------------------------------------------------

    if not cleaned_data.empty:

        numeric_values = cleaned_data[
            required_columns
        ].to_numpy(
            dtype=np.float64,
        )

        valid_rows = np.isfinite(
            numeric_values
        ).all(
            axis=1
        )

        invalid_rows = int(
            (~valid_rows).sum()
        )

        if invalid_rows > 0:
            logger.warning(
                "Removed %d rows containing invalid "
                "numeric values.",
                invalid_rows,
            )

        cleaned_data = cleaned_data.loc[
            valid_rows
        ].copy()

    # ------------------------------------------------------------------------
    # VALIDATE RESULT
    # ------------------------------------------------------------------------

    if cleaned_data.empty:
        raise ValueError(
            "No valid data remains after preprocessing."
        )

    if len(cleaned_data) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "Insufficient valid records remain after preprocessing. "
            f"Required at least {MINIMUM_TRAINING_ROWS}, "
            f"found {len(cleaned_data)}."
        )

    # ------------------------------------------------------------------------
    # RESET INDEX
    # ------------------------------------------------------------------------

    cleaned_data = cleaned_data.reset_index(
        drop=True,
    )

    # ------------------------------------------------------------------------
    # NORMALIZE NUMERIC TYPES
    # ------------------------------------------------------------------------

    for column in required_columns:
        cleaned_data[column] = cleaned_data[
            column
        ].astype(
            np.float64,
        )

    # ------------------------------------------------------------------------
    # FINAL FINITE-VALUE VALIDATION
    # ------------------------------------------------------------------------

    final_values = cleaned_data[
        required_columns
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        final_values
    ).all():
        raise ValueError(
            "Preprocessing produced invalid numeric values."
        )

    logger.info(
        (
            "Data preprocessing completed successfully | "
            "original_rows=%d | valid_rows=%d | removed_rows=%d"
        ),
        original_rows,
        len(cleaned_data),
        original_rows - len(cleaned_data),
    )

    return cleaned_data


# ============================================================================
# TRAINING DATA
# ============================================================================

def get_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Load and prepare the dataset for model training.

    Returns:
        Tuple containing:

        X:
            Model feature DataFrame.

        y:
            Target Series.

        data:
            Complete cleaned DataFrame.

    Raises:
        RuntimeError:
            If the dataset cannot be loaded.

        ValueError:
            If the dataset is invalid or insufficient.
    """

    raw_data = load_data()

    cleaned_data = prepare_data(
        raw_data
    )

    X = cleaned_data[
        FEATURE_COLUMNS
    ].copy()

    y = cleaned_data[
        TARGET_COLUMN
    ].copy()

    # ------------------------------------------------------------------------
    # FINAL FEATURE VALIDATION
    # ------------------------------------------------------------------------

    feature_values = X.to_numpy(
        dtype=np.float64,
    )

    target_values = y.to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Training features contain invalid numeric values."
        )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Training target contains invalid numeric values."
        )

    if len(X) != len(y):
        raise ValueError(
            "Training features and targets have different lengths."
        )

    if len(X) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "Training dataset contains too few valid records."
        )

    logger.info(
        (
            "Training data prepared successfully | "
            "samples=%d | features=%d"
        ),
        len(X),
        len(FEATURE_COLUMNS),
    )

    return (
        X,
        y,
        cleaned_data,
    )


# ============================================================================
# DATASET STATISTICS
# ============================================================================

def get_dataset_statistics(
    data: pd.DataFrame | None = None,
) -> dict[str, int | float]:
    """
    Calculate statistics from the actual cleaned dataset.

    No evaluation metrics are generated here. This function only
    describes the dataset used by the model.

    If data is not supplied, the current configured dataset is loaded
    and cleaned automatically.

    Args:
        data:
            Optional dataset.

    Returns:
        Dictionary containing dataset statistics.

    Raises:
        ValueError:
            If the dataset is invalid or empty.
    """

    if data is None:
        _, _, cleaned_data = get_training_data()

    else:
        cleaned_data = prepare_data(
            data
        )

    if cleaned_data.empty:
        raise ValueError(
            "Cannot calculate statistics for an empty dataset."
        )

    # ------------------------------------------------------------------------
    # GENERIC FEATURE STATISTICS
    # ------------------------------------------------------------------------

    statistics: dict[
        str,
        int | float,
    ] = {
        "records": int(
            len(cleaned_data)
        ),
        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),
    }

    for feature_name in FEATURE_COLUMNS:

        values = cleaned_data[
            feature_name
        ].to_numpy(
            dtype=np.float64,
        )

        if values.size == 0:
            raise ValueError(
                f"No values available for feature: {feature_name}"
            )

        safe_name = (
            feature_name
            .lower()
            .replace(
                "fuelconsumption_comb_mpg",
                "fuel_consumption_mpg",
            )
            .replace(
                "enginesize",
                "engine_size",
            )
        )

        statistics[
            f"{safe_name}_min"
        ] = float(
            np.min(values)
        )

        statistics[
            f"{safe_name}_max"
        ] = float(
            np.max(values)
        )

    # ------------------------------------------------------------------------
    # TARGET STATISTICS
    # ------------------------------------------------------------------------

    target_values = cleaned_data[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float64,
    )

    if target_values.size == 0:
        raise ValueError(
            "No target values are available."
        )

    target_name = (
        TARGET_COLUMN
        .lower()
        .replace(
            "co2emissions",
            "co2",
        )
    )

    statistics[
        f"{target_name}_min"
    ] = float(
        np.min(target_values)
    )

    statistics[
        f"{target_name}_max"
    ] = float(
        np.max(target_values)
    )

    logger.debug(
        "Dataset statistics calculated: %s",
        statistics,
    )

    return statistics


# ============================================================================
# PUBLIC METADATA HELPERS
# ============================================================================

def get_feature_names() -> list[str]:
    """
    Return the configured model feature names.

    A copy is returned so callers cannot accidentally modify the
    centralized feature configuration.

    Returns:
        List containing model feature names.
    """

    return list(
        FEATURE_COLUMNS
    )


def get_target_name() -> str:
    """
    Return the configured model target name.

    Returns:
        Target column name.
    """

    return TARGET_COLUMN


def get_required_columns() -> list[str]:
    """
    Return all required dataset columns.

    Returns:
        List containing feature and target column names.
    """

    return list(
        REQUIRED_COLUMNS
    )


def get_dataset_source() -> str:
    """
    Return the currently configured dataset source.

    Returns:
        Dataset URL or configured source.
    """

    return DATA_URL

