"""
Production dataset metadata service for the CO₂ Emission Predictor.

Responsibilities
----------------
- Locate the canonical CO₂ emissions dataset.
- Load the dataset safely.
- Validate its required columns.
- Calculate dataset statistics dynamically.
- Calculate data-quality information dynamically.
- Expose model/dataset metadata to the API layer.

Nothing related to dataset statistics is hardcoded.

The service is intentionally independent of FastAPI.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Final

import pandas as pd

from app.ml.preprocessing import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

SERVICE_NAME: Final[str] = "CO2 Dataset Metadata Service"

TARGET_COLUMN: Final[str] = "CO2EMISSIONS"

DATASET_FILENAME: Final[str] = "FuelConsumption.csv"

DATASET_SEARCH_PATHS: Final[tuple[Path, ...]] = (
    Path("data") / DATASET_FILENAME,
    Path("datasets") / DATASET_FILENAME,
    Path("dataset") / DATASET_FILENAME,
)


# ============================================================================
# DATASET SERVICE
# ============================================================================


class DatasetService:
    """
    Service responsible for loading and analysing the CO₂ dataset.

    Dataset statistics are calculated dynamically from the actual dataset
    on disk. No record counts, means, ranges, or model statistics are
    hardcoded.
    """

    def __init__(self) -> None:
        logger.info(
            "Initializing %s.",
            SERVICE_NAME,
        )

        self.dataset_path = self._locate_dataset()

        logger.info(
            "Using CO₂ dataset: %s",
            self.dataset_path,
        )

        self._validate_dataset()

        logger.info(
            "%s initialized successfully.",
            SERVICE_NAME,
        )

    # ========================================================================
    # DATASET LOCATION
    # ========================================================================

    @staticmethod
    def _locate_dataset() -> Path:
        """
        Locate the canonical dataset.

        Returns:
            Path to the dataset.

        Raises:
            FileNotFoundError:
                If the dataset cannot be found.
        """

        project_root = Path(__file__).resolve().parents[2]

        candidates = tuple(
            project_root / path
            for path in DATASET_SEARCH_PATHS
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        searched_paths = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise FileNotFoundError(
            (
                "CO₂ emissions dataset could not be found. "
                "Please make sure the dataset exists in the project "
                f"data directory.\nSearched:\n{searched_paths}"
            )
        )

    # ========================================================================
    # DATASET LOADING
    # ========================================================================

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from disk.

        Returns:
            Pandas DataFrame containing the dataset.

        Raises:
            RuntimeError:
                If the dataset cannot be loaded.
        """

        try:
            dataframe = pd.read_csv(
                self.dataset_path,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load CO₂ dataset: %s",
                self.dataset_path,
            )

            raise RuntimeError(
                "Unable to load the CO₂ emissions dataset."
            ) from exc

        if dataframe.empty:
            raise RuntimeError(
                "The CO₂ emissions dataset is empty."
            )

        return dataframe

    # ========================================================================
    # DATASET VALIDATION
    # ========================================================================

    def _validate_dataset(self) -> None:
        """
        Validate that the dataset contains all required columns.
        """

        dataframe = self._load_dataset()

        required_columns = [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise RuntimeError(
                (
                    "The CO₂ emissions dataset is missing required "
                    f"columns: {missing_columns}"
                )
            )

        logger.debug(
            "Dataset validation successful. Columns=%s",
            list(dataframe.columns),
        )

    # ========================================================================
    # NUMERIC CLEANING
    # ========================================================================

    @staticmethod
    def _numeric_series(
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """
        Convert a dataset column to numeric values.

        Invalid values become NaN.
        """

        return pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # ========================================================================
    # FEATURE STATISTICS
    # ========================================================================

    @classmethod
    def _feature_statistics(
        cls,
        dataframe: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate statistics for every model feature.
        """

        result: dict[str, dict[str, float]] = {}

        for column in FEATURE_COLUMNS:
            series = cls._numeric_series(
                dataframe,
                column,
            ).dropna()

            if series.empty:
                raise RuntimeError(
                    (
                        f"Feature column '{column}' "
                        "contains no valid numeric values."
                    )
                )

            result[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
            }

        return result

    # ========================================================================
    # TARGET STATISTICS
    # ========================================================================

    @classmethod
    def _target_statistics(
        cls,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate statistics for the prediction target.
        """

        series = cls._numeric_series(
            dataframe,
            TARGET_COLUMN,
        ).dropna()

        if series.empty:
            raise RuntimeError(
                (
                    f"Target column '{TARGET_COLUMN}' "
                    "contains no valid numeric values."
                )
            )

        return {
            "column": TARGET_COLUMN,
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }

    # ========================================================================
    # DATA QUALITY
    # ========================================================================

    @classmethod
    def _calculate_quality(
        cls,
        dataframe: pd.DataFrame,
    ) -> tuple[int, int, int, float]:
        """
        Calculate dataset validity information.

        A record is considered valid when all model features and the target
        contain finite numeric values.
        """

        required_columns = [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]

        numeric_dataframe = dataframe[
            required_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        finite_mask = numeric_dataframe.apply(
            lambda column: column.map(
                lambda value: (
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                )
            )
        )

        valid_mask = finite_mask.all(
            axis=1,
        )

        total_records = len(dataframe)

        valid_records = int(
            valid_mask.sum()
        )

        invalid_records = (
            total_records -
            valid_records
        )

        validation_percentage = (
            (
                valid_records /
                total_records *
                100
            )
            if total_records > 0
            else 0.0
        )

        return (
            total_records,
            valid_records,
            invalid_records,
            validation_percentage,
        )

    # ========================================================================
    # PUBLIC METADATA API
    # ========================================================================

    def get_metadata(self) -> dict[str, Any]:
        """
        Generate complete dataset metadata.

        Returns:
            JSON-serializable dataset metadata.
        """

        dataframe = self._load_dataset()

        (
            record_count,
            valid_record_count,
            invalid_record_count,
            validation_percentage,
        ) = self._calculate_quality(
            dataframe,
        )

        feature_statistics = (
            self._feature_statistics(
                dataframe,
            )
        )

        target_statistics = (
            self._target_statistics(
                dataframe,
            )
        )

        features = [
            {
                "column": "ENGINESIZE",
                "role": "Feature",
                "unit": "L",
                "description": (
                    "Engine displacement in litres."
                ),
            },
            {
                "column": "FUELCONSUMPTION_COMB_MPG",
                "role": "Feature",
                "unit": "MPG",
                "description": (
                    "Combined fuel consumption in miles per gallon."
                ),
            },
            {
                "column": TARGET_COLUMN,
                "role": "Target",
                "unit": "g/km",
                "description": (
                    "Vehicle carbon dioxide emissions."
                ),
            },
        ]

        statistics = [
            {
                "label": "Total Records",
                "value": f"{record_count:,}",
                "description": (
                    "Total records available in the dataset."
                ),
                "type": "records",
            },
            {
                "label": "Engine Size",
                "value": (
                    f"{feature_statistics['ENGINESIZE']['mean']:.2f} L"
                ),
                "description": (
                    "Average engine displacement."
                ),
                "type": "engine-size",
            },
            {
                "label": "Fuel Consumption",
                "value": (
                    f"{feature_statistics['FUELCONSUMPTION_COMB_MPG']['mean']:.2f} MPG"
                ),
                "description": (
                    "Average combined fuel consumption."
                ),
                "type": "fuel-consumption",
            },
            {
                "label": "CO₂ Emissions",
                "value": (
                    f"{target_statistics['mean']:.2f} g/km"
                ),
                "description": (
                    "Average recorded CO₂ emissions."
                ),
                "type": "target",
            },
        ]

        return {
            "datasetName": self.dataset_path.name,
            "title": "CO₂ Emissions Dataset",
            "description": (
                "Dataset used by the CO₂ emission prediction "
                "machine-learning pipeline."
            ),
            "recordCount": record_count,
            "validRecordCount": valid_record_count,
            "invalidRecordCount": invalid_record_count,
            "featureCount": len(FEATURE_COLUMNS),
            "targetCount": 1,
            "validationPercentage": validation_percentage,
            "features": features,
            "statistics": statistics,
            "model": {
                "name": "CO₂ Emission Predictor",
                "description": (
                    "The dataset provides the vehicle characteristics "
                    "used by the trained machine-learning model to "
                    "estimate CO₂ emissions."
                ),
            },
            "lastUpdated": None,
        }


# ============================================================================
# SHARED SERVICE INSTANCE
# ============================================================================

try:
    dataset_service = DatasetService()

except Exception:
    logger.exception(
        "Failed to initialize the CO₂ dataset service."
    )

    raise