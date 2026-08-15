"""
===============================================================================
CO₂ EMISSION PREDICTOR
Production-grade machine-learning model management.
===============================================================================

Responsibilities
----------------
This module is responsible for:

- Loading training data.
- Training the regression model.
- Scaling model features.
- Evaluating model performance.
- Persisting model and scaler artifacts.
- Persisting training metadata.
- Loading existing model artifacts.
- Validating model artifacts.
- Dynamically evaluating the loaded model.
- Dynamically calculating dataset statistics.
- Providing model metadata to FastAPI.
- Automatically training the model when artifacts are unavailable.

IMPORTANT
---------
Evaluation metrics are NEVER hardcoded.

The following values are calculated dynamically:

- R²
- MAE
- MSE
- RMSE
- Dataset record count
- Feature ranges
- Feature means
- Feature medians
- Feature standard deviations
- Target range
- Target mean
- Target median
- Target standard deviation

The persisted metrics.json file is metadata/cache only and is not
used as the source of truth for API evaluation metrics.

The trained model is intended for prediction purposes only.
It must not be interpreted as an official vehicle-emissions certification,
regulatory measurement, laboratory result, or legal emissions determination.
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_training_data,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY: Final[Path] = (
    BASE_DIR / "models"
)

MODEL_FILE: Final[Path] = (
    MODEL_DIRECTORY / "model.pkl"
)

SCALER_FILE: Final[Path] = (
    MODEL_DIRECTORY / "scaler.pkl"
)

METRICS_FILE: Final[Path] = (
    MODEL_DIRECTORY / "metrics.json"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME: Final[str] = (
    "CO₂ Emission Predictor"
)

MODEL_VERSION: Final[str] = "1.0.0"

ALGORITHM: Final[str] = (
    "Multiple Linear Regression"
)

TARGET_UNIT: Final[str] = "g/km"

TEST_SIZE: Final[float] = 0.20

RANDOM_STATE: Final[int] = 42

MINIMUM_TRAINING_SAMPLES: Final[int] = 10


# ============================================================================
# MODEL METRICS
# ============================================================================

@dataclass(frozen=True)
class ModelMetrics:
    """
    Dynamically calculated model evaluation metrics.
    """

    r2: float
    mae: float
    mse: float
    rmse: float

    training_samples: int
    testing_samples: int
    total_samples: int

    feature_count: int
    test_size: float
    random_state: int

    model_name: str
    model_version: str
    algorithm: str


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_training_arrays(
    X: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize training arrays.
    """

    X_array = np.asarray(
        X,
        dtype=np.float64,
    )

    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    if X_array.ndim != 2:
        raise ValueError(
            "Training features must be two-dimensional."
        )

    if y_array.ndim != 1:
        y_array = y_array.reshape(-1)

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "Feature and target datasets have different lengths."
        )

    expected_features = len(
        FEATURE_COLUMNS
    )

    if expected_features == 0:
        raise ValueError(
            "No model features have been configured."
        )

    if X_array.shape[1] != expected_features:
        raise ValueError(
            "Unexpected number of model features. "
            f"Expected {expected_features}, "
            f"received {X_array.shape[1]}."
        )

    if X_array.shape[0] < MINIMUM_TRAINING_SAMPLES:
        raise ValueError(
            "Training dataset contains too few records."
        )

    if not np.isfinite(X_array).all():
        raise ValueError(
            "Training features contain NaN or infinite values."
        )

    if not np.isfinite(y_array).all():
        raise ValueError(
            "Training targets contain NaN or infinite values."
        )

    return (
        X_array,
        y_array,
    )


def _validate_model_artifact(
    model: Any,
    scaler: Any,
) -> None:
    """
    Validate persisted model and scaler.
    """

    if not isinstance(
        model,
        LinearRegression,
    ):
        raise RuntimeError(
            "Saved model is not a LinearRegression model."
        )

    if not isinstance(
        scaler,
        StandardScaler,
    ):
        raise RuntimeError(
            "Saved scaler is not a StandardScaler."
        )

    expected_features = len(
        FEATURE_COLUMNS
    )

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    if scaler_features != expected_features:
        raise RuntimeError(
            "Saved scaler feature count does not match "
            "the configured feature count."
        )

    model_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_features != expected_features:
        raise RuntimeError(
            "Saved model feature count does not match "
            "the configured feature count."
        )

    if not hasattr(
        model,
        "coef_",
    ):
        raise RuntimeError(
            "Saved LinearRegression model is not fitted."
        )

    coefficients = np.asarray(
        model.coef_,
        dtype=np.float64,
    )

    if not np.isfinite(
        coefficients
    ).all():
        raise RuntimeError(
            "Saved model contains invalid coefficients."
        )

    intercept = np.asarray(
        model.intercept_,
        dtype=np.float64,
    )

    if not np.isfinite(
        intercept
    ).all():
        raise RuntimeError(
            "Saved model contains an invalid intercept."
        )


# ============================================================================
# ATOMIC JSON WRITER
# ============================================================================

def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Safely write JSON metadata atomically.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as temporary_file:

            json.dump(
                payload,
                temporary_file,
                indent=4,
                ensure_ascii=False,
            )

            temporary_file.write("\n")

            temporary_file.flush()

            os.fsync(
                temporary_file.fileno()
            )

        temporary_path.replace(
            path
        )

    except Exception:

        try:
            temporary_path.unlink(
                missing_ok=True
            )

        except OSError:
            logger.warning(
                "Unable to remove temporary file: %s",
                temporary_path,
            )

        raise


# ============================================================================
# TRAINING
# ============================================================================

def train_model() -> tuple[
    LinearRegression,
    StandardScaler,
    ModelMetrics,
]:
    """
    Train and evaluate the machine-learning model.
    """

    logger.info(
        "Starting CO₂ emission model training."
    )

    X, y, cleaned_data = (
        get_training_data()
    )

    X_array, y_array = (
        _validate_training_arrays(
            X,
            y,
        )
    )

    if len(cleaned_data) != len(
        X_array
    ):
        raise RuntimeError(
            "Cleaned dataset size does not "
            "match training feature matrix."
        )

    logger.info(
        "Loaded %d valid records with %d features.",
        len(X_array),
        X_array.shape[1],
    )

    # ------------------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_array,
            y_array,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    if len(X_train) == 0:
        raise ValueError(
            "Training split produced zero samples."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Testing split produced zero samples."
        )

    # ------------------------------------------------------------------------
    # SCALING
    # ------------------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # ------------------------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train,
    )

    logger.info(
        "%s training completed.",
        ALGORITHM,
    )

    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "Model produced invalid predictions."
        )

    # ------------------------------------------------------------------------
    # DYNAMIC EVALUATION
    # ------------------------------------------------------------------------

    mae = float(
        mean_absolute_error(
            y_test,
            predictions,
        )
    )

    mse = float(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    values = (
        r2,
        mae,
        mse,
        rmse,
    )

    if not all(
        np.isfinite(value)
        for value in values
    ):
        raise RuntimeError(
            "Model evaluation produced invalid metrics."
        )

    metrics = ModelMetrics(
        r2=r2,
        mae=mae,
        mse=mse,
        rmse=rmse,
        training_samples=len(
            X_train
        ),
        testing_samples=len(
            X_test
        ),
        total_samples=len(
            X_array
        ),
        feature_count=len(
            FEATURE_COLUMNS
        ),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        algorithm=ALGORITHM,
    )

    logger.info(
        (
            "Evaluation completed | "
            "R²=%.6f | MAE=%.6f | RMSE=%.6f"
        ),
        metrics.r2,
        metrics.mae,
        metrics.rmse,
    )

    return (
        model,
        scaler,
        metrics,
    )


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(
    model: LinearRegression,
    scaler: StandardScaler,
    metrics: ModelMetrics | None = None,
) -> None:
    """
    Persist model, scaler, and metadata.
    """

    _validate_model_artifact(
        model,
        scaler,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        joblib.dump(
            model,
            MODEL_FILE,
        )

        joblib.dump(
            scaler,
            SCALER_FILE,
        )

    except Exception as exc:

        logger.exception(
            "Failed to save model artifacts."
        )

        raise RuntimeError(
            "Unable to save model artifacts."
        ) from exc

    if metrics is not None:

        payload = {
            **asdict(metrics),
            "features": list(
                FEATURE_COLUMNS
            ),
            "target": TARGET_COLUMN,
            "target_unit": TARGET_UNIT,
        }

        _atomic_write_json(
            METRICS_FILE,
            payload,
        )

    logger.info(
        "Model artifacts saved successfully."
    )


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model() -> tuple[
    LinearRegression,
    StandardScaler,
]:
    """
    Load persisted model and scaler.

    If artifacts do not exist, train a new model.
    """

    model_exists = (
        MODEL_FILE.is_file()
    )

    scaler_exists = (
        SCALER_FILE.is_file()
    )

    if not model_exists or not scaler_exists:

        logger.warning(
            "Model artifacts are missing. "
            "Training a new model."
        )

        model, scaler, metrics = (
            train_model()
        )

        save_model(
            model,
            scaler,
            metrics,
        )

        return (
            model,
            scaler,
        )

    try:

        model = joblib.load(
            MODEL_FILE
        )

        scaler = joblib.load(
            SCALER_FILE
        )

    except Exception as exc:

        logger.exception(
            "Unable to load model artifacts."
        )

        raise RuntimeError(
            "Unable to load prediction model."
        ) from exc

    _validate_model_artifact(
        model,
        scaler,
    )

    return (
        model,
        scaler,
    )


# ============================================================================
# DYNAMIC MODEL EVALUATION
# ============================================================================

def _evaluate_loaded_model(
    model: LinearRegression,
    scaler: StandardScaler,
) -> ModelMetrics:
    """
    Evaluate the currently loaded model against the current dataset.

    Metrics are calculated dynamically.
    """

    X, y, cleaned_data = (
        get_training_data()
    )

    X_array, y_array = (
        _validate_training_arrays(
            X,
            y,
        )
    )

    if len(cleaned_data) != len(
        X_array
    ):
        raise RuntimeError(
            "Dataset size mismatch."
        )

    # ------------------------------------------------------------------------
    # SAME DETERMINISTIC SPLIT
    # ------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_array,
            y_array,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    if len(X_test) == 0:
        raise RuntimeError(
            "Evaluation dataset contains zero testing samples."
        )

    # ------------------------------------------------------------------------
    # PERSISTED SCALER
    # ------------------------------------------------------------------------

    X_test_scaled = scaler.transform(
        X_test
    )

    if not np.isfinite(
        X_test_scaled
    ).all():
        raise RuntimeError(
            "Scaler produced invalid evaluation values."
        )

    # ------------------------------------------------------------------------
    # PERSISTED MODEL
    # ------------------------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if len(predictions) != len(
        y_test
    ):
        raise RuntimeError(
            "Unexpected number of predictions."
        )

    # ------------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------------

    mae = float(
        mean_absolute_error(
            y_test,
            predictions,
        )
    )

    mse = float(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    return ModelMetrics(
        r2=r2,
        mae=mae,
        mse=mse,
        rmse=rmse,
        training_samples=len(
            X_train
        ),
        testing_samples=len(
            X_test
        ),
        total_samples=len(
            X_array
        ),
        feature_count=len(
            FEATURE_COLUMNS
        ),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        algorithm=ALGORITHM,
    )


# ============================================================================
# DYNAMIC DATASET STATISTICS
# ============================================================================

def _get_dynamic_dataset_statistics() -> dict[str, Any]:
    """
    Calculate dataset statistics directly from the current dataset.

    Nothing here is hardcoded.
    """

    X, y, cleaned_data = (
        get_training_data()
    )

    X_array, y_array = (
        _validate_training_arrays(
            X,
            y,
        )
    )

    statistics: dict[str, Any] = {
        "records": int(
            len(cleaned_data)
        ),

        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),

        "features": list(
            FEATURE_COLUMNS
        ),

        "target": TARGET_COLUMN,
    }

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

    feature_statistics: dict[
        str,
        dict[str, float],
    ] = {}

    for index, feature_name in enumerate(
        FEATURE_COLUMNS
    ):

        values = X_array[
            :,
            index,
        ]

        feature_statistics[
            feature_name
        ] = {
            "min": float(
                np.min(values)
            ),

            "max": float(
                np.max(values)
            ),

            "mean": float(
                np.mean(values)
            ),

            "median": float(
                np.median(values)
            ),

            "std": float(
                np.std(values)
            ),
        }

    statistics[
        "feature_statistics"
    ] = feature_statistics

    # ------------------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------------------

    statistics[
        "target_statistics"
    ] = {
        "column": TARGET_COLUMN,

        "min": float(
            np.min(y_array)
        ),

        "max": float(
            np.max(y_array)
        ),

        "mean": float(
            np.mean(y_array)
        ),

        "median": float(
            np.median(y_array)
        ),

        "std": float(
            np.std(y_array)
        ),
    }

    return statistics


# ============================================================================
# MODEL METADATA
# ============================================================================

def get_model_metadata() -> dict[str, Any]:
    """
    Generate complete dynamic model metadata.
    """

    logger.info(
        "Generating dynamic model metadata."
    )

    model, scaler = load_model()

    metrics = _evaluate_loaded_model(
        model,
        scaler,
    )

    dataset_statistics = (
        _get_dynamic_dataset_statistics()
    )

    metadata: dict[str, Any] = {

        "model_name": MODEL_NAME,

        "algorithm": ALGORITHM,

        "version": MODEL_VERSION,

        "features": list(
            FEATURE_COLUMNS
        ),

        "target": TARGET_COLUMN,

        "target_unit": TARGET_UNIT,

        "metrics": {
            "r2_score": metrics.r2,
            "mae": metrics.mae,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
        },

        "evaluation": {
            "training_samples":
                metrics.training_samples,

            "testing_samples":
                metrics.testing_samples,

            "total_samples":
                metrics.total_samples,

            "feature_count":
                metrics.feature_count,

            "test_size":
                metrics.test_size,

            "random_state":
                metrics.random_state,
        },

        "dataset":
            dataset_statistics,

        "artifacts": {
            "model_file":
                MODEL_FILE.name,

            "scaler_file":
                SCALER_FILE.name,
        },
    }

    logger.info(
        (
            "Metadata generated | "
            "R²=%.6f | MAE=%.6f | RMSE=%.6f | "
            "records=%d"
        ),
        metrics.r2,
        metrics.mae,
        metrics.rmse,
        dataset_statistics["records"],
    )

    return metadata


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_model() -> tuple[
    LinearRegression,
    StandardScaler,
]:
    """
    Initialize the prediction model.
    """

    logger.info(
        "Initializing CO₂ prediction model."
    )

    model, scaler = load_model()

    logger.info(
        "CO₂ prediction model initialized successfully."
    )

    return (
        model,
        scaler,
    )