"""
Production prediction service for the CO₂ Emission Predictor.

This module provides the application service responsible for:

- Loading the persisted machine-learning model and scaler.
- Validating prediction inputs.
- Preparing prediction data using the canonical feature definitions.
- Applying the persisted feature scaler.
- Generating CO₂ emission predictions.
- Validating prediction outputs.
- Providing structured logging for observability.

The service is intentionally independent of FastAPI so that it can be
used from API routes, background jobs, CLI commands, tests, or other
application components.

Canonical model features are imported from the preprocessing module
to ensure that training and inference always use the same feature
names and feature order.

Model features:
- ENGINESIZE
- FUELCONSUMPTION_COMB_MPG

Prediction target:
- CO2EMISSIONS

Prediction unit:
- g/km

Important:
This model provides an estimated prediction based on the training
dataset. It is not an official vehicle-emissions certification,
regulatory measurement, laboratory result, or legal determination.
"""

from __future__ import annotations

import logging
import math
from typing import Final

import numpy as np
import pandas as pd

from app.ml.model import load_model
from app.ml.preprocessing import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ============================================================================
# SERVICE CONFIGURATION
# ============================================================================

SERVICE_NAME: Final[str] = "CO2 Prediction Service"

# ---------------------------------------------------------------------------
# Input validation boundaries
# ---------------------------------------------------------------------------

# These are application-level safety boundaries, not claims about the
# physical limits of all vehicles.
MIN_ENGINE_SIZE: Final[float] = 0.1
MAX_ENGINE_SIZE: Final[float] = 20.0

MIN_FUEL_CONSUMPTION_MPG: Final[float] = 1.0
MAX_FUEL_CONSUMPTION_MPG: Final[float] = 200.0

# Number of decimal places returned by the service.
PREDICTION_DECIMAL_PLACES: Final[int] = 2


# ============================================================================
# PREDICTION SERVICE
# ============================================================================


class PredictionService:
    """
    Application service responsible for CO₂ emission predictions.

    The trained model and scaler are loaded once when the service is
    initialized and then reused for subsequent predictions.

    This avoids repeatedly loading model artifacts from disk for every
    API request.
    """

    def __init__(self) -> None:
        """
        Initialize the prediction service.

        Raises:
            RuntimeError:
                If the trained model or scaler cannot be loaded or
                does not match the configured feature set.
        """

        logger.info(
            "Initializing %s.",
            SERVICE_NAME,
        )

        try:
            self.model, self.scaler = load_model()

            self._validate_model_configuration()

        except Exception as exc:
            logger.exception(
                "Failed to initialize %s.",
                SERVICE_NAME,
            )

            raise RuntimeError(
                "The CO₂ prediction service could not be initialized."
            ) from exc

        logger.info(
            "%s initialized successfully.",
            SERVICE_NAME,
        )

    # ========================================================================
    # MODEL VALIDATION
    # ========================================================================

    def _validate_model_configuration(self) -> None:
        """
        Validate the loaded model and scaler configuration.

        This provides early failure instead of allowing malformed or
        incompatible artifacts to cause obscure errors during prediction.

        Raises:
            RuntimeError:
                If the model or scaler configuration is invalid.
        """

        if not FEATURE_COLUMNS:
            raise RuntimeError(
                "No model features have been configured."
            )

        expected_feature_count = len(FEATURE_COLUMNS)

        # --------------------------------------------------------------------
        # Validate model prediction interface
        # --------------------------------------------------------------------

        if not hasattr(self.model, "predict"):
            raise RuntimeError(
                "Loaded model does not provide a prediction interface."
            )

        # --------------------------------------------------------------------
        # Validate scaler transformation interface
        # --------------------------------------------------------------------

        if not hasattr(self.scaler, "transform"):
            raise RuntimeError(
                "Loaded scaler does not provide a transform interface."
            )

        # --------------------------------------------------------------------
        # Validate scaler feature count
        # --------------------------------------------------------------------

        scaler_feature_count = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if scaler_feature_count is None:
            raise RuntimeError(
                "Loaded scaler does not contain feature metadata."
            )

        if scaler_feature_count != expected_feature_count:
            raise RuntimeError(
                (
                    "Model scaler feature count mismatch. "
                    f"Expected {expected_feature_count}, "
                    f"received {scaler_feature_count}."
                )
            )

        # --------------------------------------------------------------------
        # Validate model feature count
        # --------------------------------------------------------------------

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count is None:
            raise RuntimeError(
                "Loaded model does not contain feature metadata."
            )

        if model_feature_count != expected_feature_count:
            raise RuntimeError(
                (
                    "Model feature count mismatch. "
                    f"Expected {expected_feature_count}, "
                    f"received {model_feature_count}."
                )
            )

        logger.debug(
            (
                "Model configuration validated successfully. "
                "Features=%s"
            ),
            list(FEATURE_COLUMNS),
        )

    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_numeric_input(
        value: float,
        field_name: str,
    ) -> float:
        """
        Validate that an input is a finite numeric value.

        Args:
            value:
                Input value to validate.

            field_name:
                Human-readable field name used in error messages.

        Returns:
            Validated value converted to float.

        Raises:
            ValueError:
                If the value is not numeric or is not finite.
        """

        # bool is a subclass of int in Python. Explicitly reject it.
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be a numeric value."
            )

        try:
            numeric_value = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be a numeric value."
            ) from exc

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"{field_name} must be a finite number."
            )

        return numeric_value

    @classmethod
    def _validate_prediction_inputs(
        cls,
        engine_size: float,
        fuel_consumption_mpg: float,
    ) -> tuple[float, float]:
        """
        Validate and normalize vehicle prediction inputs.

        Args:
            engine_size:
                Engine displacement in litres.

            fuel_consumption_mpg:
                Combined fuel consumption in MPG.

        Returns:
            Tuple containing validated engine size and fuel consumption.

        Raises:
            ValueError:
                If either input is invalid or outside the configured
                application safety boundaries.
        """

        validated_engine_size = cls._validate_numeric_input(
            engine_size,
            "engine_size",
        )

        validated_fuel_consumption = cls._validate_numeric_input(
            fuel_consumption_mpg,
            "fuel_consumption_mpg",
        )

        # --------------------------------------------------------------------
        # Engine size validation
        # --------------------------------------------------------------------

        if not (
            MIN_ENGINE_SIZE
            <= validated_engine_size
            <= MAX_ENGINE_SIZE
        ):
            raise ValueError(
                (
                    "engine_size must be between "
                    f"{MIN_ENGINE_SIZE:g} and "
                    f"{MAX_ENGINE_SIZE:g} litres."
                )
            )

        # --------------------------------------------------------------------
        # Fuel consumption validation
        # --------------------------------------------------------------------

        if not (
            MIN_FUEL_CONSUMPTION_MPG
            <= validated_fuel_consumption
            <= MAX_FUEL_CONSUMPTION_MPG
        ):
            raise ValueError(
                (
                    "fuel_consumption_mpg must be between "
                    f"{MIN_FUEL_CONSUMPTION_MPG:g} and "
                    f"{MAX_FUEL_CONSUMPTION_MPG:g} MPG."
                )
            )

        return (
            validated_engine_size,
            validated_fuel_consumption,
        )

    # ========================================================================
    # FEATURE PREPARATION
    # ========================================================================

    @staticmethod
    def _build_feature_dataframe(
        engine_size: float,
        fuel_consumption_mpg: float,
    ) -> pd.DataFrame:
        """
        Build the feature DataFrame using the canonical training order.

        The feature order comes from FEATURE_COLUMNS rather than being
        duplicated inside the service.

        This prevents training/inference feature-order mismatches.

        Args:
            engine_size:
                Engine displacement in litres.

            fuel_consumption_mpg:
                Combined fuel consumption in MPG.

        Returns:
            A single-row DataFrame ready for scaling.

        Raises:
            RuntimeError:
                If the configured feature structure is invalid.

            ValueError:
                If generated feature values are invalid.
        """

        # --------------------------------------------------------------------
        # Validate feature configuration
        # --------------------------------------------------------------------

        if len(FEATURE_COLUMNS) != 2:
            raise RuntimeError(
                (
                    "The CO₂ prediction service expects exactly two "
                    f"model features, but {len(FEATURE_COLUMNS)} were "
                    "configured."
                )
            )

        # --------------------------------------------------------------------
        # Build feature data
        # --------------------------------------------------------------------

        vehicle_data = pd.DataFrame(
            [
                {
                    FEATURE_COLUMNS[0]: engine_size,
                    FEATURE_COLUMNS[1]: fuel_consumption_mpg,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        # --------------------------------------------------------------------
        # Validate DataFrame structure
        # --------------------------------------------------------------------

        expected_shape = (
            1,
            len(FEATURE_COLUMNS),
        )

        if vehicle_data.shape != expected_shape:
            raise RuntimeError(
                (
                    "Generated prediction input has an unexpected "
                    f"shape: {vehicle_data.shape}. "
                    f"Expected: {expected_shape}."
                )
            )

        if list(vehicle_data.columns) != list(FEATURE_COLUMNS):
            raise RuntimeError(
                (
                    "Prediction feature order does not match the "
                    "configured training feature order."
                )
            )

        # --------------------------------------------------------------------
        # Validate numeric values
        # --------------------------------------------------------------------

        try:
            numeric_values = vehicle_data.to_numpy(
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Prediction input contains non-numeric values."
            ) from exc

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Prediction input contains NaN or infinite values."
            )

        return vehicle_data

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict(
        self,
        engine_size: float,
        fuel_consumption_mpg: float,
    ) -> float:
        """
        Generate a CO₂ emission prediction.

        Args:
            engine_size:
                Engine displacement in litres.

            fuel_consumption_mpg:
                Combined fuel consumption in MPG.

        Returns:
            Predicted CO₂ emissions in g/km.

        Raises:
            ValueError:
                If supplied vehicle data is invalid.

            RuntimeError:
                If the model cannot generate a valid prediction.
        """

        # --------------------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------------------

        (
            validated_engine_size,
            validated_fuel_consumption,
        ) = self._validate_prediction_inputs(
            engine_size=engine_size,
            fuel_consumption_mpg=fuel_consumption_mpg,
        )

        # --------------------------------------------------------------------
        # Build model input
        # --------------------------------------------------------------------

        vehicle_data = self._build_feature_dataframe(
            engine_size=validated_engine_size,
            fuel_consumption_mpg=validated_fuel_consumption,
        )

        logger.debug(
            "Preparing CO₂ prediction for vehicle input: %s",
            vehicle_data.to_dict(
                orient="records",
            ),
        )

        # --------------------------------------------------------------------
        # FEATURE SCALING
        # --------------------------------------------------------------------

        try:
            scaled_data = self.scaler.transform(
                vehicle_data,
            )

        except Exception as exc:
            logger.exception(
                "Failed to scale CO₂ prediction input."
            )

            raise RuntimeError(
                "Unable to preprocess the prediction input."
            ) from exc

        # --------------------------------------------------------------------
        # Validate scaled data
        # --------------------------------------------------------------------

        try:
            scaled_array = np.asarray(
                scaled_data,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The feature scaler returned invalid data."
            ) from exc

        if scaled_array.shape != (
            1,
            len(FEATURE_COLUMNS),
        ):
            raise RuntimeError(
                (
                    "The feature scaler returned data with an "
                    f"unexpected shape: {scaled_array.shape}."
                )
            )

        if not np.isfinite(scaled_array).all():
            raise RuntimeError(
                "The feature scaler produced invalid numeric values."
            )

        # --------------------------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------------------------

        try:
            predictions = self.model.predict(
                scaled_array,
            )

        except Exception as exc:
            logger.exception(
                "Machine-learning model prediction failed."
            )

            raise RuntimeError(
                "Unable to generate the CO₂ emission prediction."
            ) from exc

        # --------------------------------------------------------------------
        # VALIDATE MODEL OUTPUT
        # --------------------------------------------------------------------

        if predictions is None:
            raise RuntimeError(
                "The prediction model returned no result."
            )

        try:
            predictions_array = np.asarray(
                predictions,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The prediction model returned invalid output."
            ) from exc

        if predictions_array.size != 1:
            raise RuntimeError(
                (
                    "The prediction model returned an unexpected "
                    f"number of predictions: {predictions_array.size}."
                )
            )

        prediction = float(
            predictions_array.reshape(-1)[0]
        )

        if not math.isfinite(prediction):
            raise RuntimeError(
                "The prediction model returned an invalid numeric result."
            )

        # --------------------------------------------------------------------
        # ROUND RESULT
        # --------------------------------------------------------------------

        prediction = round(
            prediction,
            PREDICTION_DECIMAL_PLACES,
        )

        # --------------------------------------------------------------------
        # FINAL VALIDATION
        # --------------------------------------------------------------------

        if not math.isfinite(prediction):
            raise RuntimeError(
                "The final CO₂ prediction is not a valid numeric value."
            )

        # --------------------------------------------------------------------
        # LOG SUCCESS
        # --------------------------------------------------------------------

        logger.info(
            (
                "CO₂ prediction generated successfully | "
                "engine_size=%.2f L | "
                "fuel_consumption=%.2f MPG | "
                "prediction=%.2f g/km"
            ),
            validated_engine_size,
            validated_fuel_consumption,
            prediction,
        )

        return prediction


# ============================================================================
# SHARED SERVICE INSTANCE
# ============================================================================

try:
    prediction_service = PredictionService()

except Exception:
    logger.exception(
        "Fatal error while creating the CO₂ prediction service."
    )

    raise

