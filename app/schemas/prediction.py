"""
Schemas used by the CO2 prediction API.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Information about a vehicle that will be used
    to generate a CO2 emission prediction.
    """

    engine_size: float = Field(
        ...,
        gt=0,
        description="Engine size in litres.",
        examples=[3.0],
    )

    fuel_consumption_mpg: float = Field(
        ...,
        gt=0,
        description="Combined fuel consumption in MPG.",
        examples=[25.0],
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction API.
    """

    predicted_co2: float = Field(
        ...,
        description="Predicted CO2 emissions in g/km.",
    )

    engine_size: float

    fuel_consumption_mpg: float

    unit: str = "g/km"