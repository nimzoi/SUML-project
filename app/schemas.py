"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Weather(str, Enum):
    """Allowed weather categories."""

    CLEAR = "Clear"
    RAINY = "Rainy"
    SNOWY = "Snowy"
    FOGGY = "Foggy"
    WINDY = "Windy"


class TrafficLevel(str, Enum):
    """Allowed traffic levels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TimeOfDay(str, Enum):
    """Allowed times of day."""

    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"


class VehicleType(str, Enum):
    """Allowed vehicle types."""

    BIKE = "Bike"
    SCOOTER = "Scooter"
    CAR = "Car"


class PredictRequest(BaseModel):
    """Input features for a single delivery-time prediction."""

    distance_km: float = Field(..., ge=0, examples=[7.9])
    weather: Weather = Field(..., examples=["Clear"])
    traffic_level: TrafficLevel = Field(..., examples=["Medium"])
    time_of_day: TimeOfDay = Field(..., examples=["Afternoon"])
    vehicle_type: VehicleType = Field(..., examples=["Scooter"])
    preparation_time_min: int = Field(..., ge=0, examples=[12])
    courier_experience_yrs: float = Field(..., ge=0, examples=[2.0])


class PredictResponse(BaseModel):
    """Predicted delivery time."""

    eta_minutes: float


class HealthResponse(BaseModel):
    """Liveness and model-loaded status."""

    status: str
    model_loaded: bool
