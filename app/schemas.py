"""Pydantic request/response models for the laptop price API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from model.schemas import RetrainingResult


class Company(str, Enum):
    """Allowed laptop brands (as in the dataset)."""

    ACER = "Acer"
    APPLE = "Apple"
    ASUS = "Asus"
    CHUWI = "Chuwi"
    DELL = "Dell"
    FUJITSU = "Fujitsu"
    GOOGLE = "Google"
    HP = "HP"
    HUAWEI = "Huawei"
    LG = "LG"
    LENOVO = "Lenovo"
    MSI = "MSI"
    MEDIACOM = "Mediacom"
    MICROSOFT = "Microsoft"
    RAZER = "Razer"
    SAMSUNG = "Samsung"
    TOSHIBA = "Toshiba"
    VERO = "Vero"
    XIAOMI = "Xiaomi"


class TypeName(str, Enum):
    """Allowed laptop chassis types."""

    TWO_IN_ONE = "2 in 1 Convertible"
    GAMING = "Gaming"
    NETBOOK = "Netbook"
    NOTEBOOK = "Notebook"
    ULTRABOOK = "Ultrabook"
    WORKSTATION = "Workstation"


class CpuBrand(str, Enum):
    """Allowed CPU brand/tier labels."""

    I3 = "Intel Core i3"
    I5 = "Intel Core i5"
    I7 = "Intel Core i7"
    OTHER_INTEL = "Other Intel"
    AMD = "AMD"


class GpuBrand(str, Enum):
    """Allowed GPU brands."""

    INTEL = "Intel"
    NVIDIA = "Nvidia"
    AMD = "AMD"


class Os(str, Enum):
    """Allowed operating-system groups."""

    WINDOWS = "Windows"
    MAC = "Mac"
    OTHER = "Other"


class PredictRequest(BaseModel):
    """Laptop configuration for a single price prediction."""

    company: Company = Field(..., examples=["Dell"])
    type_name: TypeName = Field(..., examples=["Notebook"])
    inches: float = Field(..., gt=0, examples=[15.6])
    ram_gb: int = Field(..., ge=0, examples=[8])
    weight_kg: float = Field(..., gt=0, examples=[1.6])
    touchscreen: int = Field(..., ge=0, le=1, examples=[0])
    ips: int = Field(..., ge=0, le=1, examples=[1])
    ppi: float = Field(..., gt=0, examples=[141.2])
    cpu_brand: CpuBrand = Field(..., examples=["Intel Core i5"])
    ssd_gb: int = Field(..., ge=0, examples=[256])
    hdd_gb: int = Field(..., ge=0, examples=[0])
    gpu_brand: GpuBrand = Field(..., examples=["Intel"])
    os: Os = Field(..., examples=["Windows"])


class PredictResponse(BaseModel):
    """Predicted laptop price (dataset currency: INR)."""

    price: float


class PriceBandResponse(BaseModel):
    """Typical-error interval around a point prediction."""

    low: float
    high: float


class ContributionResponse(BaseModel):
    """Feature contribution in original price units."""

    label: str
    amount: float


class SensitivityPoint(BaseModel):
    """Predicted price for one counterfactual feature value."""

    value: Union[int, float, str]
    price: float


class ExplainResponse(BaseModel):
    """Prediction with local explanations and a simple counterfactual sensitivity curve."""

    price: float
    typical_error: float
    band: PriceBandResponse
    contributions: List[ContributionResponse]
    sensitivity_field: str
    sensitivity: List[SensitivityPoint]


class BatchPredictRequest(BaseModel):
    """Multiple laptop configurations for one batch prediction call."""

    items: List[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    """Predicted laptop prices for a batch request."""

    prices: List[float]


class HealthResponse(BaseModel):
    """Liveness and model-loaded status."""

    status: str
    model_loaded: bool


class DataSchemaResponse(BaseModel):
    """Human-readable data and validation contract exposed by the API."""

    raw_required_columns: List[str]
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    validation_gates: Dict[str, Union[int, float, None]]


class RetrainRequest(BaseModel):
    """Optional overrides for one API-triggered retraining run."""

    time_budget_s: Optional[int] = Field(None, ge=1, le=600, examples=[60])
    min_r2: Optional[float] = Field(None, le=1.0, examples=[0.7])
    max_mae: Optional[float] = Field(None, gt=0, examples=[25000])


class RetrainJobStatus(BaseModel):
    """In-memory status of a retraining job started through the API."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    detail: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    result: Optional[RetrainingResult] = None
