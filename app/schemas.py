"""Pydantic request/response models for the laptop price API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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


class HealthResponse(BaseModel):
    """Liveness and model-loaded status."""

    status: str
    model_loaded: bool
