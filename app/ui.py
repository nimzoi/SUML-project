"""Streamlit UI for the Laptop Price service.

Two modes, same code: it calls the FastAPI service when reachable, otherwise it loads
the model directly and predicts in-process. The standalone fallback lets the exact same
app run on Streamlit Community Cloud, where only the Streamlit process is hosted.
"""

from __future__ import annotations

import json
import os

import joblib
import requests
import streamlit as st

from app.inference import predict_price
from app.schemas import Company, CpuBrand, GpuBrand, Os, TypeName
from config import load_config
from model.train import train

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10
RESOLUTIONS = {
    "1366 x 768": (1366, 768),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "2560 x 1440": (2560, 1440),
    "2560 x 1600": (2560, 1600),
    "3840 x 2160 (4K)": (3840, 2160),
}


@st.cache_resource(show_spinner="Loading model...")
def _local_model():
    """Load the model artifact, training it once if missing (standalone mode)."""
    config = load_config()
    if not config.artifact_path.exists():
        train(config)
    return joblib.load(config.artifact_path)


def get_prediction(payload: dict):
    """Return (price, source): try the API, fall back to the local model."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["price"], "API"
    except requests.RequestException:
        return predict_price(_local_model(), payload), "model lokalny"


def get_model_info() -> dict:
    """Return model metrics/metadata from the API, or the local metrics file."""
    try:
        return requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT).json()
    except requests.RequestException:
        path = load_config().metrics_path
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {}


def main() -> None:  # pylint: disable=too-many-locals
    """Render the laptop spec form and show the predicted price."""
    st.set_page_config(page_title="Laptop Price", page_icon="💻")
    st.title("💻 Laptop Price Predictor")
    st.caption("Estimate a laptop's price from its specifications.")

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand", [e.value for e in Company])
        type_name = st.selectbox("Type", [e.value for e in TypeName])
        cpu_brand = st.selectbox("CPU", [e.value for e in CpuBrand])
        gpu_brand = st.selectbox("GPU", [e.value for e in GpuBrand])
        operating_system = st.selectbox("Operating system", [e.value for e in Os])
        inches = st.number_input("Screen size (inches)", min_value=10.0, max_value=20.0, value=15.6)
    with col2:
        ram_gb = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64], index=1)
        ssd_gb = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024], index=2)
        hdd_gb = st.selectbox("HDD (GB)", [0, 500, 1000, 2000], index=0)
        weight_kg = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.6)
        resolution = st.selectbox("Resolution", list(RESOLUTIONS), index=1)
        touchscreen = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS panel", value=True)

    width, height = RESOLUTIONS[resolution]
    ppi = round((width**2 + height**2) ** 0.5 / inches, 2)

    if st.button("Predict price"):
        payload = {
            "company": company,
            "type_name": type_name,
            "inches": inches,
            "ram_gb": ram_gb,
            "weight_kg": weight_kg,
            "touchscreen": int(touchscreen),
            "ips": int(ips),
            "ppi": ppi,
            "cpu_brand": cpu_brand,
            "ssd_gb": ssd_gb,
            "hdd_gb": hdd_gb,
            "gpu_brand": gpu_brand,
            "os": operating_system,
        }
        try:
            price, source = get_prediction(payload)
            st.success(f"Estimated price: {price:,.0f}  ·  ({source})")
        except (requests.RequestException, ValueError, KeyError) as ex:
            st.error(f"Prediction failed: {ex}")

    with st.expander("Model info & feature importance"):
        info = get_model_info()
        if info:
            st.write({k: info[k] for k in ("mae", "rmse", "r2", "best_estimator") if k in info})
            if info.get("feature_importance"):
                st.bar_chart(info["feature_importance"])
        else:
            st.warning("Model info unavailable.")


if __name__ == "__main__":
    main()
