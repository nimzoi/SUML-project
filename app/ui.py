"""Streamlit UI for the Food Delivery ETA service.

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

from app.inference import predict_minutes
from config import load_config
from model.train import train

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10


@st.cache_resource(show_spinner="Loading model...")
def _local_model():
    """Load the model artifact, training it once if missing (standalone mode)."""
    config = load_config()
    if not config.artifact_path.exists():
        train(config)
    return joblib.load(config.artifact_path)


def get_prediction(payload: dict):
    """Return (eta_minutes, source): try the API, fall back to the local model."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["eta_minutes"], "API"
    except requests.RequestException:
        return predict_minutes(_local_model(), payload), "model lokalny"


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


def main() -> None:
    """Render the prediction form and show the result."""
    st.set_page_config(page_title="Food Delivery ETA", page_icon="🛵")
    st.title("🛵 Food Delivery ETA")
    st.caption("Estimate delivery time from order and route features.")

    distance = st.number_input("Distance (km)", min_value=0.0, value=7.9, step=0.1)
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
    traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
    vehicle = st.selectbox("Vehicle type", ["Bike", "Scooter", "Car"])
    prep = st.number_input("Preparation time (min)", min_value=0, value=12, step=1)
    experience = st.number_input("Courier experience (yrs)", min_value=0.0, value=2.0, step=0.5)

    if st.button("Predict ETA"):
        payload = {
            "distance_km": distance,
            "weather": weather,
            "traffic_level": traffic,
            "time_of_day": time_of_day,
            "vehicle_type": vehicle,
            "preparation_time_min": prep,
            "courier_experience_yrs": experience,
        }
        try:
            eta, source = get_prediction(payload)
            st.success(f"Estimated delivery time: {eta} min  ·  ({source})")
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
