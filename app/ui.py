"""Streamlit UI for the Food Delivery ETA service."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10


def main() -> None:
    """Render the prediction form and call the API."""
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
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            st.success(f"Estimated delivery time: {response.json()['eta_minutes']} min")
        except requests.RequestException as ex:
            st.error(f"Prediction failed: {ex}")

    with st.expander("Model info & feature importance"):
        try:
            info = requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT).json()
            st.write({k: info[k] for k in ("mae", "rmse", "r2", "best_estimator") if k in info})
            if info.get("feature_importance"):
                st.bar_chart(info["feature_importance"])
        except requests.RequestException as ex:
            st.warning(f"Could not load model info: {ex}")


if __name__ == "__main__":
    main()
