"""Interfejs Streamlit do wyceny laptopa.

Dwa tryby, ten sam kod: woła usługę FastAPI, a gdy API jest niedostępne, ładuje model
bezpośrednio i liczy w procesie. Dzięki temu ta sama aplikacja działa też samodzielnie
na Streamlit Community Cloud (gdzie hostowany jest tylko proces Streamlit).
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
PLN_PER_INR = 0.045  # kurs orientacyjny, tylko do poglądowego przeliczenia
RESOLUTIONS = {
    "1366 x 768": (1366, 768),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "2560 x 1440": (2560, 1440),
    "2560 x 1600": (2560, 1600),
    "3840 x 2160 (4K)": (3840, 2160),
}


@st.cache_resource(show_spinner="Ładowanie modelu...")
def _local_model():
    """Załaduj artefakt modelu, trenując go raz, jeśli go brakuje (tryb standalone)."""
    config = load_config()
    if not config.artifact_path.exists():
        train(config)
    return joblib.load(config.artifact_path)


def get_prediction(payload: dict):
    """Zwróć (cena, źródło): najpierw API, w razie braku — model lokalny."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["price"], "API"
    except requests.RequestException:
        return predict_price(_local_model(), payload), "model lokalny"


def get_model_info() -> dict:
    """Zwróć metryki/metadane modelu z API lub z lokalnego pliku metryk."""
    try:
        return requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT).json()
    except requests.RequestException:
        path = load_config().metrics_path
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {}


def _money(value: float) -> str:
    """Sformatuj kwotę ze spacją jako separatorem tysięcy."""
    return f"{value:,.0f}".replace(",", " ")


def main() -> None:  # pylint: disable=too-many-locals
    """Wyrenderuj formularz specyfikacji i pokaż szacowaną cenę."""
    st.set_page_config(page_title="Wycena laptopa", page_icon="💻")
    st.title("💻 Wycena laptopa")
    st.caption(
        "Szybka wycena laptopa na podstawie jego specyfikacji — wpisz parametry, poznaj cenę."
    )

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Marka", [e.value for e in Company])
        type_name = st.selectbox("Typ", [e.value for e in TypeName])
        cpu_brand = st.selectbox("Procesor (CPU)", [e.value for e in CpuBrand])
        gpu_brand = st.selectbox("Karta graficzna (GPU)", [e.value for e in GpuBrand])
        operating_system = st.selectbox("System operacyjny", [e.value for e in Os])
        inches = st.number_input(
            "Przekątna ekranu (cale)", min_value=10.0, max_value=20.0, value=15.6
        )
    with col2:
        ram_gb = st.selectbox("Pamięć RAM (GB)", [4, 8, 12, 16, 24, 32, 64], index=1)
        ssd_gb = st.selectbox("Dysk SSD (GB)", [0, 128, 256, 512, 1024], index=2)
        hdd_gb = st.selectbox("Dysk HDD (GB)", [0, 500, 1000, 2000], index=0)
        weight_kg = st.number_input("Waga (kg)", min_value=0.5, max_value=5.0, value=1.6)
        resolution = st.selectbox("Rozdzielczość ekranu", list(RESOLUTIONS), index=1)
        touchscreen = st.checkbox("Ekran dotykowy")
        ips = st.checkbox("Panel IPS", value=True)

    width, height = RESOLUTIONS[resolution]
    ppi = round((width**2 + height**2) ** 0.5 / inches, 2)

    if st.button("Oszacuj cenę", type="primary"):
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
            st.metric("Szacowana cena", f"{_money(price)} INR")
            st.caption(
                f"≈ {_money(price * PLN_PER_INR)} PLN (kurs orientacyjny) · źródło: {source}"
            )
        except (requests.RequestException, ValueError, KeyError) as ex:
            st.error(f"Nie udało się policzyć ceny: {ex}")

    with st.expander("ℹ️ Jak to działa i zastosowania"):
        st.markdown(
            "- Wpisujesz specyfikację — model w sekundę zwraca szacowaną cenę.\n"
            "- Model jest wytrenowany na danych rynkowych (AutoML) i **można go dotrenować "
            "na nowych danych bez zmian w kodzie** (wystarczy podmienić plik z danymi).\n"
            "- Zastosowania: szybka wycena oferty, weryfikacja czy cena jest rynkowa, "
            "wsparcie skupu i sprzedaży sprzętu używanego.\n"
            "- Dostępne jako strona (ten interfejs) oraz jako API do integracji z innymi systemami."
        )

    with st.expander("📊 Informacje o modelu"):
        info = get_model_info()
        if info:
            labels = {"r2": "Trafność (R²)", "mae": "Średni błąd (MAE)", "best_estimator": "Model"}
            st.write({label: info[key] for key, label in labels.items() if key in info})
            if info.get("feature_importance"):
                st.caption("Co najbardziej wpływa na cenę:")
                st.bar_chart(info["feature_importance"])
        else:
            st.warning("Informacje o modelu są chwilowo niedostępne.")


if __name__ == "__main__":
    main()
