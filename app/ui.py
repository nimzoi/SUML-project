"""Streamlit interface for laptop price estimation.

The same UI works in two modes: it calls the FastAPI service when available, otherwise
it loads the model directly and predicts in-process. This keeps the app deployable on
Streamlit Community Cloud, where only the Streamlit process is hosted.
"""

from __future__ import annotations

import json
import os

import joblib
import requests
import streamlit as st

from app.explain import explain_prediction, price_band, price_sensitivity
from app.inference import predict_price
from app.schemas import Company, CpuBrand, GpuBrand, Os, PredictRequest, TypeName
from config import load_config
from model.train import train

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10
PLN_PER_INR = 0.045  # Approximate rate, used only for indicative conversion.
DATA_NOTE = (
    "Model wytrenowano na publicznym zbiorze Kaggle (rynek indyjski, ok. 2017 r.). "
    "Ceny w INR przeliczane na PLN po kursie orientacyjnym — traktuj wynik jako wycenę "
    "porównawczą, nie ofertową."
)
RESOLUTIONS = {
    "1366 x 768": (1366, 768),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "2560 x 1440": (2560, 1440),
    "2560 x 1600": (2560, 1600),
    "3840 x 2160 (4K)": (3840, 2160),
}
RAM_OPTIONS = [4, 8, 12, 16, 24, 32, 64]
SSD_OPTIONS = [0, 128, 256, 512, 1024]
HDD_OPTIONS = [0, 500, 1000, 2000]

# Defaults seeded into session_state so the form widgets can be preset-driven without
# passing index=/value= (which would clash with the Session State API).
DEFAULTS = {
    "company": "Dell",
    "type_name": "Notebook",
    "cpu_brand": "Intel Core i5",
    "gpu_brand": "Intel",
    "os": "Windows",
    "inches": 15.6,
    "ram_gb": 8,
    "ssd_gb": 256,
    "hdd_gb": 0,
    "weight_kg": 1.6,
    "resolution": "1920 x 1080 (Full HD)",
    "touchscreen": False,
    "ips": True,
}

# One-click example builds; each maps directly onto the form's session_state keys.
PRESETS = {
    "— własna konfiguracja —": None,
    "Budżetowy": {
        "company": "Acer",
        "type_name": "Notebook",
        "cpu_brand": "Other Intel",
        "gpu_brand": "Intel",
        "os": "Windows",
        "inches": 15.6,
        "ram_gb": 4,
        "ssd_gb": 128,
        "hdd_gb": 0,
        "weight_kg": 2.0,
        "resolution": "1366 x 768",
        "touchscreen": False,
        "ips": False,
    },
    "Biznesowy ultrabook": {
        "company": "Dell",
        "type_name": "Ultrabook",
        "cpu_brand": "Intel Core i7",
        "gpu_brand": "Intel",
        "os": "Windows",
        "inches": 14.0,
        "ram_gb": 16,
        "ssd_gb": 512,
        "hdd_gb": 0,
        "weight_kg": 1.3,
        "resolution": "1920 x 1080 (Full HD)",
        "touchscreen": False,
        "ips": True,
    },
    "Gamingowy": {
        "company": "MSI",
        "type_name": "Gaming",
        "cpu_brand": "Intel Core i7",
        "gpu_brand": "Nvidia",
        "os": "Windows",
        "inches": 15.6,
        "ram_gb": 16,
        "ssd_gb": 512,
        "hdd_gb": 1000,
        "weight_kg": 2.3,
        "resolution": "1920 x 1080 (Full HD)",
        "touchscreen": False,
        "ips": True,
    },
}


@st.cache_resource(show_spinner="Ładowanie modelu...")
def _local_model():
    """Load the model artifact, training it once if missing in standalone mode."""
    config = load_config()
    if not config.artifact_path.exists():
        train(config)
    return joblib.load(config.artifact_path)


def get_prediction(request: PredictRequest):
    """Return (price, source): API first, local model fallback if unavailable."""
    try:
        response = requests.post(
            f"{API_URL}/predict", json=request.model_dump(mode="json"), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["price"], "API"
    except requests.RequestException:
        return predict_price(_local_model(), request), "model lokalny"


def get_model_info() -> dict:
    """Return model metrics/metadata from the API or the local metrics file."""
    try:
        return requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT).json()
    except requests.RequestException:
        path = load_config().metrics_path
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {}


def _money(value: float) -> str:
    """Format an amount with a space as the thousands separator."""
    return f"{value:,.0f}".replace(",", " ")


def _pln(inr: float) -> str:
    """Convert INR to PLN and format the result for display."""
    return _money(inr * PLN_PER_INR)


def _init_defaults() -> None:
    """Seed session_state with default form values once if they are missing."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _apply_preset() -> None:
    """Fill the form with the selected preset values from the selector callback."""
    preset = PRESETS.get(st.session_state.get("preset"))
    if preset:
        for key, value in preset.items():
            st.session_state[key] = value


def _render_explanation(request: PredictRequest) -> None:
    """Render feature contributions against the baseline laptop in PLN."""
    contributions = explain_prediction(_local_model(), request)[:6]
    if not contributions:
        return
    st.markdown("**💡 Dlaczego ta cena? Wpływ cech względem laptopa bazowego (PLN):**")
    st.bar_chart({item.label: round(item.amount * PLN_PER_INR) for item in contributions})


def _render_sensitivity(request: PredictRequest) -> None:
    """Render RAM sensitivity while keeping the remaining parameters fixed."""
    prices = price_sensitivity(_local_model(), request, "ram_gb", RAM_OPTIONS)
    st.markdown("**📈 Co jeśli więcej RAM? Szacowana cena wg RAM (PLN):**")
    st.bar_chart({f"{ram} GB": round(price * PLN_PER_INR) for ram, price in prices.items()})


def _build_form() -> PredictRequest:  # pylint: disable=too-many-locals
    """Render the specification form and return a validated PredictRequest."""
    st.selectbox("Przykładowa konfiguracja", list(PRESETS), key="preset", on_change=_apply_preset)

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Marka", [e.value for e in Company], key="company")
        type_name = st.selectbox("Typ", [e.value for e in TypeName], key="type_name")
        cpu_brand = st.selectbox("Procesor (CPU)", [e.value for e in CpuBrand], key="cpu_brand")
        gpu_brand = st.selectbox(
            "Karta graficzna (GPU)", [e.value for e in GpuBrand], key="gpu_brand"
        )
        operating_system = st.selectbox("System operacyjny", [e.value for e in Os], key="os")
        inches = st.number_input(
            "Przekątna ekranu (cale)", min_value=10.0, max_value=20.0, key="inches"
        )
    with col2:
        ram_gb = st.selectbox("Pamięć RAM (GB)", RAM_OPTIONS, key="ram_gb")
        ssd_gb = st.selectbox("Dysk SSD (GB)", SSD_OPTIONS, key="ssd_gb")
        hdd_gb = st.selectbox("Dysk HDD (GB)", HDD_OPTIONS, key="hdd_gb")
        weight_kg = st.number_input("Waga (kg)", min_value=0.5, max_value=5.0, key="weight_kg")
        resolution = st.selectbox("Rozdzielczość ekranu", list(RESOLUTIONS), key="resolution")
        touchscreen = st.checkbox("Ekran dotykowy", key="touchscreen")
        ips = st.checkbox("Panel IPS", key="ips")

    width, height = RESOLUTIONS[resolution]
    ppi = round((width**2 + height**2) ** 0.5 / inches, 2)
    return PredictRequest(
        company=company,
        type_name=type_name,
        inches=inches,
        ram_gb=ram_gb,
        weight_kg=weight_kg,
        touchscreen=int(touchscreen),
        ips=int(ips),
        ppi=ppi,
        cpu_brand=cpu_brand,
        ssd_gb=ssd_gb,
        hdd_gb=hdd_gb,
        gpu_brand=gpu_brand,
        os=operating_system,
    )


def _render_result(request: PredictRequest) -> None:
    """Calculate price with a band, then render explanation and RAM scenario."""
    try:
        price, source = get_prediction(request)
        info = get_model_info()
        mae = float(info["mae"]) if info and "mae" in info else price * 0.15
        band = price_band(price, mae)
        st.metric("Szacowana cena", f"{_pln(price)} PLN")
        st.caption(
            f"Przedział: {_pln(band.low)}–{_pln(band.high)} PLN · ≈ {_money(price)} INR · "
            f"źródło: {source} · wycena porównawcza"
        )
        _render_explanation(request)
        _render_sensitivity(request)
    except (requests.RequestException, ValueError, KeyError) as ex:
        st.error(f"Nie udało się policzyć ceny: {ex}")


def main() -> None:
    """Render the page: form, price estimate with band and information panels."""
    st.set_page_config(page_title="Wycena laptopa", page_icon="💻")
    st.title("💻 Wycena laptopa")
    st.caption(
        "Szybka wycena laptopa na podstawie jego specyfikacji — wpisz parametry, poznaj cenę."
    )

    _init_defaults()
    request = _build_form()

    if st.button("Oszacuj cenę", type="primary"):
        _render_result(request)

    with st.expander("ℹ️ Jak to działa i zastosowania"):
        st.markdown(
            "- Wpisujesz specyfikację — model w sekundę zwraca szacowaną cenę wraz z przedziałem.\n"
            "- Panel **„Dlaczego ta cena?”** pokazuje, ile każda cecha dokłada względem "
            "laptopa bazowego (entry-level).\n"
            "- Model jest wytrenowany na danych rynkowych (AutoML) i **można go dotrenować "
            "na nowych danych bez zmian w kodzie** (wystarczy podmienić plik z danymi).\n"
            "- Zastosowania: szybka wycena oferty, weryfikacja czy cena jest rynkowa, "
            "wsparcie skupu i sprzedaży sprzętu używanego.\n"
            f"- {DATA_NOTE}"
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
