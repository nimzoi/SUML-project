"""Streamlit interface for laptop price estimation.

The same UI works in two modes: it calls the FastAPI service when available, otherwise
it loads the model directly and predicts in-process. This keeps the app deployable on
Streamlit Community Cloud, where only the Streamlit process is hosted.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd
import requests
import streamlit as st
from pydantic import ValidationError

# Make the repo root importable so `streamlit run app/ui.py` works as an entry point
# (Streamlit puts only app/ on sys.path). streamlit_app.py at the repo root is the
# primary entry point; this keeps the direct app/ui.py path working identically, so a
# Streamlit Cloud (re)deploy can point at either file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# isort: off
# pylint: disable=wrong-import-position
from app.explain import explain_prediction, price_band, price_sensitivity
from app.inference import predict_price
from app.schemas import Company, CpuBrand, GpuBrand, Os, PredictRequest, TypeName
from config import load_config
from data.load import load_data
from data.monitoring import build_data_profile, compare_data_profiles
from model.schemas import DataProfile

# pylint: enable=wrong-import-position
# isort: on

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
BATCH_LIMIT = 100
BATCH_CSV_COLUMNS = [
    "company",
    "type_name",
    "inches",
    "ram_gb",
    "weight_kg",
    "touchscreen",
    "ips",
    "resolution",
    "cpu_brand",
    "ssd_gb",
    "hdd_gb",
    "gpu_brand",
    "os",
]

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
        # Lazy import: keeps flaml/xgboost out of the UI's startup import path
        # (only needed to train, which never happens on Streamlit Cloud — model is in repo).
        from model.train import train  # pylint: disable=import-outside-toplevel

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


def get_batch_predictions(items: list[PredictRequest]):
    """Return batch predictions through the API, falling back to the local model."""
    payload = {"items": [item.model_dump(mode="json") for item in items]}
    try:
        response = requests.post(f"{API_URL}/predict-batch", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["prices"], "API"
    except (requests.RequestException, KeyError):
        model = _local_model()
        return [predict_price(model, item) for item in items], "model lokalny"


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


@st.cache_data(ttl=60, show_spinner=False)
def get_data_drift() -> dict:
    """Return a data-drift report from the API or compute it locally in standalone mode."""
    try:
        response = requests.get(f"{API_URL}/data-drift", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        config = load_config()
        current_profile = build_data_profile(load_data(config), config)
        info = get_model_info()
        reference_profile = current_profile
        if info.get("data_profile"):
            reference_profile = DataProfile(**info["data_profile"])
        report = compare_data_profiles(reference_profile, current_profile)
        return report.model_dump()


def _money(value: float) -> str:
    """Format an amount with a space as the thousands separator."""
    return f"{value:,.0f}".replace(",", " ")


def _pln(inr: float) -> str:
    """Convert INR to PLN and format the result for display."""
    return _money(inr * PLN_PER_INR)


def _sample_batch_csv() -> str:
    """Return a sample batch CSV matching the upload contract."""
    sample = pd.DataFrame(
        [
            {
                "company": "Dell",
                "type_name": "Notebook",
                "inches": 15.6,
                "ram_gb": 8,
                "weight_kg": 1.6,
                "touchscreen": 0,
                "ips": 1,
                "resolution": "1920 x 1080 (Full HD)",
                "cpu_brand": "Intel Core i5",
                "ssd_gb": 256,
                "hdd_gb": 0,
                "gpu_brand": "Intel",
                "os": "Windows",
            },
            {
                "company": "MSI",
                "type_name": "Gaming",
                "inches": 15.6,
                "ram_gb": 16,
                "weight_kg": 2.3,
                "touchscreen": 0,
                "ips": 1,
                "resolution": "1920 x 1080 (Full HD)",
                "cpu_brand": "Intel Core i7",
                "ssd_gb": 512,
                "hdd_gb": 1000,
                "gpu_brand": "Nvidia",
                "os": "Windows",
            },
        ],
        columns=BATCH_CSV_COLUMNS,
    )
    return sample.to_csv(index=False)


def _row_value(row: dict, key: str):
    """Return a scalar row value, treating blank cells and NaNs as missing."""
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _as_int(value, field: str) -> int:
    """Convert a CSV value to int with a clear field-level error."""
    if value is None:
        raise ValueError(f"Brak wartości w kolumnie `{field}`.")
    return int(float(value))


def _as_float(value, field: str) -> float:
    """Convert a CSV value to float with a clear field-level error."""
    if value is None:
        raise ValueError(f"Brak wartości w kolumnie `{field}`.")
    return float(value)


def _as_binary(value, field: str) -> int:
    """Convert 0/1, bool-like and Polish yes/no cells to API binary flags."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"tak", "true", "yes", "1"}:
            return 1
        if lowered in {"nie", "false", "no", "0"}:
            return 0
    return _as_int(value, field)


def _ppi_from_batch_row(row: dict, inches: float) -> float:
    """Read ppi directly or derive it from the uploaded resolution label."""
    ppi = _row_value(row, "ppi")
    if ppi is not None:
        return float(ppi)
    resolution = _row_value(row, "resolution")
    if resolution not in RESOLUTIONS:
        raise ValueError("Kolumna `resolution` musi zawierać jedną z wartości formularza.")
    width, height = RESOLUTIONS[resolution]
    return round((width**2 + height**2) ** 0.5 / inches, 2)


def _request_from_batch_row(row: dict) -> PredictRequest:
    """Map one uploaded CSV row to the same request contract used by the API."""
    inches = _as_float(_row_value(row, "inches"), "inches")
    return PredictRequest(
        company=_row_value(row, "company"),
        type_name=_row_value(row, "type_name"),
        inches=inches,
        ram_gb=_as_int(_row_value(row, "ram_gb"), "ram_gb"),
        weight_kg=_as_float(_row_value(row, "weight_kg"), "weight_kg"),
        touchscreen=_as_binary(_row_value(row, "touchscreen"), "touchscreen"),
        ips=_as_binary(_row_value(row, "ips"), "ips"),
        ppi=_ppi_from_batch_row(row, inches),
        cpu_brand=_row_value(row, "cpu_brand"),
        ssd_gb=_as_int(_row_value(row, "ssd_gb"), "ssd_gb"),
        hdd_gb=_as_int(_row_value(row, "hdd_gb"), "hdd_gb"),
        gpu_brand=_row_value(row, "gpu_brand"),
        os=_row_value(row, "os"),
    )


def _prepare_batch_requests(frame: pd.DataFrame):
    """Validate uploaded rows and return API-ready requests plus row-level errors."""
    if len(frame) > BATCH_LIMIT:
        return [], [{"Wiersz": "-", "Błąd": f"CSV może mieć maksymalnie {BATCH_LIMIT} rekordów."}]

    items = []
    errors = []
    for index, row in frame.iterrows():
        try:
            items.append(_request_from_batch_row(row.to_dict()))
        except (ValueError, TypeError, ValidationError) as ex:
            errors.append({"Wiersz": int(index) + 2, "Błąd": str(ex).splitlines()[0]})
    return items, errors


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


def _render_drift_report() -> None:
    """Render current training-data drift against the stored reference profile."""
    report = get_data_drift()
    status = "OK" if report.get("ok") else "Wykryto drift"
    st.write(
        {
            "Status": status,
            "Cechy z driftem": report.get("drifted_features", 0),
            "Rekordy referencyjne": report.get("reference_rows", 0),
            "Rekordy aktualne": report.get("current_rows", 0),
        }
    )
    features = report.get("features", [])
    if features:
        rows = [
            {
                "Cecha": item["feature"],
                "Typ": "numeryczna" if item["kind"] == "numeric" else "kategoryczna",
                "Score": item["score"],
                "Próg": item["threshold"],
                "Drift": "tak" if item["drifted"] else "nie",
            }
            for item in features[:8]
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(report.get("detail", ""))


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


def _render_batch_upload() -> None:
    """Render CSV upload, batch pricing and downloadable results."""
    with st.expander("📦 Wycena wsadowa CSV"):
        st.download_button(
            "Pobierz przykładowy CSV",
            data=_sample_batch_csv().encode("utf-8"),
            file_name="laptops_batch_sample.csv",
            mime="text/csv",
            key="download_batch_sample",
        )
        uploaded = st.file_uploader("CSV z konfiguracjami laptopów", type=["csv"])
        if uploaded is None:
            return

        try:
            frame = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError) as ex:
            st.error(f"Nie udało się odczytać CSV: {ex}")
            return
        if frame.empty:
            st.warning("CSV nie zawiera żadnych rekordów.")
            return

        items, errors = _prepare_batch_requests(frame)
        if errors:
            st.error("Część wierszy nie spełnia kontraktu wejściowego.")
            st.dataframe(errors, hide_index=True, use_container_width=True)
            return

        if st.button("Wyceń CSV", type="primary", key="batch_predict"):
            prices, source = get_batch_predictions(items)
            info = get_model_info()
            mae = float(info["mae"]) if info and "mae" in info else 0.0
            result = frame.copy()
            result["price_inr"] = [round(price, 2) for price in prices]
            result["price_pln"] = [round(price * PLN_PER_INR) for price in prices]
            if mae:
                result["band_low_pln"] = [
                    round(max(0.0, price - mae) * PLN_PER_INR) for price in prices
                ]
                result["band_high_pln"] = [round((price + mae) * PLN_PER_INR) for price in prices]
            st.caption(f"Wyceniono {len(result)} rekordów · źródło: {source}")
            st.dataframe(result, hide_index=True, use_container_width=True)
            st.download_button(
                "Pobierz wyniki CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="laptops_batch_predictions.csv",
                mime="text/csv",
                key="download_batch_results",
            )


def main() -> None:
    """Render the page: form, price estimate with band and information panels."""
    st.set_page_config(page_title="Wycena laptopa", page_icon="💻")
    st.title("💻 Wycena laptopa")
    st.caption(
        "Szybka wycena laptopa na podstawie jego specyfikacji — wpisz parametry, poznaj cenę."
    )

    _init_defaults()
    request = _build_form()

    if st.button("Oszacuj cenę", type="primary", key="estimate_price"):
        _render_result(request)

    _render_batch_upload()

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
            labels = {
                "r2": "Trafność (R²)",
                "mae": "Średni błąd (MAE)",
                "best_estimator": "Model",
                "mlflow_run_id": "MLflow run",
            }
            st.write({label: info[key] for key, label in labels.items() if key in info})
            if info.get("feature_importance"):
                st.caption("Co najbardziej wpływa na cenę:")
                st.bar_chart(info["feature_importance"])
        else:
            st.warning("Informacje o modelu są chwilowo niedostępne.")

    with st.expander("🩺 Monitoring danych"):
        _render_drift_report()


if __name__ == "__main__":
    main()
