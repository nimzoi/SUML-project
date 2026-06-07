"""Tests for the Streamlit UI: it imports, runs headless, presets fill, and it predicts."""

import importlib
from io import StringIO

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_ui_imports(monkeypatch):
    """The Streamlit module imports cleanly and reads API_URL from the environment."""
    monkeypatch.setenv("API_URL", "http://localhost:8000")
    module = importlib.import_module("app.ui")
    importlib.reload(module)
    assert module.API_URL == "http://localhost:8000"


def test_ui_runs_without_error():
    """The app script renders headlessly with no exception and the expected widgets."""
    app = AppTest.from_file("app/ui.py", default_timeout=30).run()
    assert app.exception == []
    assert app.session_state["ram_gb"] == 8  # seeded default


def test_ui_preset_fills_form():
    """Selecting a preset rewrites the form's session_state in one click."""
    app = AppTest.from_file("app/ui.py", default_timeout=30).run()
    app.selectbox(key="preset").set_value("Gamingowy").run()
    assert app.exception == []
    assert app.session_state["ram_gb"] == 16
    assert app.session_state["gpu_brand"] == "Nvidia"


def test_batch_sample_csv_maps_to_requests():
    """The sample batch CSV is valid and maps resolution to PPI for API requests."""
    module = importlib.import_module("app.ui")
    frame = pd.read_csv(StringIO(module._sample_batch_csv()))
    items, errors = module._prepare_batch_requests(frame)
    assert errors == []
    assert len(items) == 2
    assert items[0].ppi > 0
    assert items[1].gpu_brand.value == "Nvidia"


def test_batch_csv_reports_invalid_resolution():
    """Batch CSV validation reports row-level errors for unsupported resolution labels."""
    module = importlib.import_module("app.ui")
    frame = pd.read_csv(StringIO(module._sample_batch_csv()))
    frame.loc[0, "resolution"] = "bad"
    items, errors = module._prepare_batch_requests(frame)
    assert items
    assert errors[0]["Wiersz"] == 2
    assert "resolution" in errors[0]["Błąd"]


def test_ui_prediction_shows_pln_metric(trained_model):
    """Clicking 'Oszacuj cenę' renders a price metric headlined in PLN."""
    app = AppTest.from_file("app/ui.py", default_timeout=60).run()
    app.button(key="estimate_price").click().run()
    assert app.exception == []
    assert len(app.metric) >= 1
    assert "PLN" in app.metric[0].value
