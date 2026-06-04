"""Smoke test: the Streamlit script imports without error."""

import importlib


def test_ui_imports(monkeypatch):
    monkeypatch.setenv("API_URL", "http://localhost:8000")
    module = importlib.import_module("app.ui")
    importlib.reload(module)
    assert module.API_URL == "http://localhost:8000"
