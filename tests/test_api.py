"""Tests for the FastAPI service."""

from fastapi.testclient import TestClient


def _client():
    from app.api import app

    return TestClient(app)


def _valid_payload():
    return {
        "company": "Dell",
        "type_name": "Notebook",
        "inches": 15.6,
        "ram_gb": 8,
        "weight_kg": 1.6,
        "touchscreen": 0,
        "ips": 1,
        "ppi": 141.2,
        "cpu_brand": "Intel Core i5",
        "ssd_gb": 256,
        "hdd_gb": 0,
        "gpu_brand": "Intel",
        "os": "Windows",
    }


def test_health_ok(trained_model):
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_price(trained_model):
    response = _client().post("/predict", json=_valid_payload())
    assert response.status_code == 200
    assert response.json()["price"] > 0


def test_predict_rejects_invalid(trained_model):
    response = _client().post("/predict", json={"ram_gb": "abc"})
    assert response.status_code == 422


def test_model_info(trained_model):
    response = _client().get("/model-info")
    assert response.status_code == 200
    assert "r2" in response.json()
