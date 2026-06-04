"""Tests for the FastAPI service."""

from fastapi.testclient import TestClient


def _client():
    from app.api import app

    return TestClient(app)


def _valid_payload():
    return {
        "distance_km": 7.9,
        "weather": "Clear",
        "traffic_level": "Medium",
        "time_of_day": "Afternoon",
        "vehicle_type": "Scooter",
        "preparation_time_min": 12,
        "courier_experience_yrs": 2.0,
    }


def test_health_ok(trained_model):
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_eta(trained_model):
    response = _client().post("/predict", json=_valid_payload())
    assert response.status_code == 200
    assert response.json()["eta_minutes"] > 0


def test_predict_rejects_invalid(trained_model):
    response = _client().post("/predict", json={"distance_km": "abc"})
    assert response.status_code == 422


def test_model_info(trained_model):
    response = _client().get("/model-info")
    assert response.status_code == 200
    assert "mae" in response.json()
