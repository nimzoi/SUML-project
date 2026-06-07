"""Tests for the FastAPI service."""

from fastapi.testclient import TestClient


def _client():
    """Build a TestClient bound to the FastAPI app (imported lazily)."""
    from app.api import app

    return TestClient(app)


def _valid_payload():
    """Return a well-formed /predict request body."""
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
    """/health returns 200 with status 'ok' and the model reported as loaded."""
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_price(trained_model):
    """/predict returns 200 and a positive price for a valid payload."""
    response = _client().post("/predict", json=_valid_payload())
    assert response.status_code == 200
    assert response.json()["price"] > 0


def test_predict_rejects_invalid(trained_model):
    """/predict returns 422 when the payload fails schema validation."""
    response = _client().post("/predict", json={"ram_gb": "abc"})
    assert response.status_code == 422


def test_model_info(trained_model):
    """/model-info returns 200 and includes the R2 metric."""
    response = _client().get("/model-info")
    assert response.status_code == 200
    assert "r2" in response.json()


def test_openapi_exposes_validation_and_retraining():
    """Swagger/OpenAPI includes the operational validation and retraining endpoints."""
    response = _client().get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/data-schema" in paths
    assert "/data-drift" in paths
    assert "/explain" in paths
    assert "/validate-data" in paths
    assert "/retrain" in paths
    assert "/retrain/{job_id}" in paths


def test_validate_data_endpoint_returns_report():
    """/validate-data validates the configured training data and returns a report."""
    response = _client().post("/validate-data")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["n_rows"] >= body["min_rows"]


def test_data_schema_endpoint_returns_contract():
    """/data-schema documents raw columns, engineered features and validation gates."""
    response = _client().get("/data-schema")
    assert response.status_code == 200
    body = response.json()
    assert "Memory" in body["raw_required_columns"]
    assert "Ram" in body["numeric_features"]
    assert body["validation_gates"]["min_r2"] >= 0


def test_predict_batch_returns_prices(trained_model):
    """/predict-batch returns one price per validated request item."""
    response = _client().post(
        "/predict-batch", json={"items": [_valid_payload(), _valid_payload()]}
    )
    assert response.status_code == 200
    prices = response.json()["prices"]
    assert len(prices) == 2
    assert all(price > 0 for price in prices)


def test_explain_endpoint_returns_prediction_insights(trained_model):
    """/explain returns price, band, contributions and RAM sensitivity."""
    response = _client().post("/explain", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["price"] > 0
    assert body["band"]["high"] >= body["band"]["low"]
    assert body["sensitivity_field"] == "ram_gb"
    assert body["sensitivity"]
    assert "contributions" in body


def test_data_drift_endpoint_returns_report():
    """/data-drift returns a profile comparison report for the configured dataset."""
    response = _client().get("/data-drift")
    assert response.status_code == 200
    body = response.json()
    assert body["current_rows"] > 0
    assert body["reference_rows"] > 0
    assert body["features"]


def test_retrain_endpoint_tracks_job(monkeypatch):
    """/retrain queues a job and /retrain/{job_id} exposes the final status."""
    import app.api as api_module
    from model.schemas import DataValidationReport, ModelValidationReport, RetrainingResult

    api_module._jobs.clear()  # pylint: disable=protected-access

    def _fake_retrain(config, time_budget_s=None, min_r2=None, max_mae=None):
        data_report = DataValidationReport(
            ok=True,
            source="real",
            n_rows=1300,
            min_rows=config.validation.min_rows,
            detail="ok",
        )
        model_report = ModelValidationReport(
            ok=True,
            min_r2=min_r2 if min_r2 is not None else config.validation.min_r2,
            max_mae=max_mae if max_mae is not None else config.validation.max_mae,
            checks={"r2_above_minimum": True},
            detail="ok",
        )
        return RetrainingResult(
            status="succeeded",
            promoted=True,
            detail=f"done in {time_budget_s}s",
            data_validation=data_report,
            model_validation=model_report,
        )

    monkeypatch.setattr(api_module, "retrain_with_validation", _fake_retrain)
    client = TestClient(api_module.app)
    response = client.post("/retrain", json={"time_budget_s": 1, "min_r2": 0.1})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_response = client.get(f"/retrain/{job_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["promoted"] is True


def test_retrain_endpoint_requires_key_when_configured(monkeypatch):
    """/retrain rejects requests without X-API-Key when RETRAIN_API_KEY is configured."""
    import app.api as api_module

    api_module._jobs.clear()  # pylint: disable=protected-access
    monkeypatch.setattr(api_module, "_retrain_api_key", "secret")
    response = TestClient(api_module.app).post("/retrain", json={})
    assert response.status_code == 401
