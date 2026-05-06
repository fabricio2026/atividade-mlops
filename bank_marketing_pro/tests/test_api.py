import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 200,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": -1.8,
    "cons_price_idx": 92.893,
    "cons_conf_idx": -46.2,
    "euribor3m": 1.313,
    "nr_employed": 5099.1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "version" in data


def test_predict_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    if response.status_code == 503:
        pytest.skip("Model not loaded — run scripts/train_model.py first")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["label"] in ("yes", "no")


def test_predict_invalid_age_returns_422(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "age": 5})
    assert response.status_code == 422


def test_predict_invalid_job_returns_422(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "job": "astronaut"})
    assert response.status_code == 422


def test_predict_extra_field_returns_422(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "extra_field": "x"})
    assert response.status_code == 422


def test_predict_model_not_loaded_returns_503(client):
    import app.model as model_module
    saved = model_module._model
    model_module._model = None
    try:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    finally:
        model_module._model = saved


def test_info_returns_503_when_model_not_loaded(client):
    import app.model as model_module
    saved = model_module._model
    model_module._model = None
    try:
        response = client.get("/info")
        assert response.status_code == 503
    finally:
        model_module._model = saved
