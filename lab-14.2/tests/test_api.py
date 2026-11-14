"""Tests de integración para los endpoints de la API."""
import pytest
from tests.conftest import SAMPLE_CUSTOMER, MOCK_MODEL_INFO


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_health_model_loaded(client):
    res = client.get("/health")
    assert res.json()["model_loaded"] is True


# ── /model/info ───────────────────────────────────────────────────────────────

def test_model_info_returns_200(client):
    res = client.get("/model/info")
    assert res.status_code == 200


def test_model_info_has_required_fields(client):
    res = client.get("/model/info")
    data = res.json()
    assert "model_name" in data or "version" in data


# ── POST /predict ─────────────────────────────────────────────────────────────

def test_predict_status_200(client):
    res = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert res.status_code == 200


def test_predict_response_fields(client):
    res = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = res.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "churn_label" in data
    assert "risk_level" in data
    assert "recommendation" in data


def test_predict_binary_output(client):
    res = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert res.json()["churn_prediction"] in (0, 1)


def test_predict_probability_range(client):
    res = client.post("/predict", json=SAMPLE_CUSTOMER)
    prob = res.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_high_risk_label(client):
    # mock devuelve probabilidad 0.85 → riesgo ALTO
    res = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert res.json()["risk_level"] == "ALTO"


def test_predict_missing_field_returns_422(client):
    bad_payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure_months"}
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422


# ── POST /predict/batch ───────────────────────────────────────────────────────

def test_batch_predict_status_200(client):
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200


def test_batch_predict_total_count(client):
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    res = client.post("/predict/batch", json=payload)
    assert res.json()["total"] == 2


def test_batch_predict_all_predictions_present(client):
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    res = client.post("/predict/batch", json=payload)
    assert len(res.json()["predictions"]) == 2


def test_batch_predict_churn_count_field(client):
    payload = {"customers": [SAMPLE_CUSTOMER]}
    res = client.post("/predict/batch", json=payload)
    assert "churn_count" in res.json()


def test_batch_predict_with_customer_ids(client):
    payload = {
        "customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER],
        "customer_ids": ["CUST-001", "CUST-002"],
    }
    res = client.post("/predict/batch", json=payload)
    ids = [p["customer_id"] for p in res.json()["predictions"]]
    assert ids == ["CUST-001", "CUST-002"]


# ── Lógica de negocio: get_risk_level ─────────────────────────────────────────

def test_risk_level_bajo(client):
    import app.main as m
    assert m.get_risk_level(0.1) == "BAJO"
    assert m.get_risk_level(0.29) == "BAJO"


def test_risk_level_medio(client):
    import app.main as m
    assert m.get_risk_level(0.3) == "MEDIO"
    assert m.get_risk_level(0.59) == "MEDIO"


def test_risk_level_alto(client):
    import app.main as m
    assert m.get_risk_level(0.6) == "ALTO"
    assert m.get_risk_level(0.99) == "ALTO"


# ── Contrato de entrada ───────────────────────────────────────────────────────

def test_predict_invalid_category_returns_422(client):
    res = client.post("/predict", json={**SAMPLE_CUSTOMER, "contract_type": "Weekly"})
    assert res.status_code == 422


def test_predict_out_of_range_returns_422(client):
    res = client.post("/predict", json={**SAMPLE_CUSTOMER, "tenure_months": -5})
    assert res.status_code == 422


# ── Observabilidad ────────────────────────────────────────────────────────────

def test_metrics_exposes_model_metrics(client):
    client.post("/predict", json=SAMPLE_CUSTOMER)
    body = client.get("/metrics").text
    assert "churn_predictions_total" in body
    assert "churn_probability_bucket" in body


def test_prediction_is_logged(client):
    import json
    import app.main as m
    client.post("/predict", json=SAMPLE_CUSTOMER)
    with open(m.PREDICTION_LOG) as f:
        last = json.loads(f.readlines()[-1])
    assert last["contract_type"] == SAMPLE_CUSTOMER["contract_type"]
    assert 0.0 <= last["churn_probability"] <= 1.0
