

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
