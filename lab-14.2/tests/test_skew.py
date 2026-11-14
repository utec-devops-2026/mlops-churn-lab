"""Test de training-serving skew (CRISP-ML(Q) fase 5).

Compara, cliente por cliente, la probabilidad que da el Pipeline offline con la
que devuelve la API en ejecución. Si difieren, la API no está sirviendo el mismo
modelo que se evaluó. Necesita la API levantada:

    API_URL=http://localhost:8000 pytest tests/test_skew.py -v
"""
import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
import requests

API_URL = os.getenv("API_URL")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(not API_URL, reason="define API_URL para correr el test de skew")


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False).drop(columns=["churn"]).head(30)


def test_api_matches_offline_pipeline(sample):
    pipeline = joblib.load(DATA_DIR / "production_model.pkl")
    offline = pipeline.predict_proba(sample)[:, 1]

    payload = {"customers": sample.to_dict(orient="records")}
    res = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)
    assert res.status_code == 200
    online = [p["churn_probability"] for p in res.json()["predictions"]]

    diffs = [abs(a - b) for a, b in zip(offline, online)]
    assert max(diffs) < 1e-3, f"skew detectado: diferencia máxima {max(diffs):.4f}"


def test_contract_type_changes_the_prediction(sample):
    base = sample.iloc[0].to_dict()
    probs = []
    for contract in ["Month-to-Month", "Two Year"]:
        res = requests.post(f"{API_URL}/predict", json={**base, "contract_type": contract}, timeout=30)
        assert res.status_code == 200
        probs.append(res.json()["churn_probability"])
    assert probs[0] != probs[1], "la API ignora contract_type"
