"""
Fixtures compartidas para todos los tests.
Mocks todos los artefactos ML antes de importar app/main.py,
así los tests corren sin necesitar archivos entrenados.
"""
import builtins
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Datos de prueba ───────────────────────────────────────────────────────────

MOCK_FEATURES = [
    "gender", "SeniorCitizen", "tenure", "MonthlyCharges", "support_tickets",
    "Contract_One Year", "Contract_Two Year",
    "InternetService_Fiber", "InternetService_No",
]

MOCK_MODEL_INFO = {
    "model_name": "churn-predictor",
    "version": "2",
    "stage": "Production",
    "model_type": "LogisticRegression",
    "metrics": {"accuracy": 0.733, "f1": 0.692, "roc_auc": 0.828},
}

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "senior_citizen": 0,
    "tenure_months": 12,
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber",
    "monthly_charges": 70.35,
    "support_tickets_last_6m": 3,
}


# ── Fixture principal ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """TestClient con todos los artefactos ML mockeados (sin archivos reales)."""
    _model = MagicMock()
    _model.predict.return_value = np.array([1])
    _model.predict_proba.return_value = np.array([[0.15, 0.85]])

    _scaler = MagicMock()
    _scaler.transform.side_effect = lambda x: x.values

    _features_df = pd.DataFrame(MOCK_FEATURES, columns=[0])

    _orig_open = builtins.open

    def _selective_open(path, *args, **kwargs):
        if "production_model_info" in str(path):
            return io.StringIO(json.dumps(MOCK_MODEL_INFO))
        return _orig_open(path, *args, **kwargs)

    with patch("joblib.load", side_effect=lambda p: _model if "model" in str(p) else _scaler), \
         patch("pandas.read_csv", return_value=_features_df), \
         patch("builtins.open", side_effect=_selective_open):

        sys.modules.pop("app.main", None)

        import app.main as app_module
        from fastapi.testclient import TestClient

    # Los mocks solo hacen falta mientras se importa la API
    yield TestClient(app_module.app)
