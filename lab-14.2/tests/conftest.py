"""
Fixtures compartidas para todos los tests.
La API se importa con el modelo mockeado, así los tests unitarios corren
sin necesitar artefactos entrenados.
"""
import builtins
import io
import json
import os
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Datos de prueba ───────────────────────────────────────────────────────────

MOCK_FEATURES = [
    "tenure_months", "monthly_charges", "support_tickets_last_6m",
    "gender", "contract_type", "internet_service", "senior_citizen",
]

MOCK_MODEL_INFO = {
    "model_name": "churn-predictor",
    "version": "2",
    "alias": "champion",
    "model_type": "random_forest",
    "threshold": 0.3,
    "features": MOCK_FEATURES,
    "metrics": {"f1": 0.70, "roc_auc": 0.80},
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
def client(tmp_path_factory):
    """TestClient con el Pipeline mockeado (probabilidad fija 0.85)."""
    _model = MagicMock()
    _model.predict_proba.side_effect = lambda X: np.tile([0.15, 0.85], (len(X), 1))

    _orig_open = builtins.open

    def _selective_open(path, *args, **kwargs):
        if "production_model_info" in str(path):
            return io.StringIO(json.dumps(MOCK_MODEL_INFO))
        return _orig_open(path, *args, **kwargs)

    os.environ["PREDICTION_LOG"] = str(tmp_path_factory.mktemp("logs") / "predictions.jsonl")

    with patch("joblib.load", return_value=_model), \
         patch("builtins.open", side_effect=_selective_open):

        sys.modules.pop("app.main", None)

        import app.main as app_module
        from fastapi.testclient import TestClient

    yield TestClient(app_module.app)
