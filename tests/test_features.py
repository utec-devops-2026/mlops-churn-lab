"""Tests del Pipeline de preprocesamiento (CRISP-ML(Q) fase 2)."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import feature_columns, load_config, load_raw
from features import build_pipeline


def _fitted_pipeline():
    cfg = load_config()
    df = load_raw(cfg)
    X, y = df[feature_columns(cfg)], (df[cfg["data"]["target"]] == "Yes").astype(int)
    return build_pipeline(LogisticRegression(max_iter=500), cfg).fit(X, y), X


def test_pipeline_accepts_raw_columns():
    pipeline, X = _fitted_pipeline()
    proba = pipeline.predict_proba(X.head(5))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_contract_type_changes_the_prediction():
    """El bug original: la API ignoraba contract_type. Aquí debe influir."""
    pipeline, X = _fitted_pipeline()
    row = X.head(1).copy()
    row["contract_type"] = "Month-to-Month"
    monthly = pipeline.predict_proba(row)[0, 1]
    row["contract_type"] = "Two Year"
    two_year = pipeline.predict_proba(row)[0, 1]
    assert monthly != two_year


def test_unknown_category_does_not_crash():
    pipeline, X = _fitted_pipeline()
    row = X.head(1).copy()
    row["internet_service"] = "Satellite"
    assert 0 <= pipeline.predict_proba(row)[0, 1] <= 1