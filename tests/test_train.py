"""Tests de configuración de los candidatos de modelado."""
from sklearn.ensemble import GradientBoostingClassifier

from common import feature_columns, load_config, load_raw
from features import build_pipeline
from train import CANDIDATES, get_estimator


def test_gradient_boosting_candidate_runs_with_logged_parameters():
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    estimator, params = get_estimator("gradient_boosting", seed)

    assert "gradient_boosting" in CANDIDATES
    assert isinstance(estimator, GradientBoostingClassifier)
    assert params == {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": seed,
    }

    data = load_raw(cfg)
    X = data[feature_columns(cfg)]
    y = (data[cfg["data"]["target"]] == "Yes").astype(int)
    pipeline = build_pipeline(estimator, cfg).fit(X, y)

    assert pipeline.predict_proba(X.head(5)).shape == (5, 2)
