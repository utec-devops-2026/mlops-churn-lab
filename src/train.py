"""CRISP-ML(Q) fase 3 · Modelado con MLflow tracking.

Entrena un baseline y tres candidatos. Cada candidato es un Pipeline
(preprocesamiento + clasificador) evaluado con validación cruzada estratificada
sobre train. El mejor se elige por F1 de validación cruzada: el set de test
no participa en la selección, queda reservado para la fase de evaluación.
"""
import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from common import DATA_DIR, MLFLOW_URI, feature_columns, load_config
from features import build_pipeline

CANDIDATES = ["logistic", "random_forest", "xgboost"]
SCORING = {"f1": "f1", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall"}


def get_estimator(name: str, seed: int):
    estimators = {
        # Baseline: predice siempre la clase positiva. Si un modelo no le gana, no aporta.
        "baseline": (DummyClassifier(strategy="constant", constant=1), {"strategy": "constant=1"}),
        "logistic": (
            LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed),
            {"C": 1.0, "max_iter": 500, "class_weight": "balanced"},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced",
                                   random_state=seed, n_jobs=-1),
            {"n_estimators": 200, "max_depth": 5, "class_weight": "balanced"},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed,
                          eval_metric="logloss", verbosity=0),
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        ),
    }
    return estimators[name]


def train_one(name, X, y, cfg):
    seed = cfg["project"]["random_state"]
    estimator, params = get_estimator(name, seed)
    pipeline = build_pipeline(estimator, cfg)
    cv = StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=seed)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_type": name, "dataset": "telco_churn", "crisp_mlq_phase": "modeling"})
        mlflow.log_params({**params, "cv_folds": cv.get_n_splits(), "train_size": len(X)})

        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        metrics = {}
        for metric in SCORING:
            fold_scores = scores[f"test_{metric}"]
            metrics[f"cv_{metric}_mean"] = round(float(fold_scores.mean()), 4)
            metrics[f"cv_{metric}_std"] = round(float(fold_scores.std()), 4)
        mlflow.log_metrics(metrics)

        # Modelo final: se reentrena con todo train
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, "model", input_example=X.head(3))

    return {"model_type": name, "run_id": run.info.run_id, "metrics": metrics, "pipeline": pipeline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=CANDIDATES + ["all"])
    args = parser.parse_args()

    cfg = load_config()
    target = cfg["data"]["target"]
    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    X, y = train[feature_columns(cfg)], train[target]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["project"]["experiment"])

    baseline = train_one("baseline", X, y, cfg)
    print(f"baseline       F1 CV = {baseline['metrics']['cv_f1_mean']:.4f}")

    results = []
    for name in CANDIDATES if args.model == "all" else [args.model]:
        result = train_one(name, X, y, cfg)
        results.append(result)
        m = result["metrics"]
        print(f"{name:<14} F1 CV = {m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f} | ROC-AUC CV = {m['cv_roc_auc_mean']:.4f}")

    selection = cfg["training"]["selection_metric"]
    best = max(results, key=lambda r: r["metrics"][selection])
    print(f"\nMejor modelo por {selection}: {best['model_type']}")

    meta = {
        "best_model": best["model_type"],
        "best_run_id": best["run_id"],
        "selection_metric": selection,
        "metrics": best["metrics"],
        "baseline_metrics": baseline["metrics"],
        "all_results": [{"model": r["model_type"], **r["metrics"]} for r in results],
    }
    with open(DATA_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(best["pipeline"], DATA_DIR / "best_model.pkl")
    print(f"Guardado data/best_model.pkl | run_id: {best['run_id']}")


if __name__ == "__main__":
    main()