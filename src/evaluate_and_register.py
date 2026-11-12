"""CRISP-ML(Q) fase 4 · Evaluación, quality gate y registro.

1. Elige el umbral de decisión que minimiza el costo de negocio (con predicciones
   out-of-fold de train, nunca con test).
2. Mide el modelo en test: métricas globales, costo y métricas por segmento.
3. Aplica el quality gate de configs/churn.yaml.
4. Registra la versión como @challenger y la promueve a @champion solo si pasa
   el gate y no es peor que el champion vigente.
"""
import json
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import DATA_DIR, MLFLOW_URI, REPORTS_DIR, feature_columns, load_config


def expected_cost(y_true, y_pred, cost_fn, cost_fp) -> float:
    """Costo promedio por cliente de las decisiones del modelo."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fn * cost_fn + fp * cost_fp) / len(y_true))


def choose_threshold(pipeline, X, y, cfg) -> float:
    """Umbral de menor costo, calculado con predicciones out-of-fold de train."""
    seed = cfg["project"]["random_state"]
    cv = StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=seed)
    proba = cross_val_predict(clone(pipeline), X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    costs = [expected_cost(y, (proba >= t).astype(int), **cfg["business"]) for t in grid]
    return float(grid[int(np.argmin(costs))])


def classification_metrics(y_true, proba, threshold) -> dict:
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def slice_metrics(test, proba, threshold, cfg) -> dict:
    """Recall por segmento: un modelo bueno en promedio puede fallar en un grupo."""
    target = cfg["data"]["target"]
    y_pred = (proba >= threshold).astype(int)
    slices = {}
    for col in cfg["quality_gate"]["slices"]:
        groups = {}
        for value, idx in test.groupby(col).indices.items():
            y_true = test[target].to_numpy()[idx]
            groups[str(value)] = {
                "n": int(len(idx)),
                "recall": round(float(recall_score(y_true, y_pred[idx], zero_division=0)), 4),
            }
        recalls = [g["recall"] for g in groups.values()]
        slices[col] = {"groups": groups, "recall_gap": round(max(recalls) - min(recalls), 4)}
    return slices


def champion_cost(client, name, X_test, y_test, cfg):
    """Costo del champion vigente en el mismo test set, o None si no hay champion."""
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except Exception:
        return None, None
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    threshold = float(version.tags.get("threshold", 0.5))
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    return version.version, expected_cost(y_test, y_pred, **cfg["business"])


def write_model_card(info: dict, report: dict, cfg: dict):
    gate_rows = "\n".join(
        f"| {c['check']} | {c['value']} | {c['required']} | {'OK' if c['passed'] else 'FALLA'} |"
        for c in report["gate"]["checks"]
    )
    slice_rows = "\n".join(
        f"| {col} | {group} | {vals['n']} | {vals['recall']} |"
        for col, data in report["slices"].items()
        for group, vals in data["groups"].items()
    )
    m = report["test_metrics"]
    card = f"""# Model card · {info['model_name']} v{info['version']}

## Uso previsto
Priorizar clientes para la campaña de retención. La salida es una probabilidad de
churn; la decisión de contactar se toma con el umbral {info['threshold']}.
No usar para negar servicios ni para decisiones sin revisión humana.

## Datos
Dataset `{cfg['data']['raw_path']}`, {report['train_rows']} filas de train y {report['test_rows']} de test.
Features: {', '.join(info['features'])}.

## Modelo
Tipo: {info['model_type']} dentro de un Pipeline de scikit-learn (preprocesamiento incluido).
Run de MLflow: `{info['run_id']}`.

## Métricas en test (umbral {info['threshold']})
| F1 | Precision | Recall | ROC-AUC | Costo por cliente |
|---|---|---|---|---|
| {m['f1']} | {m['precision']} | {m['recall']} | {m['roc_auc']} | {report['cost']['model']} |

Costo sin modelo: contactar a todos {report['cost']['contact_all']}, no contactar a nadie {report['cost']['contact_none']}.

## Quality gate
| Check | Valor | Requerido | Resultado |
|---|---|---|---|
{gate_rows}

## Recall por segmento
| Segmento | Grupo | n | Recall |
|---|---|---|---|
{slice_rows}

## Limitaciones
Dataset pequeño ({report['train_rows'] + report['test_rows']} clientes): las métricas tienen alta varianza.
El umbral depende de los costos de negocio definidos en `configs/churn.yaml`; si cambian, hay que recalcularlo.
"""
    with open(REPORTS_DIR / "model_card.md", "w") as f:
        f.write(card)


def main():
    cfg = load_config()
    name = cfg["project"]["name"]
    target = cfg["data"]["target"]
    features = feature_columns(cfg)
    gate_cfg = cfg["quality_gate"]

    with open(DATA_DIR / "best_model_meta.json") as f:
        meta = json.load(f)
    pipeline = joblib.load(DATA_DIR / "best_model.pkl")
    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
    X_test, y_test = test[features], test[target]

    # 1. Umbral por costo (solo con train)
    threshold = choose_threshold(pipeline, train[features], train[target], cfg)

    # 2. Evaluación en test
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, proba, threshold)
    metrics_050 = classification_metrics(y_test, proba, 0.5)
    costs = {
        "model": round(expected_cost(y_test, (proba >= threshold).astype(int), **cfg["business"]), 2),
        "model_threshold_050": round(expected_cost(y_test, (proba >= 0.5).astype(int), **cfg["business"]), 2),
        "contact_all": round(expected_cost(y_test, np.ones(len(y_test), dtype=int), **cfg["business"]), 2),
        "contact_none": round(expected_cost(y_test, np.zeros(len(y_test), dtype=int), **cfg["business"]), 2),
    }
    slices = slice_metrics(test, proba, threshold, cfg)

    # 3. Quality gate
    baseline_cost = min(costs["contact_all"], costs["contact_none"])
    max_gap = max(s["recall_gap"] for s in slices.values())
    checks = [
        {"check": "f1", "value": metrics["f1"], "required": f">= {gate_cfg['min_f1']}",
         "passed": metrics["f1"] >= gate_cfg["min_f1"]},
        {"check": "roc_auc", "value": metrics["roc_auc"], "required": f">= {gate_cfg['min_roc_auc']}",
         "passed": metrics["roc_auc"] >= gate_cfg["min_roc_auc"]},
        {"check": "recall_gap", "value": max_gap, "required": f"<= {gate_cfg['max_recall_gap']}",
         "passed": max_gap <= gate_cfg["max_recall_gap"]},
    ]
    if gate_cfg["must_beat_baseline"]:
        checks.append({"check": "costo_vs_sin_modelo", "value": costs["model"],
                       "required": f"< {baseline_cost}", "passed": costs["model"] < baseline_cost})
    gate_passed = all(c["passed"] for c in checks)

    # 4. Registro: toda versión evaluada entra como @challenger
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    champion_version, champ_cost = champion_cost(client, name, X_test, y_test, cfg)

    mv = mlflow.register_model(model_uri=f"runs:/{meta['best_run_id']}/model", name=name)
    client.set_registered_model_alias(name, "challenger", mv.version)
    client.set_model_version_tag(name, mv.version, "threshold", str(threshold))
    client.set_model_version_tag(name, mv.version, "test_f1", str(metrics["f1"]))
    client.set_model_version_tag(name, mv.version, "quality_gate", "passed" if gate_passed else "failed")

    beats_champion = champ_cost is None or costs["model"] <= champ_cost
    promoted = gate_passed and beats_champion

    report = {
        "model_type": meta["best_model"],
        "version": str(mv.version),
        "threshold": threshold,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_metrics": metrics,
        "test_metrics_threshold_050": metrics_050,
        "cv_metrics": meta["metrics"],
        "cost": costs,
        "slices": slices,
        "gate": {"passed": gate_passed, "checks": checks},
        "champion": {"previous_version": champion_version, "previous_cost": champ_cost,
                     "beats_champion": beats_champion},
        "promoted": promoted,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Umbral elegido por costo: {threshold}")
    print(f"Test: {metrics}")
    print(f"Costo por cliente: {costs}")
    for c in checks:
        print(f"  {'OK   ' if c['passed'] else 'FALLA'} {c['check']}: {c['value']} (requerido {c['required']})")

    if not promoted:
        reason = "quality gate" if not gate_passed else f"no supera al champion v{champion_version}"
        print(f"Versión {mv.version} queda como @challenger. Motivo: {reason}")
        sys.exit(1)

    client.set_registered_model_alias(name, "champion", mv.version)
    info = {
        "model_name": name,
        "version": str(mv.version),
        "alias": "champion",
        "run_id": meta["best_run_id"],
        "model_type": meta["best_model"],
        "model_uri": f"models:/{name}@champion",
        "threshold": threshold,
        "features": features,
        "metrics": metrics,
    }
    joblib.dump(pipeline, DATA_DIR / "production_model.pkl")
    with open(DATA_DIR / "production_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    write_model_card(info, report, cfg)
    print(f"Versión {mv.version} promovida a @champion")


if __name__ == "__main__":
    main()