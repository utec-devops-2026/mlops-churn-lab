"""CRISP-ML(Q) fase 6 · Monitoreo y mantenimiento.

Tres señales, de la más temprana a la más tardía:
  1. Data drift       ¿cambiaron las features?            PSI + KS / chi-cuadrado
  2. Prediction drift ¿cambió lo que predice el modelo?   PSI de la probabilidad
  3. Performance      ¿sigue acertando?                   F1, recall (necesita etiquetas)

En churn la etiqueta real llega semanas después de la predicción, así que las
señales 1 y 2 son las únicas disponibles en el día a día.

Uso:
  python src/monitor.py --scenario none          # 8 semanas sin drift
  python src/monitor.py --scenario price_hike    # data drift
  python src/monitor.py --scenario competitor    # concept drift
  python src/monitor.py --from-log logs/predictions.jsonl   # tráfico real, sin etiquetas
"""
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, recall_score

from common import DATA_DIR, REPORTS_DIR, feature_columns, load_config

EXIT_RETRAIN = 42
SCENARIOS = ["none", "price_hike", "competitor"]


# ── Métricas de drift ───────────────────────────────────────────────────────

def psi_numeric(reference, current, bins=5) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference, current) -> float:
    categories = sorted(set(reference) | set(current))
    ref = np.array([(reference == c).mean() for c in categories])
    cur = np.array([(current == c).mean() for c in categories])
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def data_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    feats, mon = cfg["features"], cfg["monitoring"]
    report = {}
    for col in feats["numeric"]:
        psi = psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
        p_value = float(stats.ks_2samp(reference[col], current[col]).pvalue)
        report[col] = {"psi": round(psi, 4), "test": "ks", "p_value": round(p_value, 4)}
    for col in feats["categorical"] + feats["binary"]:
        ref, cur = reference[col].astype(str), current[col].astype(str)
        psi = psi_categorical(ref.to_numpy(), cur.to_numpy())
        table = pd.crosstab(
            pd.concat([ref, cur], ignore_index=True),
            np.r_[np.zeros(len(ref)), np.ones(len(cur))],
        )
        p_value = float(stats.chi2_contingency(table)[1])
        report[col] = {"psi": round(psi, 4), "test": "chi2", "p_value": round(p_value, 4)}
    for col, r in report.items():
        r["status"] = ("critical" if r["psi"] >= mon["psi_critical"]
                       else "warning" if r["psi"] >= mon["psi_warning"] else "ok")
    return report


# ── Escenarios simulados ────────────────────────────────────────────────────

def apply_scenario(df: pd.DataFrame, scenario: str, intensity: float, rng) -> pd.DataFrame:
    """intensity va de 0 (sin drift) a 1 (drift máximo)."""
    df = df.copy()
    if scenario == "price_hike" and intensity > 0:
        # Sube la tarifa y se disparan los reclamos: cambian las features (covariate shift)
        df["monthly_charges"] = (df["monthly_charges"] * (1 + 0.5 * intensity)).round(2)
        df["support_tickets_last_6m"] = df["support_tickets_last_6m"] + int(round(4 * intensity))
    if scenario == "competitor" and intensity > 0 and "churn" in df.columns:
        # Un competidor capta a los clientes de contrato largo y, a la vez, un plan de
        # fidelización retiene a los Month-to-Month. Las features son las mismas: lo que
        # cambia es la relación entre features y churn (concept drift)
        monthly = df["contract_type"] == "Month-to-Month"
        flip = rng.random(len(df)) < 0.8 * intensity
        df.loc[~monthly & (df["churn"] == 0) & flip, "churn"] = 1
        df.loc[monthly & (df["churn"] == 1) & flip, "churn"] = 0
    return df


# ── Evaluación de una ventana ───────────────────────────────────────────────

def evaluate_window(model, info, reference, ref_proba, current, cfg, labeled=None) -> dict:
    """current: features observadas en producción. labeled: clientes cuya etiqueta ya llegó."""
    features, mon = feature_columns(cfg), cfg["monitoring"]
    threshold = float(info.get("threshold", 0.5))

    drift = data_drift(reference, current, cfg)
    drifted = [c for c, r in drift.items() if r["status"] == "critical"]
    psi_max = max(r["psi"] for r in drift.values())

    proba = model.predict_proba(current[features])[:, 1]
    prediction_psi = psi_numeric(ref_proba, proba)
    pred = (proba >= threshold).astype(int)

    performance = None
    f1_drop = 0.0
    if labeled is not None:
        y_pred = (model.predict_proba(labeled[features])[:, 1] >= threshold).astype(int)
        f1 = float(f1_score(labeled["churn"], y_pred, zero_division=0))
        f1_drop = float(info["metrics"]["f1"]) - f1
        performance = {
            "n_labeled": len(labeled),
            "f1": round(f1, 4),
            "recall": round(float(recall_score(labeled["churn"], y_pred, zero_division=0)), 4),
            "f1_drop": round(f1_drop, 4),
        }

    reasons = []
    if len(drifted) >= mon["min_drifted_features"]:
        reasons.append(f"data drift en {drifted}")
    if prediction_psi >= mon["psi_critical"]:
        reasons.append(f"prediction drift (PSI {prediction_psi:.2f})")
    if performance and f1_drop >= mon["max_f1_drop"]:
        reasons.append(f"F1 cayó {f1_drop:.2f}")

    if reasons:
        decision = "REENTRENAR"
    elif psi_max >= mon["psi_warning"] or prediction_psi >= mon["psi_warning"]:
        decision = "ALERTA"
    else:
        decision = "MONITOREAR"

    return {
        "n": len(current),
        "data_drift": drift,
        "drifted_features": drifted,
        "psi_max": round(psi_max, 4),
        "prediction_drift": {"psi": round(prediction_psi, 4),
                             "predicted_churn_rate": round(float(pred.mean()), 4)},
        "performance": performance,
        "decision": decision,
        "reasons": reasons,
    }


def push_metrics(gateway: str, week: int, window: dict):
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    values = {
        "model_simulated_week": week,
        "model_psi_max": window["psi_max"],
        "model_drifted_features": len(window["drifted_features"]),
        "model_prediction_psi": window["prediction_drift"]["psi"],
        "model_predicted_churn_rate": window["prediction_drift"]["predicted_churn_rate"],
        "model_retrain_needed": int(window["decision"] == "REENTRENAR"),
    }
    if window["performance"]:
        values["model_f1"] = window["performance"]["f1"]
        values["model_recall"] = window["performance"]["recall"]
    for name, value in values.items():
        Gauge(name, name, registry=registry).set(value)
    for col, r in window["data_drift"].items():
        Gauge(f"model_psi_{col}", f"PSI de {col}", registry=registry).set(r["psi"])
    push_to_gateway(gateway, job="churn_monitor", registry=registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="none", choices=SCENARIOS)
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--drift-start", type=int, default=5, help="semana en que empieza el drift")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--from-log", default=None, help="JSONL de predicciones de la API (sin etiquetas)")
    parser.add_argument("--push", default=None, help="URL del Pushgateway, ej. localhost:9091")
    parser.add_argument("--interval", type=float, default=0, help="segundos entre semanas al hacer push")
    parser.add_argument("--fail-on-retrain", action="store_true", help=f"exit {EXIT_RETRAIN} si hay que reentrenar")
    parser.add_argument("--output", default=str(REPORTS_DIR / "monitoring_report.json"))
    args = parser.parse_args()

    cfg = load_config()
    features = feature_columns(cfg)
    model = joblib.load(DATA_DIR / "production_model.pkl")
    with open(DATA_DIR / "production_model_info.json") as f:
        info = json.load(f)

    reference = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    ref_proba = model.predict_proba(reference[features])[:, 1]
    windows = []

    if args.from_log:
        current = pd.read_json(args.from_log, lines=True, dtype={"internet_service": str})[features]
        window = evaluate_window(model, info, reference, ref_proba, current, cfg)
        windows.append({"window": "api_log", **window})
        if args.push:
            push_metrics(args.push, 0, window)
        source = args.from_log
    else:
        # Tráfico simulado: clientes de toda la cartera (solo features, sin etiqueta).
        # Etiquetas: solo llegan para el set de test, que el modelo nunca vio.
        test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
        production = pd.concat([reference, test], ignore_index=True)[features]
        for week in range(1, args.weeks + 1):
            rng = np.random.default_rng(cfg["project"]["random_state"] + week)
            steps = max(args.weeks - args.drift_start + 1, 1)
            intensity = max(0, week - args.drift_start + 1) / steps
            sample = production.sample(args.sample_size, replace=True, random_state=week)
            current = apply_scenario(sample, args.scenario, intensity, rng)
            labeled = apply_scenario(test, args.scenario, intensity, rng)
            window = evaluate_window(model, info, reference, ref_proba, current, cfg, labeled)
            windows.append({"window": f"week_{week}", "intensity": round(intensity, 2), **window})
            if args.push:
                push_metrics(args.push, week, window)
                time.sleep(args.interval)
        source = f"scenario:{args.scenario}"

    final = windows[-1]
    report = {
        "source": source,
        "model_version": info.get("version"),
        "reference_f1": info["metrics"]["f1"],
        "thresholds": cfg["monitoring"],
        "decision": final["decision"],
        "reasons": final["reasons"],
        "windows": windows,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'ventana':<10} {'psi_max':>8} {'pred_psi':>9} {'f1':>7}  decisión")
    for w in windows:
        f1 = f"{w['performance']['f1']:.3f}" if w["performance"] else "  s/e"
        print(f"{w['window']:<10} {w['psi_max']:>8.3f} {w['prediction_drift']['psi']:>9.3f} {f1:>7}  {w['decision']}")
    print(f"\nDecisión final: {final['decision']} {final['reasons'] or ''}")
    print(f"Reporte: {args.output}")

    if args.fail_on_retrain and final["decision"] == "REENTRENAR":
        sys.exit(EXIT_RETRAIN)


if __name__ == "__main__":
    main()
