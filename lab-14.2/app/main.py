"""CRISP-ML(Q) fase 5 · API de inferencia.

La API no preprocesa: arma un DataFrame con las columnas crudas y se lo pasa al
Pipeline entrenado. El umbral de decisión viene de production_model_info.json.
Cada predicción se registra en un log JSONL (insumo del monitoreo) y se expone
como métrica en /metrics para Prometheus.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "data"))
PREDICTION_LOG = os.getenv("PREDICTION_LOG", str(ROOT / "logs" / "predictions.jsonl"))

model = joblib.load(MODEL_DIR / "production_model.pkl")
with open(MODEL_DIR / "production_model_info.json") as f:
    model_info = json.load(f)

FEATURES = model_info.get("features", [
    "tenure_months", "monthly_charges", "support_tickets_last_6m",
    "gender", "contract_type", "internet_service", "senior_citizen",
])
THRESHOLD = float(model_info.get("threshold", 0.5))
HIGH_RISK = max(THRESHOLD, 0.6)

app = FastAPI(
    title="Churn Predictor API",
    description="API de predicción de abandono de clientes para Fintech",
    version=str(model_info.get("version", "1")),
)

# Métricas HTTP (requests, latencia, status) en /metrics
Instrumentator().instrument(app).expose(app)

# Métricas del modelo: qué está prediciendo en producción
PREDICTIONS = Counter("churn_predictions_total", "Predicciones servidas", ["risk_level"])
PROBABILITY = Histogram(
    "churn_probability", "Distribución de la probabilidad de churn predicha",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

_log_lock = threading.Lock()


class CustomerFeatures(BaseModel):
    """Contrato de entrada: mismos rangos y categorías que el contrato de datos."""
    gender: Literal["Male", "Female"] = Field(..., examples=["Male"])
    senior_citizen: int = Field(..., ge=0, le=1, examples=[0])
    tenure_months: int = Field(..., ge=0, le=120, examples=[12], description="Meses como cliente")
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"] = Field(..., examples=["Month-to-Month"])
    internet_service: Optional[Literal["Fiber", "DSL", "None"]] = Field(None, examples=["Fiber"])
    monthly_charges: float = Field(..., ge=0, le=500, examples=[85.5])
    support_tickets_last_6m: int = Field(..., ge=0, le=50, examples=[3])


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_label: str
    churn_probability: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]
    customer_ids: Optional[List[str]] = None


def to_frame(customers: List[dict]) -> pd.DataFrame:
    """Columnas crudas en el orden de entrenamiento. Sin escalado ni encoding."""
    df = pd.DataFrame(customers)
    df["internet_service"] = df["internet_service"].fillna("None")
    return df[FEATURES]


def get_risk_level(probability: float) -> str:
    if probability < THRESHOLD:
        return "BAJO"
    if probability < HIGH_RISK:
        return "MEDIO"
    return "ALTO"


def get_recommendation(data: dict, risk: str) -> str:
    if risk == "BAJO":
        return "Sin acción requerida"
    if risk == "MEDIO":
        return "Incluir en campaña de retención"
    if data.get("contract_type") == "Month-to-Month" and data.get("monthly_charges", 0) > 70:
        return "Ofrecer contrato anual con descuento"
    if data.get("support_tickets_last_6m", 0) >= 5:
        return "Soporte proactivo urgente"
    if data.get("tenure_months", 99) <= 12:
        return "Programa de fidelización temprana"
    return "Contactar con oferta de retención personalizada"


def log_predictions(rows: List[dict]):
    """Guarda features + predicción. Sin esto no hay cómo medir drift en producción."""
    if not PREDICTION_LOG:
        return
    path = Path(PREDICTION_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def score(customers: List[dict], customer_ids: List[Optional[str]]) -> List[dict]:
    probabilities = model.predict_proba(to_frame(customers))[:, 1]
    now = datetime.now(timezone.utc).isoformat()
    results, log_rows = [], []
    for data, cid, probability in zip(customers, customer_ids, probabilities):
        probability = round(float(probability), 4)
        prediction = int(probability >= THRESHOLD)
        risk = get_risk_level(probability)
        PREDICTIONS.labels(risk_level=risk).inc()
        PROBABILITY.observe(probability)
        results.append({
            "customer_id": cid,
            "churn_prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": probability,
            "risk_level": risk,
            "recommendation": get_recommendation(data, risk),
            "model_version": str(model_info.get("version", "1")),
        })
        log_rows.append({
            "ts": now,
            "model_version": str(model_info.get("version", "1")),
            **{k: data[k] if data[k] is not None else "None" for k in FEATURES},
            "churn_probability": probability,
            "churn_prediction": prediction,
        })
    log_predictions(log_rows)
    return results


@app.get("/")
def root():
    return {
        "service": "Churn Predictor API",
        "version": model_info.get("version", "1"),
        "model_type": model_info.get("model_type", "unknown"),
        "threshold": THRESHOLD,
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info_endpoint():
    return model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, customer_id: Optional[str] = None):
    return score([customer.model_dump()], [customer_id])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    customers = [c.model_dump() for c in request.customers]
    ids = request.customer_ids or [f"CUST-{i:04d}" for i in range(len(customers))]
    results = score(customers, ids)
    return {
        "total": len(results),
        "churn_count": sum(r["churn_prediction"] for r in results),
        "predictions": results,
    }
