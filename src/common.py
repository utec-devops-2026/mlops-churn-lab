"""Utilidades compartidas: rutas, configuración y carga del dataset crudo."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "churn.yaml"
MLFLOW_URI = f"file://{ROOT / 'mlruns'}"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def feature_columns(cfg: dict) -> list:
    feats = cfg["features"]
    return feats["numeric"] + feats["categorical"] + feats["binary"]


def load_raw(cfg: dict) -> pd.DataFrame:
    """Lee el CSV crudo.

    keep_default_na=False: pandas convierte el texto "None" en NaN por defecto.
    En este dataset "None" es una categoría válida de internet_service
    (cliente sin internet), no un dato faltante.
    """
    path = ROOT / cfg["data"]["raw_path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])