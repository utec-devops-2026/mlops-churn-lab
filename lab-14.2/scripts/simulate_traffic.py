"""Envía tráfico a la API simulando un escenario de producción.

    python scripts/simulate_traffic.py --n 300
    python scripts/simulate_traffic.py --n 300 --scenario price_hike
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import feature_columns, load_config  # noqa: E402
from monitor import apply_scenario  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--scenario", default="none", choices=["none", "price_hike"])
    parser.add_argument("--intensity", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config()
    features = feature_columns(cfg)
    customers = pd.concat([
        pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False),
        pd.read_csv(ROOT / "data" / "test.csv", keep_default_na=False),
    ])[features].sample(args.n, replace=True, random_state=7)
    customers = apply_scenario(customers, args.scenario, args.intensity, np.random.default_rng(7))
    # El contrato de la API rechaza valores fuera de rango: se respeta el tope
    customers["monthly_charges"] = customers["monthly_charges"].clip(upper=500)
    customers["support_tickets_last_6m"] = customers["support_tickets_last_6m"].clip(upper=50)

    counts = {}
    for row in customers.to_dict(orient="records"):
        res = requests.post(f"{args.api}/predict", json=row, timeout=10)
        key = res.json()["risk_level"] if res.status_code == 200 else f"HTTP {res.status_code}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{args.n} requests ({args.scenario}): {counts}")


if __name__ == "__main__":
    main()
