"""CRISP-ML(Q) fase 1 · Quality gate de datos.

Valida el CSV crudo contra el contrato de configs/churn.yaml y escribe
reports/data_validation.json. Termina con exit 1 si el contrato no se cumple,
así el pipeline se detiene antes de entrenar con datos malos.
"""
import argparse
import json
import sys

import pandas as pd

from common import REPORTS_DIR, load_config, load_raw


def validate(df: pd.DataFrame, cfg: dict) -> list:
    """Devuelve la lista de checks: [{check, passed, detail}]."""
    data_cfg = cfg["data"]
    checks = []

    def add(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("min_rows", len(df) >= data_cfg["min_rows"], f"{len(df)} filas (mínimo {data_cfg['min_rows']})")

    id_col = data_cfg["id_column"]
    if id_col in df.columns:
        dups = int(df[id_col].duplicated().sum())
        add("unique_id", dups == 0, f"{dups} ids duplicados")

    for col, rule in data_cfg["columns"].items():
        if col not in df.columns:
            add(f"{col}:exists", False, "columna ausente")
            continue

        nulls = int(df[col].isna().sum())
        add(f"{col}:no_nulls", nulls == 0, f"{nulls} nulos")

        if rule["type"] == "category":
            allowed = {str(v) for v in rule["values"]}
            unexpected = sorted(set(df[col].dropna().astype(str)) - allowed)
            add(f"{col}:allowed_values", not unexpected, f"valores inesperados: {unexpected}")
        else:
            values = pd.to_numeric(df[col], errors="coerce")
            not_numeric = int(values.isna().sum()) - nulls
            add(f"{col}:numeric", not_numeric == 0, f"{not_numeric} valores no numéricos")
            out = int(((values < rule["min"]) | (values > rule["max"])).sum())
            add(f"{col}:range", out == 0, f"{out} fuera de [{rule['min']}, {rule['max']}]")

    target = data_cfg["target"]
    if target in df.columns:
        rate = float((df[target] == "Yes").mean())
        lo, hi = data_cfg["target_rate"]["min"], data_cfg["target_rate"]["max"]
        add("target_rate", lo <= rate <= hi, f"churn rate {rate:.3f} (esperado entre {lo} y {hi})")

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="CSV a validar (por defecto, el de configs/churn.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    if args.data:
        cfg["data"]["raw_path"] = args.data
    df = load_raw(cfg)
    checks = validate(df, cfg)
    failed = [c for c in checks if not c["passed"]]

    report = {
        "dataset": cfg["data"]["raw_path"],
        "rows": len(df),
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "passed": not failed,
        "checks": checks,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "data_validation.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Contrato de datos: {len(checks) - len(failed)}/{len(checks)} checks OK")
    for c in failed:
        print(f"  FALLA {c['check']}: {c['detail']}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()