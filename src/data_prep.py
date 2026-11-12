"""CRISP-ML(Q) fase 2 · Preparación de datos.

Solo limpia y parte el dataset. El escalado y el encoding NO se hacen aquí:
viven dentro del Pipeline de scikit-learn (src/features.py) para que el modelo
aplique en producción exactamente las mismas transformaciones que en entrenamiento.
"""
import json

from sklearn.model_selection import train_test_split

from common import DATA_DIR, feature_columns, load_config, load_raw


def clean(df, cfg):
    target = cfg["data"]["target"]
    df = df.drop(columns=[cfg["data"]["id_column"]], errors="ignore")
    df[target] = (df[target] == "Yes").astype(int)
    return df[feature_columns(cfg) + [target]]


def main():
    cfg = load_config()
    target = cfg["data"]["target"]
    df = clean(load_raw(cfg), cfg)

    train, test = train_test_split(
        df,
        test_size=cfg["project"]["test_size"],
        random_state=cfg["project"]["random_state"],
        stratify=df[target],
    )

    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_churn_rate": round(float(train[target].mean()), 4),
        "test_churn_rate": round(float(test[target].mean()), 4),
        "features": feature_columns(cfg),
    }
    with open(DATA_DIR / "data_prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()