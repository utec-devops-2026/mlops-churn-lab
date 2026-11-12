"""Preprocesamiento como parte del modelo.

El ColumnTransformer se entrena junto con el clasificador dentro de un Pipeline.
El artefacto final recibe las columnas crudas del negocio y devuelve la
probabilidad de churn: no hay código de preprocesamiento duplicado en la API.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    feats = cfg["features"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feats["numeric"]),
            # handle_unknown="ignore": una categoría nunca vista no tumba la API
            ("cat", OneHotEncoder(handle_unknown="ignore"), feats["categorical"]),
            ("bin", "passthrough", feats["binary"]),
        ]
    )


def build_pipeline(estimator, cfg: dict) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(cfg)), ("model", estimator)])