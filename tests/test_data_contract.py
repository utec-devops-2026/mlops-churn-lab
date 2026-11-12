"""Tests del contrato de datos (CRISP-ML(Q) fase 1)."""
import pandas as pd

from common import load_config, load_raw
from validate_data import validate


def _failed(df):
    return {c["check"] for c in validate(df, load_config()) if not c["passed"]}


def test_raw_dataset_meets_contract():
    cfg = load_config()
    assert _failed(load_raw(cfg)) == set()


def test_none_is_a_category_not_a_null():
    df = load_raw(load_config())
    assert df["internet_service"].isna().sum() == 0
    assert "None" in set(df["internet_service"])


def test_unexpected_category_fails():
    df = load_raw(load_config())
    df.loc[0, "contract_type"] = "Weekly"
    assert "contract_type:allowed_values" in _failed(df)


def test_out_of_range_fails():
    df = load_raw(load_config())
    df.loc[0, "monthly_charges"] = -10
    assert "monthly_charges:range" in _failed(df)


def test_missing_column_fails():
    df = load_raw(load_config()).drop(columns=["tenure_months"])
    assert "tenure_months:exists" in _failed(df)