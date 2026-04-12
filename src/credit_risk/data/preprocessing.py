from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

from src.credit_risk.utils.decorators import log_execution_time

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clean_loan_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Clean loan amount column by removing currency symbols and commas."""
    df = df.copy()

    df["loan_amnt"] = (
        df["loan_amnt"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    return df


def convert_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> pd.DataFrame:
    """Convert selected columns to numeric dtype."""
    df = df.copy()

    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def map_target_column(
    df: pd.DataFrame,
    original_target_column: str,
    positive_class: str,
    negative_class: str,
    target_column: str,
) -> pd.DataFrame:
    """Map original target labels to binary target."""
    df = df.copy()

    df[target_column] = df[original_target_column].map(
        {
            positive_class: 1,
            negative_class: 0,
        }
    )
    return df


def map_historical_default(df: pd.DataFrame) -> pd.DataFrame:
    """Map historical_default values from Y/N to 1/0."""
    df = df.copy()

    df["historical_default"] = df["historical_default"].map(
        {
            "Y": 1,
            "N": 0,
        }
    )
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using median for numerics and mode for categoricals."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """Drop selected columns if they exist."""
    df = df.copy()
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_cols)


@log_execution_time
def preprocess_credit_data(
    df: pd.DataFrame,
    model_config_path: str | Path,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline for the credit dataset."""
    config = load_yaml_config(model_config_path)
    df = df.copy()

    numeric_columns = [
        "customer_age",
        "customer_income",
        "employment_duration",
        "cred_hist_length",
        "loan_int_rate",
        "term_years",
    ]

    # 1) Limpieza y conversión base
    df = clean_loan_amount(df)
    df = convert_numeric_columns(df, numeric_columns)

    # 2) Eliminar registros inválidos de edad
    initial_rows = len(df)
    df = df[df["customer_age"] >= 18].copy()
    removed_rows = initial_rows - len(df)
    logger.info("Removed %s rows with invalid age", removed_rows)

    # 3) Variables objetivo/binarias
    df = map_target_column(
        df=df,
        original_target_column=config["data"]["original_target_column"],
        positive_class=config["data"]["positive_class"],
        negative_class=config["data"]["negative_class"],
        target_column=config["data"]["target_column"],
    )

    df = map_historical_default(df)

    # 4) Imputación simple
    df = impute_missing_values(df)

    return df
