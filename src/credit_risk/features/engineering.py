from __future__ import annotations

import numpy as np
import pandas as pd

from src.credit_risk.utils.decorators import log_execution_time


def add_debt_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create debt-to-income ratio feature."""
    df = df.copy()

    df["debt_income_ratio"] = df["loan_amnt"] / df["customer_income"]
    df["debt_income_ratio"] = df["debt_income_ratio"].replace([np.inf, -np.inf], np.nan)

    return df


def add_credit_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit maturity feature."""
    df = df.copy()

    df["credit_maturity"] = df["cred_hist_length"] / df["customer_age"]
    df["credit_maturity"] = df["credit_maturity"].replace([np.inf, -np.inf], np.nan)

    return df


def add_employment_years(df: pd.DataFrame) -> pd.DataFrame:
    """Convert employment duration in months to years."""
    df = df.copy()

    df["employment_years"] = df["employment_duration"] / 12
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create age group categorical feature."""
    df = df.copy()

    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[18, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
        include_lowest=True,
    )
    return df


@log_execution_time
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    df = df.copy()

    df = add_debt_income_ratio(df)
    df = add_credit_maturity(df)
    df = add_employment_years(df)
    df = add_age_group(df)

    return df
