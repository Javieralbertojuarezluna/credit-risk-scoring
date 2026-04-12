from __future__ import annotations

import pandera.pandas as pa
from pandera import Check
from pandera.typing import DataFrame


def get_credit_risk_schema() -> pa.DataFrameSchema:
    """Return Pandera schema for the processed credit risk dataset."""
    return pa.DataFrameSchema(
        {
            "customer_id": pa.Column(float, nullable=False),
            "customer_age": pa.Column(int, Check.ge(18), nullable=False),
            "customer_income": pa.Column(int, Check.gt(0), nullable=False),
            "home_ownership": pa.Column(str, nullable=False),
            "employment_duration": pa.Column(float, Check.ge(0), nullable=False),
            "loan_intent": pa.Column(str, nullable=False),
            "loan_grade": pa.Column(str, nullable=False),
            "loan_amnt": pa.Column(float, Check.gt(0), nullable=False),
            "loan_int_rate": pa.Column(float, Check.gt(0), nullable=False),
            "term_years": pa.Column(int, Check.gt(0), nullable=False),
            "historical_default": pa.Column(float, Check.isin([0, 1]), nullable=False),
            "cred_hist_length": pa.Column(int, Check.ge(0), nullable=False),
            "Current_loan_status": pa.Column(str, nullable=False),
            "target": pa.Column(int, Check.isin([0, 1]), nullable=False),
            "debt_income_ratio": pa.Column(float, Check.ge(0), nullable=True),
            "credit_maturity": pa.Column(float, Check.ge(0), nullable=True),
            "employment_years": pa.Column(float, Check.ge(0), nullable=True),
            "age_group": pa.Column(object, nullable=True),
        },
        strict=False,
        coerce=True,
    )


def validate_credit_risk_data(df) -> DataFrame:
    """Validate processed credit risk dataframe against Pandera schema."""
    schema = get_credit_risk_schema()
    return schema.validate(df)
