from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.credit_risk.app.inference import prepare_input_data
from src.credit_risk.models.sklearn_model import SklearnCreditModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")

MODEL_PATH = PROJECT_ROOT / "src" / "credit_risk" / "models" / "sklearn_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")


@st.cache_resource
def load_model() -> SklearnCreditModel:
    """Load trained sklearn model."""
    return SklearnCreditModel.load(MODEL_PATH)


def generate_explanation(input_data: dict, probability: float) -> str:
    """Generate a simple business-friendly explanation of the prediction."""
    reasons = []

    if input_data["customer_income"] < 20000:
        reasons.append("ingreso bajo")

    if input_data["loan_int_rate"] > 15:
        reasons.append("tasa de interés elevada")

    if input_data["historical_default"] == 1:
        reasons.append("historial de impagos")

    if input_data["loan_grade"] in ["E", "F", "G"]:
        reasons.append("calificación crediticia baja")

    if input_data["employment_duration"] < 12:
        reasons.append("poca estabilidad laboral")

    if probability > 0.8:
        nivel = "alto riesgo"
    elif probability > 0.5:
        nivel = "riesgo moderado"
    else:
        nivel = "bajo riesgo"

    if reasons:
        return f"El cliente presenta {nivel} debido a: " + ", ".join(reasons) + "."
    return f"El cliente presenta {nivel} con un perfil financiero saludable."


def main() -> None:
    """Main Streamlit app."""
    st.title("Credit Risk Scoring")
    st.write(
        "Esta aplicación estima el riesgo de crédito de un solicitante "
        "a partir de variables financieras y demográficas."
    )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    model = load_model()

    st.sidebar.header("Navegación")
    section = st.sidebar.radio(
        "Ir a:",
        ["Predicción", "Acerca del modelo"],
    )

    if st.sidebar.button("Limpiar historial"):
        st.session_state.prediction_history = []
        st.rerun()

    if section == "Predicción":
        render_prediction_section(model)

    if section == "Acerca del modelo":
        render_about_section()


def render_prediction_section(model: SklearnCreditModel) -> None:
    """Render prediction interface."""
    st.header("Predicción individual")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            customer_age = st.number_input(
                "Edad",
                min_value=18,
                max_value=100,
                value=30,
            )
            customer_income = st.number_input(
                "Ingreso mensual",
                min_value=1,
                value=50000,
            )
            employment_duration = st.number_input(
                "Duración de empleo (meses)",
                min_value=0.0,
                value=24.0,
            )
            home_ownership = st.selectbox(
                "Tipo de vivienda",
                ["RENT", "OWN", "MORTGAGE"],
                format_func=lambda x: {
                    "OWN": "Propia",
                    "RENT": "Rentada",
                    "MORTGAGE": "Hipoteca",
                }.get(x, x),
            )
            historical_default = st.selectbox(
                "Historial de impagos",
                ["N", "Y"],
                format_func=lambda x: {
                    "N": "No",
                    "Y": "Sí",
                }.get(x, x),
            )

        with col2:
            loan_amnt = st.number_input(
                "Monto del préstamo",
                min_value=1.0,
                value=10000.0,
            )
            loan_int_rate = st.number_input(
                "Tasa de interés del préstamo",
                min_value=0.01,
                value=12.0,
            )
            term_years = st.number_input(
                "Plazo del préstamo (años)",
                min_value=1,
                value=5,
            )
            loan_intent = st.selectbox(
                "Propósito del préstamo",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"],
                format_func=lambda x: {
                    "PERSONAL": "Personal",
                    "EDUCATION": "Educación",
                    "MEDICAL": "Salud",
                    "VENTURE": "Inversión",
                }.get(x, x),
            )
            loan_grade = st.selectbox(
                "Grado del préstamo",
                ["A", "B", "C", "D", "E", "F", "G"],
                format_func=lambda x: {
                    "A": "A - Excelente (muy bajo riesgo)",
                    "B": "B - Bueno (bajo riesgo)",
                    "C": "C - Aceptable (riesgo medio)",
                    "D": "D - Riesgoso (riesgo medio-alto)",
                    "E": "E - Alto riesgo",
                    "F": "F - Muy alto riesgo",
                    "G": "G - Riesgo extremo",
                }.get(x, x),
            )
            st.caption(
                "La calificación crediticia refleja el nivel de riesgo "
                "del solicitante según su perfil financiero."
            )
            cred_hist_length = st.number_input(
                "Antigüedad del historial crediticio (años)",
                min_value=0,
                value=5,
            )

        submitted = st.form_submit_button("Calcular riesgo")

    if submitted:
        input_data = {
            "customer_id": 999999.0,
            "customer_age": int(customer_age),
            "customer_income": int(customer_income),
            "home_ownership": home_ownership,
            "employment_duration": float(employment_duration),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "term_years": int(term_years),
            "historical_default": 1 if historical_default == "Y" else 0,
            "cred_hist_length": int(cred_hist_length),
        }

        df_input = prepare_input_data(input_data)

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0, 1]

        explanation = generate_explanation(input_data, probability)

        history_row = {
            "customer_age": int(customer_age),
            "customer_income": int(customer_income),
            "home_ownership": home_ownership,
            "employment_duration": float(employment_duration),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "term_years": int(term_years),
            "historical_default": 1 if historical_default == "Y" else 0,
            "cred_hist_length": int(cred_hist_length),
            "probability_default": float(probability),
            "prediction": int(prediction),
        }

        st.session_state.prediction_history.append(history_row)

        st.subheader("Resultado")
        st.metric("Probabilidad de impago (default)", f"{probability:.2%}")

        if prediction == 1:
            st.error("El solicitante presenta ALTO riesgo de impago.")
        else:
            st.success("El solicitante presenta BAJO riesgo de impago.")

        st.write("### Interpretación del resultado")
        st.info(explanation)

        chart_df = pd.DataFrame(
            {
                "Clase": ["No Default", "Default"],
                "Probabilidad": [1 - probability, probability],
            }
        )
        st.bar_chart(chart_df.set_index("Clase"))

        st.write("### Variables derivadas calculadas")
        st.dataframe(df_input, use_container_width=True)

    st.write("### Historial de clientes evaluados")
    history_df = pd.DataFrame(st.session_state.prediction_history)

    if not history_df.empty:
        history_df["prediction"] = history_df["prediction"].map(
            {0: "Bajo riesgo", 1: "Alto riesgo"}
        )
        history_df["probability_default"] = history_df["probability_default"].map(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(history_df, use_container_width=True)


def render_about_section() -> None:
    """Render model information."""
    st.header("Acerca del modelo")
    st.write(
        """
        Este proyecto compara dos enfoques para estimar riesgo crediticio:

        - Regresión logística con pipeline de scikit-learn
        - MLP implementado en PyTorch

        En esta primera versión de la app se utiliza el modelo de scikit-learn
        para inferencia, debido a su estabilidad e interpretabilidad.
        """
    )

    metrics_df = pd.DataFrame(
        {
            "Modelo": ["Logistic Regression", "PyTorch MLP"],
            "Precision Default": [0.9037, 0.9771],
            "Recall Default": [0.7822, 0.8721],
            "F1 Default": [0.8386, 0.9216],
            "ROC-AUC": [0.9668, 0.9866],
        }
    )

    st.write("### Comparación de modelos")
    st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    main()
