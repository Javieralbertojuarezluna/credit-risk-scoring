# Credit Risk Scoring

## 1. Overview

Este proyecto desarrolla un sistema de **evaluación de riesgo crediticio** utilizando técnicas de Machine Learning tradicionales (scikit-learn) y Deep Learning (PyTorch), integrando buenas prácticas de ingeniería de software y MLOps.

Se implementa un pipeline completo que incluye:
- limpieza y validación de datos
- feature engineering
- entrenamiento de modelos
- evaluación y comparación
- tracking de experimentos
- aplicación interactiva (Streamlit)
- despliegue con Docker

---

## 2. Problem Statement

En el sector financiero, es crítico estimar la probabilidad de que un cliente incurra en **default (incumplimiento de pago)**.

El objetivo de este proyecto es:

> Construir un modelo que estime la probabilidad de default de un cliente a partir de variables financieras y demográficas.

Este problema es de clasificación binaria:
- 0 → No default
- 1 → Default

---

## 3. Dataset

El dataset contiene información de clientes como:

- edad (`customer_age`)
- ingreso (`customer_income`)
- tipo de vivienda (`home_ownership`)
- duración de empleo (`employment_duration`)
- monto del préstamo (`loan_amnt`)
- tasa de interés (`loan_int_rate`)
- historial crediticio (`cred_hist_length`)
- historial de default (`historical_default`)

---

## 4. Data Pipeline

El pipeline de datos está dividido en tres capas principales:

### 4.1 Data Loading
- lectura desde YAML (`paths.yaml`)
- uso de `pathlib`
- separación de configuración

### 4.2 Preprocessing

Incluye:

- limpieza de variables (ej. `loan_amnt`)
- conversión de tipos
- eliminación de datos inválidos (ej. edad < 18)
- creación de variable objetivo (`target`)
- imputación de valores faltantes

```python
df = preprocess_credit_data(df, model_config)
```

### 4.3 Feature Engineering

Se crean variables derivadas:

- debt_income_ratio
- credit_maturity
- employment_years
- age_group

```python
df = build_features(df)
```

### 4.4 Validación con Pandera

Se utiliza Pandera para asegurar:

- tipos correctos
- rangos válidos
- consistencia de datos

```python
df = validate_credit_risk_data(df)
```

### 4.5 Architecture Overview

Flujo del sistema:

Data -> Preprocessing -> Feature Engineering -> Model -> API (Streamlit) -> User Interface -> Docker - External Access (ngrok)

## 5. Models

### 5.1 Logistic Regression (scikit-learn)

Se implementa un pipeline con:

- ColumnTransformer
- StandardScaler
- OneHotEncoder
- LogisticRegression

Incluye:

- validación cruzada
- búsqueda de hiperparámetros

### 5.2 MLP(PyTorch)

Se implementa un modelo de red neuronal:

- arquitectura multicapa
- función de pérdida: BCEWithLogitsLoss
- optimizador: Adam
- uso de Dataset custom y DataLoader

## 6. Model Evaluation

Se evaluan los modelos con:
- Precision
- Recall
- F1-score
- ROC-AUC

Se utiliza una dataclass (ModelMetrics) para estructurar resultados.

## Resultados:

| Modelo              | Precision | Recall | F1     | ROC-AUC |
| ------------------- | --------- | ------ | ------ | ------- |
| Logistic Regression | 0.9037    | 0.7822 | 0.8386 | 0.9668  |
| PyTorch MLP         | 0.9771    | 0.8721 | 0.9216 | 0.9866  |


## 7. Model Comparison

El modelo de PyTorch presenta:

- mejor recall
- mejor ROC-AUC

Lo cual es relevabte en riesgo crediticio, donde detectar defaultes es prioritario.

Sin embargo, la regresión logística ofrece:

- mayor interpretabilidad
- mayor estabilidad

## 8. Experiment Tracking (MLflow)

Se utiliza MLflow para registrar:

- parámetros
- métricas
- artefactos (modelos)

``` bash
mlflow ui
```

## 9. Testing

Se implementan pruebas con pytest:

- carga de datos
- preprocessing
- feature engineering
- modelo sklearn
- modelo PyTorch

Resultado:

- Total tests: 5
- Passed: 5
- Failed: 0

## 10. Code Quality

Se utilizan:

- ruff (linting y formating)
- pre-commit (validación automática)


## 11. Application(Streamlit)

Se desarrolló una aplicación interactiva que permite:

- ingresar datos del cliente
- calcular probabilidad de default
- visualizar resultados
- comparar modelos

``` bash
uv run streamlit run src/credit_risk/app/streamlit_app.py
```

### 11.1 Model Interpretation (Business Layer)

Además de la probabilidad de default, la aplicación proporciona una interpretación cualitativa del riesgo basada en las variables clave del cliente.

Por ejemplo:
- Bajo ingreso anual
- Alto monto del préstamo
- Historial previo de impago

Esto permite el output del modelo en insights accionables para áreas de negocio.

## 12. Deployment (Docker)

Se containerizó la aplicación.

### 12.1 Build

``` bash
docker build -t credit-risk-scoring .
```

### 12.2 Run
``` bash
docker run -p 8501:8501 credit-risk-scoring
```

Acceder en:

http://localhost:8501

También se probó despliegue público usando ngrok.

### 12.3 Public Access (ngrok)

Para pruebas externas, se expuso la aplicación mediante un túnel seguro utilizando ngrok:
``` bash
ngrok http 8501
```

## 12.4 Docker Details

La aplicación se ejecuta dentro de un contenedor basado en `python:3.12-slim`.

Incluye:

- Instalación de dependencias con `uv`
- copia de código fuente, modelos y configuraciones
- ejecución de Streamlit como servicio principal

El contenedor expone el puerto 8501 para acceso a la aplicación.

Este enfoque permite:

- portabilidad entre entornos
- aislamiento de dependencias
- despliegie consistente

## 13. Project Structure

src/
  credit_risk/
    data/
    features/
    models/
    evaluation/
    tracking/
    app/
    utils/
tests/
configs/
notebooks/
    models/

## 14. How to run

### 14.1 Instalar dependencias
``` bash
uv sync
```

### 14.2 Ejecutar tests
``` bash
uv run pytest
```

### 14.3 Ejecutar app
``` bash
uv run streamlit run src/credit_risk/app/streamlit_app.py
```

## 15. Design Decisions

- Eliminación de edades inválidad (-18 años) para evitar ruido
- Uso de Pandera para validación estructural
- Uso de MLflow para trazabilidad y posible gobierno de modelos
- Uso de PyTorch para explorar modelos no lineales
- Uso de dataclass para estructurar métricas


## 16. Limitaciones

- dataset sintético / simplificado
- falta interpretación (SHAP, etc.)

## 17. Future work

- Incorporar explainability
- mejorar arquitectura de red neuronal
- agregar monitoreo de drift
- despliegue en cloud

## 18. Conclusión
Este proyecto demuestra la construcción de un sistema de Machine Learning end-to-end, desde el procesamiento de datos hasta su despliegue como aplicación interactiva.

Se integran prácticas de:

- ingeniería de datos
- modelado
- MLOps
- desarrollo de producto

El resultado es una solución reproducible y escalable para evaluación de riesgo crediticio.

## 19. Author

Proyecto desarrollado como parte del diplonado de AI & LLMs for Financial Markets - Módulo Python Avanzado

Alumno: Javier Alberto Juarez Luna
