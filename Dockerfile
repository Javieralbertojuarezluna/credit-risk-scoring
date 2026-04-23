FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
COPY configs ./configs
COPY data ./data
COPY .env.example ./
COPY README.md ./
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "src/credit_risk/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
