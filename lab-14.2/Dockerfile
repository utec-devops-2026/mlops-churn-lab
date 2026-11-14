FROM python:3.10-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app/ ./app/

# Artefactos que produce el job evaluate del pipeline
COPY data/production_model.pkl       ./data/production_model.pkl
COPY data/production_model_info.json ./data/production_model_info.json

# El proceso no corre como root
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

ENV PREDICTION_LOG=/app/logs/predictions.jsonl
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
