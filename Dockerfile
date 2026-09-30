# Scoring API image. The trained model bundle is NOT baked in: mount it at run time
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" credit-risk-api
# so code and model releases are versioned independently.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=/app/models/model_bundle.joblib
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
