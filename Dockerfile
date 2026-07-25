FROM python:3.11-slim

WORKDIR /app

# libgomp1 is LightGBM's OpenMP runtime dependency — not present in the slim
# base image and LightGBM fails to import without it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API, model library, the pre-trained models, the training data (needed for
# retrain-at-startup / /train / /predict feature building), and the unit
# tests, so the same image can both serve the API and run `pytest` to prove
# the bundle works — per the case-study requirement to bundle API + model +
# tests in one Docker image.
COPY aavail/ aavail/
COPY api/ api/
COPY models/ models/
COPY cs-train/ cs-train/
COPY tests/ tests/
COPY pytest.ini .

ENV AAVAIL_DATA_DIRS=cs-train
ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "api.app:app"]
