# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

WORKDIR /app

# Minimal OS deps: TLS certs for HTTPS downloads (HF model downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && update-ca-certificates

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code + vocab files into the image
COPY app /app/app
COPY data_model_v2 /app/data_model_v2

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
