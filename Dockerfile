# Production Dockerfile for the KAIRO FastAPI backend
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (compiler for hmmlearn wheels on some platforms, curl for the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Application code, trained models and research results
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY results/ ./results/

# Run as a non-root user; /app/data holds the SQLite execution store (mount a volume here)
RUN useradd --system --uid 1001 kairo \
    && mkdir -p /app/data \
    && chown -R kairo:kairo /app/data
USER kairo

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
