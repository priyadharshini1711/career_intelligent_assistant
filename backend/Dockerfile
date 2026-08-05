# Slim rather than Alpine: sentence-transformers pulls scientific wheels, and
# musl means building several of them from source instead of downloading them.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep the model cache inside the image so a cold container does not
    # download 90MB on its first request.
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /app

# curl is only here for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Torch first, from the CPU index. The default wheel bundles CUDA and costs
# roughly 2.5GB in an image that will never see a GPU -- this cuts it to ~200MB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image. Without this the first request after
# every deploy pays a download, and the container needs egress to Hugging Face
# just to answer a question.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY pyproject.toml .

# Non-root, and the model cache has to be readable by it.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /opt/models /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/system/health || exit 1

# No --reload, and a single worker: session state lives in process memory, so
# a second worker would serve requests for documents it has never seen. The
# README covers what has to change before this can scale horizontally.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
