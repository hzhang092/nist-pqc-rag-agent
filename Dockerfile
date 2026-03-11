# syntax=docker/dockerfile:1.7

ARG PYTORCH_CUDA_BASE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_CUDA_BASE} AS base-cuda

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/conda/bin:${PATH} \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

FROM base-cuda AS builder-api

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Reuse the base image's preinstalled torch/CUDA stack from /opt/conda.
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"

COPY pyproject.toml README.md /app/
COPY api /app/api
COPY eval /app/eval
COPY rag /app/rag

RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[core,retrieval,agent]"

FROM builder-api AS builder-allinone

RUN pip install ".[ingest]"

FROM base-cuda AS api-cuda

COPY --from=builder-api /opt/venv /opt/venv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base-cuda AS allinone-cuda

COPY --from=builder-allinone /opt/venv /opt/venv
COPY scripts /app/scripts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--reload","--host", "0.0.0.0", "--port", "8000"]
