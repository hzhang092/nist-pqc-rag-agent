# Docker Setup Summary (2026-02-20)

## Scope

This summary captures the Docker-related work completed for the project packaging milestone:

- Step 2: dependency pinning and alignment
- Step 3: Docker build context hardening
- Step 4: base Docker image and build flow
- Step 5: docker compose runtime setup with persistent mounts
- Step 6: env/secrets template and runtime configuration
- Step 7: developer task runner and README runbook
- Follow-up: force CPU-only PyTorch to avoid CUDA/NVIDIA wheel downloads

The Week-1 data contracts were preserved:

- chunk citations remain `doc_id`, `start_page`, `end_page`
- no JSONL schema changes were introduced
- deterministic retrieval behavior remained stable in smoke tests

## Success Criteria

- A clean machine can build and run retrieval with Docker Compose.
- Dockerized runs persist generated artifacts in host `data/`, `reports/`, and `runs/`.
- Containerized deterministic retrieval tests pass.
- Docker image uses CPU-only torch (`torch==2.7.0+cpu`) and does not install CUDA/NVIDIA wheels.

## Files Added

- `.dockerignore`
- `.env.example`
- `Dockerfile`
- `docker-compose.yml`
- `scripts/docker.ps1`

## Files Updated

- `requirements.txt`
- `pyproject.toml`
- `README.md`

## Change Summary By File

### `requirements.txt`

- Pinned runtime dependencies to concrete versions used by this repo.
- Added explicit test dependency (`pytest`) for container sanity checks.

### `pyproject.toml`

- Aligned `[project].dependencies` with pinned runtime deps.
- Added `[project.optional-dependencies].dev` with `pytest`.

### `.dockerignore`

- Excluded VCS/cache/temp files and local artifacts (`runs`, `reports/eval`, `data/processed`).
- Excluded `.env` to prevent secret leakage into image layers.

### `Dockerfile`

- Added deterministic and container-friendly env defaults:
  - `PYTHONDONTWRITEBYTECODE=1`
  - `PYTHONUNBUFFERED=1`
  - `PYTHONHASHSEED=0`
  - `PIP_NO_CACHE_DIR=1`
- Installed dependencies in cached layers.
- Forced CPU-only torch installation:
  - `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0`

### `docker-compose.yml`

- Added `rag` service and default command.
- Mounted host directories:
  - `./data:/app/data`
  - `./reports:/app/reports`
  - `./runs:/app/runs`
- Added named volume `hf_cache` for model cache reuse.
- Wired API key and retrieval env variables via compose environment.

### `.env.example`

- Added API key placeholders and defaults for retrieval/answer/agent settings.
- Included determinism knobs (`LLM_TEMPERATURE`, `PYTHONHASHSEED`).

### `scripts/docker.ps1`

- Added repeatable task entrypoints:
  - `build`, `ingest`, `clean`, `chunk`, `embed`, `index-faiss`, `index-bm25`
  - `search`, `ask`, `eval`, `test`, `pipeline`
- Fixed argument handling bug in compose invocation helper.

### `README.md`

- Added Docker quickstart:
  - `.env` setup
  - image build
  - ingestion/index pipeline commands
  - retrieval/eval commands
  - PowerShell task-runner shortcuts
- Documented CPU-only torch decision.

## Verification Commands Run

The following commands were executed after changes:

```powershell
docker compose config
docker compose build rag
docker compose run --rm rag python -m rag.search "ML-KEM key generation"
docker compose run --rm rag python -m pytest tests/test_embed_store_records.py tests/test_retrieve_determinism.py tests/test_retrieve_rrf.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/docker.ps1 -Task test
docker compose run --rm rag python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Verification Outcomes

- Compose file parsed successfully.
- Docker image built successfully.
- Retrieval command returned expected top-k cited hits.
- Determinism-focused tests passed in container.
- CPU-only torch confirmed:
  - version reported as `2.7.0+cpu`
  - `torch.version.cuda` is `None`
  - `torch.cuda.is_available()` is `False`

## Notes / Known Behavior

- First `sentence-transformers` query in a fresh cache may still be slower due to model weight download.
- Hugging Face warns when unauthenticated; optional `HF_TOKEN` can reduce rate-limit friction.
