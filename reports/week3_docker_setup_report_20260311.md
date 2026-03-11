# Week 3 Docker Setup Report

## What this Docker setup is trying to do

The project now has a Docker setup that is meant to support two different jobs:

1. run the FastAPI app for normal question answering
2. run the heavier ingestion and indexing pipeline when you want to rebuild artifacts

The goal is to keep one Dockerfile and one compose file, but still avoid turning every container into a giant "do everything all the time" image.

## High-level design

This setup uses:

- one `Dockerfile`
- two image targets
- one `docker-compose.yml`
- mounted folders for data, reports, runs, and Hugging Face cache

The two image targets are:

- `api-cuda`
  - meant for serving the API
  - includes app code and runtime dependencies for retrieval and answering
  - does not include the ingestion-only packages

- `allinone-cuda`
  - meant for ingestion, chunking, embedding, indexing, and optional serving
  - includes everything in `api-cuda`
  - adds ingestion packages like `docling`, `llama-parse`, and `pypdf`

## How the Dockerfile is set up

The Dockerfile starts from a CUDA-enabled PyTorch image:

- `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`

That base was chosen so it lines up with the current FAISS package:

- `faiss-gpu-cu12`

The Dockerfile uses multiple stages:

- `base-cuda`
  - common base image
  - common environment variables
  - working directory

- `builder-api`
  - creates a Python virtual environment
  - installs only the packages needed for the API and retrieval path
  - reuses the PyTorch/CUDA stack already present in the base image

- `builder-allinone`
  - starts from `builder-api`
  - installs the ingestion-only dependencies

- `api-cuda`
  - final runtime image for the API
  - copies in the prepared virtual environment
  - starts FastAPI with `uvicorn`

- `allinone-cuda`
  - final runtime image for the full pipeline
  - copies in the fuller virtual environment
  - also starts FastAPI by default, but is mainly used with `docker compose run ...`

In simple terms: the Dockerfile builds the Python environment once, then reuses it in smaller runtime images.

## How Docker Compose is set up

The compose file defines two services:

- `api`
  - default service
  - builds from the `api-cuda` target
  - exposes port `8000`
  - meant for `/health`, `/search`, `/ask`, and `/ask-agent`

- `allinone`
  - optional service under the `allinone` profile
  - builds from the `allinone-cuda` target
  - exposes port `8001`
  - requests GPU access
  - meant for ingestion and index-building jobs

The compose file also shares environment variables across both services through a common block so the runtime settings stay consistent.

## What is mounted from the host

The setup intentionally keeps large and changing data outside the image.

Mounted paths include:

- `data/raw_pdfs`
- `data/processed`
- `reports`
- `runs`
- a named Hugging Face cache volume

This matters because:

- raw PDFs should not be baked into image layers
- processed artifacts can be rebuilt and should stay on the host
- reports and run traces should survive container rebuilds
- model caches are large and should be reused across runs

## How model access works

By default, the container is set up to talk to a model server that runs outside the container.

The default `.env` uses:

- `LLM_BACKEND=ollama`
- `LLM_BASE_URL=http://host.docker.internal:11434`

That means the app container expects Ollama to be running on the host machine, not inside Docker.

If you want to use Gemini instead, you can switch:

- `LLM_BACKEND=gemini`
- set `GEMINI_API_KEY`

## Normal ways to use it

### 1. Start the API for normal use

```bash
cp .env.example .env
docker compose build api
docker compose up --build api
```

Once it is running, the main API is available on port `8000`.

Useful endpoints:

- `GET /health`
- `GET /search`
- `POST /ask`
- `POST /ask-agent`

### 2. Rebuild the retrieval artifacts

If you want to rerun ingestion or indexing, use the GPU-capable all-in-one service:

```bash
docker compose --profile allinone build allinone
docker compose --profile allinone run --rm allinone python -m rag.ingest
docker compose --profile allinone run --rm allinone python scripts/clean_pages.py
docker compose --profile allinone run --rm allinone python scripts/make_chunks.py
docker compose --profile allinone run --rm allinone python -m rag.embed
docker compose --profile allinone run --rm allinone python -m rag.index_faiss
docker compose --profile allinone run --rm allinone python -m rag.index_bm25
```

This path is the one to use when you need:

- new parsed pages
- new chunks
- new embeddings
- new FAISS and BM25 indexes

### 3. Run search or eval inside Docker

```bash
docker compose run --rm api python -m rag.search "ML-KEM key generation"
docker compose run --rm api python -m eval.run
```

## Suggested day-to-day workflow

For normal development:

1. keep your processed artifacts on the host
2. run `docker compose up --build api`
3. hit the API endpoints from your browser, `curl`, or client code

Only use `allinone` when you need to rebuild the retrieval pipeline.

That keeps the normal loop simpler and avoids paying the cost of the full ingestion stack every time.

## Helper script

There is also a helper script:

- `scripts/docker.ps1`

It wraps common tasks such as:

- `build`
- `serve`
- `build-allinone`
- `pipeline`
- `search`
- `ask`
- `ask-agent`

This is mainly a convenience layer over the compose commands.

## Things to know

- The images are still large because the base is a CUDA PyTorch runtime image.
- The setup is optimized to reduce extra bloat above that base, not to make a tiny image.
- The API image is the default path.
- The all-in-one image is the rebuild path.
- The current `uvicorn` command uses `--reload`, which is convenient for development but is not the final production-style setting.

## Bottom line

The Docker setup is organized around one simple idea:

- use a smaller serving image for everyday API work
- use a fuller GPU image only when you need to rebuild the document pipeline

That gives the project a cleaner "small internal AI system" story:

- an API that can be started quickly
- a separate rebuild path for ingestion and indexing
- persistent host-mounted data
- GPU support where it actually matters
