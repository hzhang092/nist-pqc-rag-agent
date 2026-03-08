# Week 3 Progress

## 2026-03-07 - Prose-spacing fix for identifier normalization

### Problem
The parser output in `data/processed/*_parsed.json` was removing the space before a word when the previous token ended with `.`.

Observed regressions included:
- `U.S.Department`
- `Gina M.Raimondo`
- `Laurie E.Locascio`
- sentence joins such as `channel.A shared secret key`
- numbered prose joins such as `4.Approving Authority.`

This came from the identifier-normalization feature that was introduced to preserve dotted PQC keywords such as `ML-KEM.Decaps` and `ML-KEM.ParamSets`.

### Root cause
The issue was not in Docling markdown extraction itself.

- `markdown` in parsed artifacts was already correct.
- The corruption happened in `rag/parsers/base.py` when `markdown_to_text()` called `normalize_identifier_like_spans()` from `rag/text_normalize.py`.
- The old implementation used a span-wide regex that compacted any alnum `.` alnum pattern, which was too broad for normal prose.

### Implementation
Updated `rag/text_normalize.py` to use pairwise, separator-aware normalization instead of span-wide compaction.

Behavior after the fix:
- keep escaped-separator cleanup for `\\_`, `\\.`, `\\-`
- keep spacing compaction for `_` and `-` between identifier tokens
- compact unescaped `.` only when it matches one of these supported identifier cases:
  - numeric refs such as `3 . 3 -> 3.3`
  - hyphen/underscore identifiers such as `ML-KEM . Decaps -> ML-KEM.Decaps`
  - all-caps dotted chains such as `NIST . FIPS . 203 -> NIST.FIPS.203`
- preserve normal prose spacing after periods

No change was needed in `rag/parsers/docling_backend.py`.

### Regression coverage
Added tests for:
- positive normalization cases in `tests/test_text_normalize.py`
- negative prose-preservation cases in `tests/test_text_normalize.py`
- parser-layer `markdown_to_text()` regression in `tests/test_parsers_base.py`

### Verification
Verified with:
- `conda run -n pyt python -m pytest tests/test_text_normalize.py tests/test_parsers_base.py tests/test_docling_backend.py`

Result:
- `10 passed`

Targeted Docling parse verification on FIPS 203 confirmed:
- page 1 contains `U.S. Department of Gina M. Raimondo` and `Laurie E. Locascio`
- page 3 contains `channel. A shared secret key`
- page 4 contains `4. Approving Authority. Secretary of Commerce.`

### Status
- code fix: complete
- regression tests: complete
- full artifact rebuild / full eval rerun: not completed in this update

## 2026-03-07 - Day 1 local serving path (Ollama SDK + FastAPI)

### Goal
Close the main Week 3 productization gap by adding a configurable local generation path and a small API surface without changing retrieval behavior.

This update keeps the retrieval stack frozen and changes only:
- generation backend resolution
- shared service orchestration
- shallow latency hooks
- FastAPI service endpoints

### High-Level Before/After Architecture

### Before
`raw_pdfs -> rag.ingest (parser factory: llamaparse|docling) -> pages.jsonl(+optional markdown/parser metadata) -> clean -> chunk(v1|v2 markdown-aware) -> embed(+breadcrumb text) -> faiss/bm25 -> retrieve/eval -> rag.ask (Gemini-only generation path)`

### After
`raw_pdfs -> rag.ingest (parser factory: llamaparse|docling) -> pages.jsonl(+optional markdown/parser metadata) -> clean -> chunk(v1|v2 markdown-aware) -> embed(+breadcrumb text) -> faiss/bm25 -> retrieve(+retrieve/rerank timing hooks) -> rag.service -> llm backend factory (gemini|ollama via Ollama Python SDK) -> rag.ask / rag.lc.graph / FastAPI(/health,/search,/ask) -> traces/eval`

### Implementation

#### 1) LLM backend abstraction
Added a small generation interface under `rag/llm/`:
- `base.py` defines the backend protocol
- `factory.py` resolves `gemini` vs `ollama`
- `gemini.py` remains the Gemini backend implementation
- `ollama.py` now uses the official Ollama Python SDK client instead of hand-written HTTP calls

This keeps application code depending on a stable model boundary rather than backend-specific imports.

#### 2) Shared service layer
Added `rag/service.py` to centralize:
- `ask_question(...)`: 
- `search_query(...)`: Handles search requests by retrieving relevant document chunks based on a query.
- `health_status(...)`: Checks the health of the LLM backend and retrieval components.

This prevents CLI and API paths from drifting and keeps the Day 1 output contract in one place.

The service layer now provides:
- explicit `refusal_reason`
- `trace_summary`
- `timing_ms`
- `/search` metadata enrichment via `chunk_id -> {section_path, block_type}` join against processed chunk metadata

#### 3) FastAPI surface
Added `api/main.py` with:
- `GET /health`
- `GET /search`
- `POST /ask`

Endpoint behavior:
- `/health` reports `status`, `llm_backend`, `llm_model`, `backend_reachable`, `retrieval_ready`
- `/search` returns retrieval hits plus `section_path` and `block_type` when present
- `/ask` returns `answer`, `citations`, `refusal_reason`, `trace_summary`, and `timing_ms`

#### 4) Timing hooks
Added shallow Day 1 timing only:
- retrieval latency
- rerank latency
- generation latency
- total latency

No retrieval ranking logic or retrieval defaults were changed in this update.

### Verification
Verified with:
- `conda run -n pyt python -m pytest -q tests/test_rag_answer.py tests/test_rag_answer_deterministic.py tests/test_retrieve_determinism.py tests/test_llm_factory.py tests/test_api.py`
- `conda run -n pyt python -m rag.search_faiss "ML-KEM key generation"`
- `LLM_BACKEND=ollama LLM_MODEL="qwen3:8B" LLM_BASE_URL="http://localhost:11434" conda run -n pyt python -m rag.ask "What is ML-KEM?" --json --no-evidence`
- `LLM_BACKEND=ollama LLM_MODEL="qwen3:8B" LLM_BASE_URL="http://localhost:11434" conda run -n pyt uvicorn api.main:app --host 127.0.0.1 --port 8001`

Observed results:
- targeted Day 1 test gate passed (`27 passed`)
- `rag.ask` returned cited output through Ollama with `timing_ms`
- real uvicorn smoke passed for:
  - `GET /health`
  - `GET /search?q=ML-KEM%20key%20generation&k=3`
  - `POST /ask`

Full-suite regression check in `pyt` was also run:
- `106 passed, 2 failed, 2 skipped`
- failures remain in `tests/test_docling_preflight.py`
- these were not part of the Day 1 serving-path change

### Status
- local LLM backend abstraction: complete
- Ollama SDK backend: complete
- shared service layer: complete
- FastAPI `/health`, `/search`, `/ask`: complete
- shallow timing hooks: complete
- targeted Day 1 verification: complete
- full-suite green baseline: not restored in this update
