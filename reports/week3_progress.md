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

## 2026-03-09 - Day 2 query analysis node + explicit `/ask-agent`

### Goal
Upgrade the LangGraph path from heuristic routing to explicit structured query analysis, while keeping the answer node and direct `/ask` path stable.

This update was intentionally scoped to:
- add a schema-enforced `analyze_query` node
- make the graph consume analyzed `canonical_query`, `mode_hint`, `required_anchors`, `compare_topics`, and optional `doc_ids`
- expose a real `/ask-agent` serving path

This update explicitly did not:
- redesign the answer node contract
- refactor the LLM backend interface for repo-wide structured output
- change direct `/ask` retrieval defaults

### High-Level Before/After Graph

### Before
`route -> retrieve -> assess_evidence -> optional refine_query -> answer -> verify_or_refuse`

Behavior before this update:
- graph entry began at `route`
- route still depended on `_heuristic_route(...)`
- compare topic parsing and anchor extraction were split across multiple helper paths
- graph retrieval tools still inferred `mode_hint` and allowed default query-variant expansion
- FastAPI exposed `/health`, `/search`, `/ask`, but not `/ask-agent`

### After
`analyze_query -> route -> retrieve -> assess_evidence -> optional refine_query -> answer -> verify_or_refuse`

Behavior after this update:
- graph entry begins with `analyze_query`
- routing is now driven by analyzed fields rather than top-level heuristic intent parsing
- graph retrieval uses explicit analyzed inputs and disables retrieval-side auto query expansion for the graph path
- FastAPI now exposes `/ask-agent`
- direct `/ask` remains the unchanged control path

### Implementation

#### 1) Query analysis state and schema
Extended the LangGraph state with explicit analysis fields:
- `original_query`
- `canonical_query`
- `mode_hint`
- `required_anchors`
- `compare_topics`
- `doc_ids`
- `doc_family`
- `analysis_notes`
- `answer_prompt_question`
- serialized `query_analysis`

The graph now records both:
- retrieval-oriented normalized intent (`canonical_query`)
- user-facing answer wording (`answer_prompt_question = original_query`)

#### 2) `analyze_query` node
Added `node_analyze_query` as the first graph node.

Implementation style:
- deterministic pre-extraction first for:
  - `Algorithm N`
  - `Table N`
  - `Section x.y`
  - dotted identifiers such as `ML-KEM.Decaps`
  - explicit doc mentions such as `FIPS 203`, `ML-KEM`, `SP 800-227`
  - compare topic pairs
- bounded LLM JSON analysis at temperature `0`
- local schema validation
- deterministic fallback if the model response is invalid or unavailable

The analysis schema now includes:
- `canonical_query`
- `mode_hint` constrained to `definition | algorithm | compare | general`
- `required_anchors`
- `compare_topics`
- `doc_ids`
- optional `doc_family`
- optional `analysis_notes`

#### 3) Routing, retrieval, and refinement
`route` was downgraded to a thin dispatcher over analyzed fields:
- if `compare_topics` exists, route to `compare`
- if `mode_hint == definition`, route to `resolve_definition`
- otherwise route to `retrieve`

Graph retrieval now consumes analyzed values directly:
- `canonical_query`
- `mode_hint`
- `doc_ids`

For LangGraph-entered requests:
- retrieval-side automatic `mode_hint` inference is disabled
- retrieval-side query fusion / mode-variant expansion is disabled

Day 2 retrieval filtering was kept intentionally narrow:
- only explicit allowlisted document IDs are supported
- scope is limited to:
  - `NIST.FIPS.203`
  - `NIST.FIPS.204`
  - `NIST.FIPS.205`
  - `NIST.SP.800-227`
  - `NIST.IR.8545`
  - `NIST.IR.8547.ipd`
- shared retrieval uses deterministic post-filtering when backend-native filtering is unavailable

Refinement now starts from analyzed state rather than reparsing the raw question where avoidable:
- anchor checks use `required_anchors`
- compare refinement uses `compare_topics`
- refine queries start from `canonical_query`

#### 4) Explicit agent serving path
Added a shared-service agent wrapper and FastAPI `POST /ask-agent`.

`/ask-agent` returns:
- `answer`
- `citations`
- `refusal_reason`
- `trace_summary`
- `timing_ms`
- `analysis`

The graph path now exposes trace-visible analysis fields such as:
- `original_query`
- `canonical_query`
- `mode_hint`
- `compare_topics`
- `doc_ids`
- `answer_prompt_question`

#### 5) Intentional non-change: answer node
The answer path was left intentionally stable for Day 2:
- `node_answer` still calls `_call_rag_answer(...)`
- answer-side citation recovery remains a later Week 3 task
- direct `/ask` was not rewritten

### Regression Coverage
Added or updated tests for:
- deterministic analysis output for definition, algorithm, compare, and general queries
- fallback behavior when analysis JSON is invalid
- graph trace order beginning with `analyze_query`
- routing from analyzed fields rather than heuristic compare parsing
- graph retrieval plumbing for `mode_hint`, `doc_ids`, and disabled query fusion
- `/ask-agent` service and API behavior
- retrieval eval adapter compatibility after adding optional `doc_ids`

### Verification
Verified with:
- baseline before edits:
  - `conda run -n pyt python -m pytest -q tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py`
- implementation verification:
  - `conda run -n pyt python -m py_compile rag/lc/state.py rag/lc/state_utils.py rag/lc/graph.py rag/lc/tools.py rag/retrieve.py rag/service.py api/main.py tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py tests/test_retrieve_eval_api.py`
  - `conda run -n pyt python -m pytest -q tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py`
  - `conda run -n pyt python -m pytest -q tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py tests/test_retrieve_eval_api.py`

Observed results:
- baseline slice before edits: `20 passed, 1 skipped`
- Day 2 verification slice after edits: `30 passed, 1 skipped`
- extended Day 2 slice after edits: `33 passed, 1 skipped`
- syntax / import compilation passed for all touched Day 2 modules and tests

Not completed in this update:
- live uvicorn smoke of `/ask-agent` against a real local model backend
- full-suite rerun beyond the targeted Day 2 slice

### Status
- query-analysis state: complete
- `analyze_query` graph entry: complete
- analysis-driven routing: complete
- graph-path retrieval plumbing for `mode_hint` / `doc_ids`: complete
- graph-path disablement of auto query fusion / mode inference: complete
- FastAPI `/ask-agent`: complete
- direct `/ask` preserved as control path: complete
- targeted Day 2 verification: complete
- live real-model `/ask-agent` smoke: not completed in this update
- full-suite green baseline: not restored in this update
