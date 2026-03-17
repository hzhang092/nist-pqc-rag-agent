# Repo-Grounded Technical Review

Date: 2026-03-12

## What this repo currently is

This repo is a self-contained, local-first NIST PQC assistant: six core NIST PDFs are already checked into `data/raw_pdfs`, processed artifacts and indexes are already present in `data/processed`, and the code supports two user paths today: a direct citation-enforced QA path and a bounded LangGraph agent path. The implementation is strongest in deterministic ingestion artifacts, structure-aware chunking, hybrid retrieval, citation contracts, and retrieval evaluation. Graph-lite and Neo4j are present, but they are not part of the main answer path.

## Files and directories examined most closely

- Top-level docs and planning
  - `README.md`
  - `reports/project_overview.md`
  - `reports/week3_plan.md`
  - `reports/week3_progress.md`
  - `reports/week3_docker_setup_report_20260311.md`
- Packaging and environment
  - `pyproject.toml`
  - `requirements.txt`
  - `.env.example`
  - `.dockerignore`
- Docker and runtime
  - `Dockerfile`
  - `docker-compose.yml`
  - `docker-compose.neo4j.yml`
  - `scripts/docker.ps1`
- Core pipeline and service code under `rag/`
  - `rag/config.py`
  - `rag/versioning.py`
  - `rag/text_normalize.py`
  - `rag/ingest.py`
  - `rag/clean.py`
  - `rag/chunk.py`
  - `rag/embed.py`
  - `rag/index_faiss.py`
  - `rag/index_bm25.py`
  - `rag/retrieve.py`
  - `rag/rag_answer.py`
  - `rag/service.py`
  - `rag/ask.py`
  - `rag/search.py`
  - `rag/search_faiss.py`
- Parser, retriever, LLM, and agent subsystems
  - `rag/parsers/base.py`
  - `rag/parsers/factory.py`
  - `rag/parsers/llamaparse_backend.py`
  - `rag/parsers/docling_backend.py`
  - `rag/retriever/base.py`
  - `rag/retriever/factory.py`
  - `rag/retriever/faiss_retriever.py`
  - `rag/retriever/bm25_retriever.py`
  - `rag/llm/base.py`
  - `rag/llm/factory.py`
  - `rag/llm/gemini.py`
  - `rag/llm/ollama.py`
  - `rag/lc/state.py`
  - `rag/lc/state_utils.py`
  - `rag/lc/tools.py`
  - `rag/lc/trace.py`
  - `rag/lc/graph.py`
  - `rag/agent/ask.py`
- API and eval
  - `api/main.py`
  - `eval/dataset.py`
  - `eval/metrics.py`
  - `eval/run.py`
  - `eval/day4/questions.jsonl`
- Graph-lite sidecar
  - `rag/graph/build_lite.py`
  - `rag/graph/export_neo4j.py`
  - `rag/graph/helpers.py`
  - `rag/graph/types.py`
- Scripts and tests
  - `scripts/check_artifacts.py`
  - `scripts/docling_preflight.py`
  - `scripts/mini_retrieval_sanity.py`
  - `tests/README.md`
  - `tests/test_api.py`
  - `tests/test_lc_graph.py`
  - `tests/test_retrieve_rrf.py`
  - `tests/test_chunk_determinism.py`
  - `tests/test_docling_preflight.py`
  - `tests/test_integration_smoke.py`
- Existing runtime artifacts
  - `data/processed/manifest.json`
  - `data/processed/pages.jsonl`
  - `data/processed/pages_clean.jsonl`
  - `data/processed/chunks.jsonl`
  - `data/processed/chunk_store.jsonl`
  - `data/processed/faiss.index`
  - `data/processed/bm25.pkl`
  - `data/processed/graph_lite_nodes.jsonl`
  - `data/processed/graph_lite_edges.jsonl`

## Main execution path

Artifact build path:

`rag.ingest -> scripts/clean_pages.py -> scripts/make_chunks.py -> rag.embed -> rag.index_faiss + rag.index_bm25`

Query path:

`CLI/API -> rag.service -> rag.retrieve -> rag.rag_answer`

Agent path:

`CLI/API -> rag.service.ask_agent_question -> rag.lc.graph -> rag.rag_answer`

This is the actual mainline execution path visible in the code and verified against current checked-in artifacts.

## A. Project overview

### What problem this project solves

The repo builds a question-answering system over NIST post-quantum cryptography standards and related NIST publications. The system is designed to answer technical questions with explicit page-grounded citations rather than free-form summaries.

The target use case is not generic web QA. The corpus is a fixed, standards-heavy technical set with algorithms, notation, tables, and document-family-specific terms such as `ML-KEM.KeyGen`, `ML-DSA`, and `SLH-DSA`.

### What kind of RAG system it is

It is a file-backed, hybrid RAG system with:

- PDF parsing into deterministic JSONL page artifacts
- structure-aware chunking
- dense vector retrieval with FAISS
- lexical retrieval with custom BM25
- deterministic query variant expansion
- Reciprocal Rank Fusion
- lightweight reranking
- citation-first answer generation
- a bounded LangGraph agent path

It exposes:

- direct retrieval CLI
- direct QA CLI
- LangGraph agent CLI
- FastAPI endpoints for search and question answering
- an evaluation harness for retrieval quality and optional answer quality

### Current scope of the project

The active corpus in repo contents and planning docs is:

- `NIST.FIPS.203`
- `NIST.FIPS.204`
- `NIST.FIPS.205`
- `NIST.SP.800-227`
- `NIST.IR.8545`
- `NIST.IR.8547.ipd`

These PDFs already exist in `data/raw_pdfs`.

The repo also includes already-generated processed artifacts and indexes for this corpus in `data/processed`.

### What the repo appears to prioritize

Based on code, tests, docs, and artifact layout, the repo currently prioritizes:

- deterministic ingestion and chunk artifacts
- stable chunk IDs and page citation integrity
- retrieval quality over a fixed corpus
- bounded and inspectable agent behavior
- eval-backed iteration
- local deployment and Dockerized usage

Less emphasis is placed on:

- UI/demo polish
- multi-user production concerns
- long-running service architecture
- cloud deployment
- graph-native retrieval

## B. Current end-to-end workflow

### Step-by-step narrative

1. Raw input documents

The raw PDFs live in `data/raw_pdfs`. The repo already includes six NIST documents. `data/raw_pdfs/readme.md` links back to the NIST publication pages.

2. Ingestion and parsing

`rag/ingest.py` is the ingestion entrypoint.

Behavior:

- loads all `*.pdf` in `data/raw_pdfs`
- resolves the parser backend from `rag.parsers.factory.get_parser_backend`
- uses `pypdf.PdfReader` to get the true page count
- parses the PDF with the selected backend
- normalizes parsed page records into a stable page schema
- validates page coverage and page-number correctness
- writes:
  - unified `data/processed/pages.jsonl`
  - per-document `data/processed/<pdf_stem>_parsed.json`
- updates `data/processed/manifest.json`

Backends:

- `llamaparse`
- `docling`

Important note:

- `reports/project_overview.md` describes `llamaparse` as the conservative default
- current code in `rag/config.py` defaults `PARSER_BACKEND` to `docling`
- current checked-in manifest also shows `parser_backend: docling`

That is a real repo/doc drift point, not an inference.

3. Text extraction and parser normalization

The parser layer preserves both `text` and optional `markdown`.

`rag/parsers/base.py` provides `markdown_to_text()`, which removes markdown syntax while preserving useful text. It runs `normalize_identifier_like_spans()` from `rag/text_normalize.py` so parsed identifiers such as `ML-KEM . Decaps` become `ML-KEM.Decaps` without collapsing normal prose such as `U.S. Department`.

`rag/parsers/docling_backend.py` adds significant parser-specific cleanup:

- strips Docling image placeholders and location tags
- injects decoded formula latex into markdown when available
- cleans formula noise
- adaptively batches page conversion
- falls back per-page after batch failures
- cleans CUDA memory between fallback runs

This is the most sophisticated parser path in the repo.

4. Cleaning

`scripts/clean_pages.py` calls `rag.clean.run_clean()` over `pages.jsonl` and writes `pages_clean.jsonl`.

The cleaning pipeline in `rag/clean.py` does:

- unicode normalization
- ligature replacement
- whitespace normalization
- de-hyphenation across line breaks
- standalone page-number removal
- repeated header/footer boilerplate detection per document
- smart prose line joining while preserving:
  - tables
  - pseudocode
  - math-like lines

The cleaning logic is explicitly structure-preserving rather than flattening all line breaks.

5. Chunking

`scripts/make_chunks.py` calls `rag.chunk.run_chunking_per_page()` using:

- `pages_clean.jsonl` as input
- `chunks.jsonl` as output
- config `target_chars=1400`, `overlap_blocks=1`, `min_chars=250`, `max_chars=2200`
- `chunker_version=SETTINGS.CHUNKER_VERSION`

Chunking behavior:

- `v1` is simple block packing over cleaned page text
- `v2` is markdown-aware and structure-aware

`v2` in `rag/chunk.py`:

- tracks heading hierarchy across pages with a document-level heading stack
- detects:
  - headings
  - algorithms
  - tables
  - lists
  - fenced code
  - math blocks
- emits per-chunk metadata:
  - `chunk_id`
  - `doc_id`
  - `page_number`
  - `start_page`
  - `end_page`
  - `block_type`
  - `section_path`
  - `chunker_version`

Important design detail:

- chunks are built per page
- checked-in artifacts use single-page chunks
- page-level citations are therefore preserved directly

6. Embedding generation

`rag/embed.py` reads `chunks.jsonl`.

It:

- loads chunks in file order
- builds embedding text using `build_embedding_text()`
- prepends `doc_id > section_path` when available
- uses `SentenceTransformer("BAAI/bge-base-en-v1.5")`
- encodes with normalized embeddings
- writes:
  - `data/processed/embeddings.npy`
  - `data/processed/chunk_store.jsonl`
  - `data/processed/emb_meta.json`

`chunk_store.jsonl` is important because it aligns `vector_id` with embedding row index and preserves the original text plus chunk metadata for later retrieval.

7. Indexing and storage

Dense index:

- `rag/index_faiss.py`
- loads `embeddings.npy`
- normalizes with `faiss.normalize_L2`
- builds exact `faiss.IndexFlatIP`
- writes `data/processed/faiss.index`

Lexical index:

- `rag/index_bm25.py`
- tokenizes chunk store text with compound-token-preserving regex rules
- expands compounds like `ML-KEM.KeyGen`
- writes `data/processed/bm25.pkl`

Artifact versioning:

- all major stages update `data/processed/manifest.json`
- manifest records stage parameters and artifact hashes

8. Retrieval pipeline

The shared retrieval entrypoint is `rag.retrieve.retrieve()`.

Modes:

- `base`
- `hybrid`

`base`:

- uses one backend from `rag.retriever.factory.get_retriever()`
- current implemented backends are `faiss` and `bm25`

`hybrid`:

- runs both FAISS and BM25
- uses deterministic query variants if enabled
- fuses all rankings with Reciprocal Rank Fusion
- optionally reranks fused candidates

9. Reranking

Reranking is implemented in `rag.retrieve.rerank_fused_hits()`.

It is designed as a do-no-harm promotion stage:

- preserves fused ranking by default
- computes a promotion score from:
  - prior rank
  - BM25 score over candidate text
  - anchor overlap
- only promotes candidates that satisfy mode-specific gates

Mode hints:

- `definition`
- `algorithm`
- `compare`
- `general`

The gates are intentionally conservative.

10. Answer generation

The direct answer path uses `rag.rag_answer.build_cited_answer()`.

Flow:

- select evidence hits
- optionally add neighbor chunks from adjacent `vector_id`s
- assign citation keys `c1`, `c2`, ...
- build prompt with evidence blocks
- call the configured LLM backend
- validate that:
  - inline citation markers are present
  - every sentence has a citation
  - every used citation key exists
- otherwise refuse

Deterministic fallback exists for:

- algorithm step extraction
- comparison answers

11. Citation / grounding logic

Grounding is enforced by code, not only by prompt wording.

Contracts in `rag/types.py` and `rag/rag_answer.py` ensure:

- non-refusal answers must have citations
- inline markers must reference valid keys
- refusal answers must have zero citations
- citation records preserve:
  - `doc_id`
  - `start_page`
  - `end_page`
  - `chunk_id`

12. Evaluation flow

`eval/run.py`:

- loads labeled questions from JSONL
- runs retrieval with stage outputs
- computes:
  - Recall@k
  - MRR@k
  - nDCG@k
- also computes secondary diagnostics:
  - strict page-overlap hit rate
  - doc-only hit rate
  - near-page hit rate
- attributes misses to stages such as:
  - upstream missing
  - outside rerank pool
  - rerank demotion
  - rerank insufficient promotion
- optionally runs `rag.ask --json` for answer metrics
- writes:
  - per-question JSONL
  - summary JSON
  - summary Markdown

13. Serving layer / API / CLI

CLI entrypoints:

- `python -m rag.search_faiss`
- `python -m rag.search`
- `python -m rag.ask`
- `python -m rag.agent.ask`
- `python -m eval.run`

API entrypoints in `api/main.py`:

- `GET /health`
- `GET /search`
- `POST /ask`
- `POST /ask-agent`

14. Dockerized execution path

Two service modes exist:

- `api`
  - serves FastAPI
  - depends on prebuilt processed artifacts
- `allinone`
  - used for ingest/index rebuild work
  - includes ingest dependencies and GPU settings

### ASCII architecture / workflow diagram

```text
data/raw_pdfs/*.pdf
    |
    v
rag.ingest
    |
    +--> data/processed/pages.jsonl
    +--> data/processed/*_parsed.json
    +--> data/processed/manifest.json (stage=ingest)
    |
    v
scripts/clean_pages.py
    |
    v
data/processed/pages_clean.jsonl
    |
    v
scripts/make_chunks.py
    |
    v
data/processed/chunks.jsonl
    |
    +--> rag.embed
    |      |
    |      +--> embeddings.npy
    |      +--> emb_meta.json
    |      +--> chunk_store.jsonl
    |
    +--> rag.index_faiss --> faiss.index
    |
    +--> rag.index_bm25 --> bm25.pkl
    |
    v
rag.retrieve
    |
    +--> FAISS retrieval
    +--> BM25 retrieval
    +--> query variants
    +--> RRF fusion
    +--> optional rerank
    |
    +--> direct path: rag.ask / api:/ask
    |         |
    |         v
    |      rag.rag_answer
    |
    +--> agent path: rag.agent.ask / api:/ask-agent
              |
              v
         rag.lc.graph
              |
              v
         rag.rag_answer
```

### Where artifacts are stored

Raw input:

- `data/raw_pdfs`

Processed pipeline artifacts:

- `data/processed/pages.jsonl`
- `data/processed/pages_clean.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/embeddings.npy`
- `data/processed/emb_meta.json`
- `data/processed/chunk_store.jsonl`
- `data/processed/faiss.index`
- `data/processed/bm25.pkl`
- `data/processed/manifest.json`

Graph-lite sidecar:

- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`
- `data/processed/neo4j_import`

Run outputs:

- `reports/eval`
- `runs/agent`

## C. Code architecture and implementation details

### `rag/config.py`

Role:

- central environment-backed settings dataclass

Important functions / structures:

- `_env_str`, `_env_int`, `_env_bool`, `_env_int_any`
- `Settings`
- `validate_settings()`

How it interacts:

- imported broadly across ingestion, retrieval, QA, service, and agent code
- settings hash is used by manifest/versioning logic

Visible design choices:

- one frozen dataclass for all knobs
- retrieval, answering, LLM, and agent controls are colocated
- config validation lists backends such as `pgvector` and `chroma` even though they are not implemented in retriever factory

### `rag/versioning.py`

Role:

- manifest build/write/load
- artifact hash recording
- retriever compatibility checks

Important functions:

- `compute_settings_hash()`
- `update_manifest()`
- `ensure_manifest_compat()`

How it interacts:

- called by ingest, chunk, embed, and indexing scripts
- called by FAISS and BM25 retrievers to enforce stage compatibility

Visible design choices:

- deterministic JSON serialization
- stage-specific compatibility checking exists
- separate `scripts/check_artifacts.py` compares the full settings hash, which is broader than index-specific compatibility

### `rag/ingest.py`

Role:

- repo ingestion entrypoint

Important functions:

- `_normalize_page_record()`
- `_validate_page_numbers()`
- `parse_and_validate()`
- `main()`

How it interacts:

- uses parser backend factory
- writes page JSONL artifacts
- updates manifest

Visible design choices:

- deterministic ordering by page number
- strict page coverage checks
- writes both unified and per-document artifacts

### `rag/parsers/base.py`

Role:

- parser protocol and markdown-to-text normalization

Important structures:

- `ParsedPage`
- `ParserBackend`
- `markdown_to_text()`

How it interacts:

- shared by both parsing backends

Visible design choices:

- keeps parser interface tiny
- markdown cleanup is deterministic and identifier-aware

### `rag/parsers/llamaparse_backend.py`

Role:

- LlamaParse implementation of parser backend

Important methods:

- `_get_parser()`
- `parse_pdf()`
- `backend_version()`

How it interacts:

- returns page records matching parser contract

Visible design choices:

- preserves markdown as both `text` and `markdown`
- depends on external LlamaParse service and API key

### `rag/parsers/docling_backend.py`

Role:

- Docling implementation of parser backend

Important methods:

- `_build_converter_with_enrichments()`
- `_page_markdown_batch()`
- `_extract_formula_latex()`
- `_inject_formulas()`
- `parse_pdf()`

How it interacts:

- called from ingest
- produces markdown and normalized plain text per page

Visible design choices:

- high investment in parser cleanup for technical PDFs
- adaptive batching for GPU memory pressure
- heavy use of environment tuning flags

### `rag/clean.py`

Role:

- page-level cleaning after parsing

Important classes / functions:

- `CleanConfig`
- `detect_boilerplate()`
- `clean_page_text()`
- `run_clean()`

How it interacts:

- consumes `pages.jsonl`
- writes `pages_clean.jsonl`

Visible design choices:

- preserves structure-aware line breaks
- boilerplate detection is document-specific, not global

### `rag/chunk.py`

Role:

- chunk generation

Important classes:

- `ChunkConfig`
- `StructuredBlock`

Important functions:

- `split_into_blocks()`
- `split_markdown_into_structured_blocks()`
- `pack_structured_blocks_into_chunks()`
- `run_chunking_per_page()`

How it interacts:

- consumes `pages_clean.jsonl`
- writes `chunks.jsonl`
- updates manifest

Visible design choices:

- `v2` uses markdown and heading stack across pages
- algorithm/table/code/math blocks are first-class chunking units
- oversized blocks are split with token overlap
- chunk IDs are deterministic and page-scoped

### `rag/embed.py`

Role:

- embedding generation and chunk store creation

Important functions:

- `build_embedding_text()`
- `build_store_records()`
- `main()`

How it interacts:

- consumes `chunks.jsonl`
- writes embedding and chunk store artifacts

Visible design choices:

- embedding text gets breadcrumb context
- `chunk_store.jsonl` is the retrieval sidecar used by FAISS and answer neighbor expansion

### `rag/index_faiss.py`

Role:

- exact FAISS index build

Important function:

- `main()`

How it interacts:

- consumes `embeddings.npy` and `emb_meta.json`
- writes `faiss.index`

Visible design choices:

- exact search, not IVF/HNSW
- cosine similarity via normalized vectors plus inner product

### `rag/index_bm25.py`

Role:

- BM25 artifact builder

Important functions:

- `tokenize()`
- `build_bm25_artifact()`
- `main()`

How it interacts:

- consumes `chunk_store.jsonl`
- writes `bm25.pkl`

Visible design choices:

- custom tokenizer preserves and expands PQC compound identifiers
- artifact stores postings, idf, doc lengths, and chunk metadata

### `rag/retriever/base.py`

Role:

- swappable retrieval protocol and `ChunkHit` schema

Important structures:

- `ChunkHit`
- `Retriever`

How it interacts:

- base type shared by retriever implementations and higher-level retrieval code

Visible design choices:

- intentionally small interface

### `rag/retriever/faiss_retriever.py`

Role:

- dense retrieval implementation

Important methods:

- `_load_store()`
- `search()`

How it interacts:

- loads `faiss.index`
- loads `chunk_store.jsonl`
- enforces manifest compatibility

Visible design choices:

- dedupes by page span with `max_hits_per_page`
- instantiates `SentenceTransformer` inside retriever

### `rag/retriever/bm25_retriever.py`

Role:

- lexical retrieval implementation

Important methods:

- `search()`
- `score_text()`

How it interacts:

- loads `bm25.pkl`
- used both as retriever and rerank scoring model

Visible design choices:

- BM25 can score arbitrary text for reranking

### `rag/retrieve.py`

Role:

- shared retrieval orchestration

Important functions:

- `query_variants()`
- `rrf_fuse()`
- `rerank_fused_hits()`
- `hybrid_search()`
- `base_search()`
- `retrieve()`
- `retrieve_with_stages()`
- `retrieve_with_stages_and_timing()`
- `retrieve_for_eval_with_stages()`
- `execute_query_plan()`

How it interacts:

- called by direct search, direct QA, eval, and agent tool layer

Visible design choices:

- retrieval is where most ranking logic lives
- graph path and direct path share the same core
- stage-aware outputs exist explicitly for diagnostics

### `rag/rag_answer.py`

Role:

- evidence selection and citation-enforced answer generation

Important functions:

- `select_evidence()`
- `build_context_and_citations()`
- `enforce_inline_citations()`
- `_algorithm_fallback_answer()`
- `_comparison_fallback_answer()`
- `build_cited_answer()`

How it interacts:

- called by direct QA and LangGraph answer node

Visible design choices:

- answer validation is strict and failure-biased
- deterministic fallback exists for realistic failure cases
- neighbor chunks are included by vector-id adjacency rather than by page adjacency

### `rag/llm/*`

Role:

- backend abstraction for answer generation and planner analysis

Files:

- `rag/llm/base.py`
- `rag/llm/factory.py`
- `rag/llm/gemini.py`
- `rag/llm/ollama.py`

Important behavior:

- `get_backend()` chooses `gemini` or `ollama`
- `OllamaBackend` uses official Python SDK client
- `GeminiBackend` uses `google-genai`

Visible design choices:

- stable backend boundary is small
- no structured-output abstraction across the whole repo, only plain text plus local validation

### `rag/service.py`

Role:

- shared orchestration for API and CLI

Important functions:

- `health_status()`
- `search_query()`
- `ask_question()`
- `ask_agent_question()`

How it interacts:

- wraps retrieval, answer generation, and trace summary
- enriches search results with chunk metadata such as `section_path` and `block_type`

Visible design choices:

- service layer centralizes payload shape for direct use paths
- retrieval and generation timings are captured here

### `rag/lc/state.py` and `rag/lc/state_utils.py`

Role:

- structured agent state schema and mutation helpers

Important structures:

- `EvidenceItem`
- `Citation`
- `CompareTopics`
- `QueryAnalysis`
- `Plan`
- `AgentState`

Important helpers:

- `init_state()`
- `set_query_analysis()`
- `set_plan()`
- `set_evidence()`
- `set_answer()`
- `set_final_answer()`

Visible design choices:

- state is dict-like for LangGraph, but key payloads are dataclasses
- agent trace writing is explicit and centralized

### `rag/lc/tools.py`

Role:

- LangChain tool adapter layer

Important tools:

- `retrieve`
- `resolve_definition`
- `compare`
- `summarize`

How it interacts:

- adapts retrieval functions into tool-style JSON payloads
- supports planner query arguments like `sparse_query`, `dense_query`, `subqueries`, and `doc_ids`

Visible design choices:

- tools are adapter-oriented and retrieval-backend-agnostic
- direct chunk metadata lookup exists for page-range summary tool

### `rag/lc/graph.py`

Role:

- bounded LangGraph controller

Important functions:

- `_build_query_analysis()`
- `node_analyze_query()`
- `node_route()`
- `node_retrieve()`
- `node_assess_evidence()`
- `node_refine_query()`
- `node_answer()`
- `node_verify_or_refuse()`
- `build_graph()`
- `run_agent()`

How it interacts:

- consumes state helpers and tool layer
- answer node delegates to `rag.rag_answer`

Visible design choices:

- explicit budgets for steps, tool calls, and retrieval rounds
- query analysis has deterministic fallback plus bounded LLM JSON planning
- graph path disables retrieval-side automatic variant expansion

Important observed limitation:

- `node_route()` currently always emits `Plan(action="retrieve")`
- the specialized tool actions exist, but the default route path does not currently select them

### `rag/lc/trace.py`

Role:

- trace normalization and summary generation

Important functions:

- `summarize_trace()`
- `write_trace()`

How it interacts:

- used by `rag.agent.ask`
- used by `rag.service.ask_agent_question`

Visible design choices:

- human-readable trace summaries and raw state dumps are both supported

### `api/main.py`

Role:

- FastAPI app

Endpoints:

- `GET /health`
- `GET /search`
- `POST /ask`
- `POST /ask-agent`

How it interacts:

- thin wrapper over `rag.service`

Visible design choices:

- stable direct QA endpoint and separate explicit agent endpoint
- no streaming or auth layer

### `eval/*`

Role:

- eval dataset loading, scoring, and report generation

Files:

- `eval/dataset.py`
- `eval/metrics.py`
- `eval/run.py`

Visible design choices:

- deterministic JSONL dataset contract
- retrieval metrics are the primary score surface
- answer metrics are explicitly marked as model-dependent

### `rag/graph/*`

Role:

- graph-lite sidecar over documents, sections, algorithms, and seeded terms

Files:

- `rag/graph/build_lite.py`
- `rag/graph/export_neo4j.py`
- `rag/graph/helpers.py`
- `rag/graph/types.py`

Visible design choices:

- graph is offline and artifact-driven
- seeded terms, section paths, and algorithm headers are used to build nodes/edges
- not integrated into the live retriever

## D. Retrieval methodology

### Retrieval backends used

Implemented backends:

- dense: FAISS via `rag/retriever/faiss_retriever.py`
- lexical: BM25 via `rag/retriever/bm25_retriever.py`

Config validation in `rag/config.py` also mentions:

- `pgvector`
- `chroma`

But these are not implemented in `rag/retriever/factory.py`. That is directly observable code drift.

### Lexical / dense / hybrid retrieval details

Dense leg:

- embedding model: `BAAI/bge-base-en-v1.5`
- search over exact `IndexFlatIP`
- `SentenceTransformer.encode(... normalize_embeddings=True)`
- page-span dedupe in FAISS retriever

Lexical leg:

- custom BM25 artifact over chunk store text
- tokenization preserves compounds like `ML-KEM.KeyGen`
- compounds are also expanded into parts so lexical matching can recover split references

Hybrid:

- runs both dense and lexical searches
- collects results for one or more deterministic query variants
- fuses all rankings with RRF

### Rank fusion or score combination

Rank fusion:

- `rrf_fuse(rankings, top_k, k0)`
- score rule: `1 / (k0 + rank)`

Reranking:

- rerank does not replace the fused ranking wholesale
- it computes mode-aware promotion scores
- it only promotes hits that clear conservative gates

Mode weights in current code:

- `definition`: `(0.35, 0.45, 0.20)`
- `algorithm`: `(0.45, 0.20, 0.35)`
- `compare`: `(0.60, 0.30, 0.10)`
- `general`: `(0.70, 0.20, 0.10)`

These correspond to:

- prior rank weight
- BM25-on-candidate-text weight
- anchor overlap weight

### Query transformation / query variants / mode hints

Implemented deterministic query transformation:

- definition templates:
  - `definition of <term>`
  - `<term> stands for`
  - `<term> notation`
- compare templates:
  - per-topic use-case queries
  - `<a> vs <b>`
- domain bias:
  - PQC standards expansion for broad corpus-level questions
- exact technical rewrites:
  - `ML-KEM.KeyGen`
  - `ML-DSA.Sign`
  - `ML-KEM.Decaps`
- algorithm-specific variants:
  - `Algorithm N`
  - `Algorithm N ML-KEM.KeyGen`

Mode hint behavior:

- direct retrieval can infer mode from the raw query
- graph retrieval supplies analyzed mode hint and turns off retrieval-side auto expansion

### Reranking logic

`rerank_fused_hits()` computes:

- prior normalized score
- BM25 score of query against each candidate text
- anchor overlap count

It then:

- applies mode-specific weights
- checks mode-specific promotion gates
- promotes a limited number of strong candidates
- preserves existing order when no candidate clearly earns promotion

This is explicitly conservative reranking rather than aggressive reordering.

### Thresholds, filters, heuristics, and fallback behavior

Important retrieval knobs:

- `TOP_K`
- `RETRIEVAL_CANDIDATE_MULTIPLIER`
- `RETRIEVAL_RRF_K0`
- `RETRIEVAL_RERANK_POOL`
- `RETRIEVAL_QUERY_FUSION`
- `RETRIEVAL_ENABLE_RERANK`

Filtering:

- `doc_ids` filtering is supported
- graph path uses explicit doc allowlists and planner-produced doc filters

Fallback behavior:

- if rerank does not find promotable candidates, fused order is preserved
- if graph retrieval is insufficient, the graph can refine and retry

### How the system tries to improve precision vs recall

Recall improvements:

- multiple query variants
- hybrid dense plus lexical retrieval
- candidate expansion before fusion
- compare-mode subqueries in planner path

Precision improvements:

- page-span dedupe on FAISS results
- conservative rerank promotion gates
- explicit doc filtering
- strict answer-side citation validation

## E. Models, tools, and technologies used

Only tools actually supported in repo code are listed below.

### Python packages and frameworks

- `fastapi`
  - API serving
- `uvicorn`
  - ASGI server
- `numpy`
  - embeddings and array handling
- `python-dotenv`
  - `.env` loading in selected entrypoints
- `faiss-gpu-cu12`
  - dense vector index
- `sentence-transformers`
  - embedding model runtime
- `tqdm`
  - progress bars during embedding
- `google-genai`
  - Gemini API client
- `langchain-core`
  - tool definitions
- `langgraph`
  - bounded agent graph
- `ollama`
  - Ollama Python SDK
- `docling`
  - PDF parsing backend
- `llama-parse`
  - PDF parsing backend
- `pypdf`
  - PDF page counting
- `pytest`
  - test runner

### Models

Embedding model:

- `BAAI/bge-base-en-v1.5`

LLM backends:

- Ollama-backed local model
  - default in `.env.example`: `qwen3:8B`
- Gemini backend
  - fallback configured via `GEMINI_API_KEY` and `GEMINI_MODEL`

Important observed issue:

- `GeminiBackend.get_model_name()` prioritizes `SETTINGS.LLM_MODEL`
- default `LLM_MODEL` is `qwen3:8B`
- so `LLM_BACKEND=gemini` requires overriding `LLM_MODEL` too, otherwise the configured Gemini backend will inherit a non-Gemini model name

### Storage / retrieval technologies

- file-backed JSONL artifacts
- NumPy embeddings
- FAISS exact index
- custom BM25 pickle

There is no live database-backed vector store in the current implementation.

### Serving / API / CLI tooling

- FastAPI app in `api/main.py`
- CLI entrypoints under `rag.*`
- PowerShell Docker task runner in `scripts/docker.ps1`

### LangChain / LangGraph

- LangChain Core tools for tool wrappers
- LangGraph for bounded state-machine-style orchestration

### Docker / Compose

- single multistage `Dockerfile`
- `docker-compose.yml` for API and all-in-one pipeline service
- `docker-compose.neo4j.yml` for optional Neo4j

### Testing / eval tooling

- `pytest`
- custom retrieval/answer eval harness in `eval/`

## F. Methodology and project mechanisms

### How chunking is designed

Chunking is designed around standards-style technical PDF structure, not generic paragraph splitting.

Key characteristics:

- preserves page boundaries
- extracts structured blocks from markdown
- retains algorithm, table, and math sections as distinct chunk types
- persists `section_path`
- splits oversized blocks with overlap rather than flattening them badly

This is one of the main technical differentiators of the repo.

### How evidence is selected

Evidence selection in `rag/rag_answer.py` is separate from retrieval ranking:

- dedupe by `chunk_id`
- sort deterministically by score and stable tie-breakers
- take up to `ASK_MAX_CONTEXT_CHUNKS`
- optionally add neighboring chunks from adjacent `vector_id`s within the same document
- enforce total context character budget

This means the answer model does not necessarily see every retrieved hit, only the budgeted evidence set.

### How answer synthesis is done

Answer synthesis uses:

- evidence blocks headed with citation keys
- a prompt that requires every sentence to include inline citations
- backend abstraction for Gemini or Ollama
- strict local validation after generation

The answer model is treated as a constrained generator rather than an authority.

### How citations are attached

Citations are attached by building a deterministic key map:

- `c1`, `c2`, `c3`, ...

Each citation records:

- `doc_id`
- `start_page`
- `end_page`
- `chunk_id`

The answer text must reference only these keys.

### How refusal works

Refusal is central to the current methodology.

Direct path refusal occurs when:

- evidence count is below minimum threshold
- generated text is empty
- citations are missing
- inline citation contract fails

Agent path refusal occurs when:

- evidence is insufficient after bounded retrieval/refinement
- answer draft is empty
- citations are missing after answer step

### How trace/logging/artifacts are produced

Pipeline artifacts:

- manifest with stage hashes and artifact hashes

Agent artifacts:

- trace summaries
- raw state dumps
- saved under `runs/agent`

Eval artifacts:

- per-question JSONL
- summary JSON
- summary Markdown

### How evaluation is structured

Evaluation is retrieval-first.

For each labeled question:

- retrieve final hits
- inspect pre-rerank and post-rerank stage outputs
- compute strict metrics
- compute relaxed diagnostics
- attribute miss cause

Optional answer evaluation:

- calls `rag.ask --json`
- scores citation presence and refusal correctness

### How workflow state moves in the agent

Current LangGraph state progression:

- `analyze_query`
- `route`
- `retrieve`
- `assess_evidence`
- optional `refine_query`
- `answer`
- `verify_or_refuse`

State carries:

- analyzed query representation
- doc filters
- evidence
- citations
- refusal reasons
- timing and trace events

## G. What is already implemented vs not yet implemented

### Already implemented and visible in repo

- deterministic ingestion into page JSONL
- parser abstraction with Docling and LlamaParse
- structure-aware chunking `v2`
- BGE embeddings
- exact FAISS index
- custom BM25 lexical retrieval
- hybrid retrieval with RRF
- conservative reranking
- direct QA with strict citation validation
- FastAPI with `/health`, `/search`, `/ask`, `/ask-agent`
- bounded LangGraph controller
- retrieval evaluation harness
- graph-lite artifact build
- Docker and Compose setup

### Partial / in-progress areas

- graph-lite is built but not integrated into retrieval or answering
- LangGraph tools for `compare`, `resolve_definition`, and `summarize` exist, but the default route node currently still chooses only `retrieve`
- answer-side robustness exists via deterministic fallbacks, but there is no generalized repair pass
- Neo4j export exists as an offline export path only

### Features that seem planned but not yet implemented

- actual retriever implementations for `pgvector` and `chroma`
- a graph-aware retrieval path
- a full answer-side citation repair stage
- more explicit query-analysis routing into specialized actions
- any UI beyond CLI and HTTP API

These are based on code paths and config surface, not generic roadmap assumptions.

### Limitations / technical debt / likely weak points

- config and docs drift:
  - docs describe conservative `llamaparse` default
  - current code and manifest default to `docling`
- environment loading inconsistency:
  - some paths call `load_dotenv()`
  - many direct runtime paths do not
- `scripts/check_artifacts.py` compares the full settings hash, which can trigger rebuild warnings from irrelevant runtime changes
- `scripts/docling_preflight.py` and its tests are out of sync
- retriever/model objects are recreated per request rather than cached process-wide
- graph-lite artifacts exist, but the main user path does not benefit from them

## H. Dockerized usage guide

### What Docker files exist and their purpose

- `Dockerfile`
  - multistage build for two final images
- `docker-compose.yml`
  - main compose file for API and all-in-one pipeline service
- `docker-compose.neo4j.yml`
  - optional Neo4j service for graph import work
- `scripts/docker.ps1`
  - PowerShell wrapper for common compose tasks

### Image targets

In `Dockerfile`:

- `api-cuda`
  - serving image
  - installs `core`, `retrieval`, and `agent` extras
- `allinone-cuda`
  - rebuild image
  - installs `ingest` extras on top of API runtime

Both are based on:

- `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`

### Build steps

API image:

```bash
cp .env.example .env
docker compose build api
```

All-in-one image:

```bash
docker compose --profile allinone build allinone
```

### Run steps

Run API:

```bash
docker compose up --build api
```

Run all-in-one service interactively for pipeline commands:

```bash
docker compose --profile allinone run --rm allinone python -m rag.ingest
```

### Required environment variables

From `.env.example`, the important ones are:

LLM/runtime:

- `LLM_BACKEND`
- `LLM_MODEL`
- `LLM_BASE_URL`
- `LLM_TIMEOUT_S`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Ingestion/chunking:

- `PARSER_BACKEND`
- `PARSER_STRICT_PAGE_MATCH`
- `CHUNKER_VERSION`
- `DOCLING_*`

Retrieval:

- `VECTOR_BACKEND`
- `RETRIEVAL_MODE`
- `TOP_K`
- `RETRIEVAL_QUERY_FUSION`
- `RETRIEVAL_RRF_K0`
- `RETRIEVAL_CANDIDATE_MULTIPLIER`
- `RETRIEVAL_ENABLE_RERANK`
- `RETRIEVAL_RERANK_POOL`

Answering/agent:

- `ASK_*`
- `AGENT_*`
- `LLM_TEMPERATURE`

### Mounted volumes / bind mounts

`api` service mounts:

- `./data/processed:/app/data/processed`
- `./reports:/app/reports`
- `./runs:/app/runs`
- `hf_cache:/cache/huggingface`

`allinone` service mounts:

- `./data/raw_pdfs:/app/data/raw_pdfs`
- `./data/processed:/app/data/processed`
- `./data/debug:/app/data/debug`
- `./reports:/app/reports`
- `./runs:/app/runs`
- `hf_cache:/cache/huggingface`

### How the container accesses models, caches, raw data, processed artifacts, and indexes

Models:

- by default, the app calls a host-side Ollama server at `http://host.docker.internal:11434`
- Gemini can be used if env vars are set

Caches:

- Hugging Face cache goes to `/cache/huggingface` via named volume

Raw data:

- mounted only into `allinone`

Processed artifacts and indexes:

- mounted from host `data/processed` into both services

This means processed data is not baked into image layers.

### How to run the main pipeline inside Docker

```bash
docker compose --profile allinone run --rm allinone python -m rag.ingest
docker compose --profile allinone run --rm allinone python scripts/clean_pages.py
docker compose --profile allinone run --rm allinone python scripts/make_chunks.py
docker compose --profile allinone run --rm allinone python -m rag.embed
docker compose --profile allinone run --rm allinone python -m rag.index_faiss
docker compose --profile allinone run --rm allinone python -m rag.index_bm25
```

### How to run the API / app inside Docker

```bash
docker compose up --build api
```

Available endpoints:

- `GET /health`
- `GET /search`
- `POST /ask`
- `POST /ask-agent`

### GPU assumptions

- `allinone` requests `gpus: all`
- Docling defaults in `.env.example` target CUDA
- API image is also CUDA-based, though the compose service itself does not explicitly request GPU

### Common failure points or setup gotchas visible from the repo

- `.env` is not checked in and currently missing in this repo state
- Docker assumes host-side Ollama unless Gemini is configured
- `api` service does not mount `data/raw_pdfs`, so it cannot perform ingest work
- `uvicorn` runs with `--reload`, which is development-oriented
- `scripts/check_artifacts.py` may report rebuild needed even if core index artifacts are still compatible

### Multiple supported Docker workflows

1. API-only workflow

- use `api`
- rely on existing `data/processed`
- suitable for normal search/QA/demo use

2. Pipeline rebuild workflow

- use `allinone`
- run ingest/clean/chunk/embed/index commands manually
- suitable when changing parsing, chunking, or retrieval artifacts

3. Optional graph export workflow

- generate graph-lite artifacts
- export Neo4j CSV/import scripts
- start Neo4j with `docker-compose.neo4j.yml` if needed

## I. How to use the project

### Local setup

The repo has `pyproject.toml` extras. A typical local install is:

```bash
pip install -e ".[core,retrieval,agent,ingest,dev]"
```

The README references a conda env named `eleven`, but the available local envs at inspection time were:

- `base`
- `pyt`
- `tsf`

The runnable environment used for verification here was `pyt`.

### How to run ingestion

```bash
python -m rag.ingest
```

This writes `data/processed/pages.jsonl` and per-doc parsed JSON files.

### How to clean parsed pages

```bash
python scripts/clean_pages.py
```

### How to build chunks

```bash
python scripts/make_chunks.py
```

### How to build indexes

```bash
python -m rag.embed
python -m rag.index_faiss
python -m rag.index_bm25
```

### How to query the system

Vector-only quick search:

```bash
python -m rag.search_faiss "ML-KEM key generation"
```

Shared retrieval CLI:

```bash
python -m rag.search "Algorithm 19" --mode hybrid --k 5
python -m rag.retrieve "ML-KEM.KeyGen" --mode hybrid --k 5
```

Direct QA:

```bash
python -m rag.ask "What is ML-KEM?"
python -m rag.ask "What is ML-KEM?" --json --no-evidence
```

Agent QA:

```bash
python -m rag.agent.ask "What is ML-KEM?"
python -m rag.agent.ask "What is ML-KEM?" --json
```

### How to run evaluations

```bash
python -m eval.run
python -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval
```

Optional answer-eval path:

```bash
python -m eval.run --with-answers
```

### How to inspect outputs / artifacts

Core artifacts:

- `data/processed/manifest.json`
- `data/processed/pages.jsonl`
- `data/processed/pages_clean.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/chunk_store.jsonl`
- `data/processed/faiss.index`
- `data/processed/bm25.pkl`

Eval outputs:

- `reports/eval`

Agent traces:

- `runs/agent`

### How to use the Docker version

API:

```bash
docker compose up --build api
```

Pipeline:

```bash
docker compose --profile allinone run --rm allinone python -m rag.ingest
docker compose --profile allinone run --rm allinone python scripts/clean_pages.py
docker compose --profile allinone run --rm allinone python scripts/make_chunks.py
docker compose --profile allinone run --rm allinone python -m rag.embed
docker compose --profile allinone run --rm allinone python -m rag.index_faiss
docker compose --profile allinone run --rm allinone python -m rag.index_bm25
```

### What commands a new engineer would likely run first

Because the repo already contains built artifacts, the fastest first commands are:

```bash
python -m rag.search_faiss "ML-KEM key generation"
python -m rag.search "Algorithm 19" --mode hybrid --k 5
python -m rag.ask "What is ML-KEM?" --json --no-evidence
python -m eval.run
```

These give immediate signal before touching the pipeline build path.

## J. Concise engineering assessment

### Strongest technical aspects of the repo

- deterministic, file-backed artifact pipeline with manifest tracking
- strong structure-aware chunking for technical PDFs
- credible hybrid retrieval implementation with diagnostics
- strict citation enforcement in the answer layer
- bounded, inspectable LangGraph controller with trace summaries
- already runnable local and Dockerized usage paths

### Most important current limitations

- config and documentation drift around parser defaults and runtime assumptions
- incomplete backend surface compared with advertised config values
- inconsistent environment loading across entrypoints
- graph-lite is not integrated into live retrieval
- some scripts and tests are stale relative to current code
- retriever/model objects are recreated per request, increasing latency

### Top 5 highest-value next improvements

1. Unify environment loading and runtime configuration semantics across CLI, service, API, and eval paths.
2. Cache FAISS retriever, BM25 artifact, and embedding model at process scope in the serving path.
3. Fix repo drift:
   - parser default docs vs code
   - `scripts/docling_preflight.py` vs its tests
   - Gemini model-selection behavior
4. Either implement or remove placeholder backends and action surfaces that are not truly supported.
5. Decide whether graph-lite is only a side artifact or a real retrieval feature, then wire it into at least one query class if it is intended to matter.

## Runtime verification notes

The following paths were executed successfully in the local `pyt` environment on 2026-03-12:

- `python -m rag.search_faiss "ML-KEM key generation"`
- `python -m rag.search "Algorithm 19" --mode hybrid --k 5`
- `python -m eval.run --dataset eval/day4/questions.jsonl --outdir /tmp/nist-pqc-eval-smoke`
- `python -m rag.ask "What is ML-KEM?" --json --no-evidence`
- `python -m rag.agent.ask "What is ML-KEM?" --json --no-trace`
- `python -c "from rag.service import health_status; print(...)"` via the `pyt` env

Observed runtime facts from those checks:

- `health_status()` returned `status: ok`, `llm_backend: ollama`, `retrieval_ready: true`
- current direct QA path can return cited answers
- current agent path runs with analyzed query state and trace events
- current eval smoke run produced:
  - `Recall@8: 0.6429`
  - `MRR@8: 0.3318`
  - `nDCG@8: 0.3969`

Important live-quality observation:

- direct QA for broad queries can retrieve secondary supporting docs rather than only the expected primary standard unless explicit doc scoping is imposed
- the agent path improves that on definition-style questions because analyzed `doc_ids` can narrow retrieval

## Test verification notes

Targeted tests executed in the `pyt` environment:

```bash
python -m pytest -q tests/test_api.py tests/test_retrieve_rrf.py tests/test_eval_run.py tests/test_lc_graph.py tests/test_docling_preflight.py
```

Observed result:

- `40 passed`
- `2 failed`

The failing tests were both in `tests/test_docling_preflight.py`.

Grounded cause:

- `scripts/docling_preflight.py` currently hardcodes `PAGE = 20`
- the tests expect page count `1` in one case
- the tests also expect an unsupported `--pages` flag

This is a direct script/test mismatch, not a flaky runtime issue.

## Quick start for a new engineer

1. Start with existing artifacts, not a rebuild.

```bash
python -m rag.search_faiss "ML-KEM key generation"
python -m rag.ask "What is ML-KEM?" --json --no-evidence
```

2. Run retrieval eval to establish the current baseline.

```bash
python -m eval.run
```

3. Inspect the live artifacts before changing code.

- `data/processed/manifest.json`
- `data/processed/chunks.jsonl`
- `reports/eval`
- `runs/agent`

4. Use the API path for service-level work.

```bash
docker compose up --build api
```

5. Only use the rebuild path when touching ingest/chunk/index behavior.

```bash
docker compose --profile allinone run --rm allinone python -m rag.ingest
```

## Gaps / risks / next steps

- The repo is already functional and demoable, but there is visible operational drift between docs, config defaults, tests, and live runtime behavior.
- The highest near-term risk is misunderstanding which configuration actually governs runtime:
  - parser defaults
  - `.env` loading
  - Gemini model selection
  - manifest/rebuild checks
- The strongest engineering payoff is now in tightening runtime contracts and ergonomics rather than adding more retrieval complexity.
- If the project is being presented externally, the most important cleanup items are:
  - fix Docling preflight drift
  - align docs with current defaults
  - cache retriever/model objects in service
  - clarify supported backends and graph scope
