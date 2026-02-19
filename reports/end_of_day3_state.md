# Current-State Technical Description (for Peer Critique)

## Executive summary
This project is currently a local, citation-grounded PQC RAG system over 6 NIST PDFs. It includes deterministic ingestion and chunk artifacts, hybrid retrieval (`FAISS + BM25 + RRF + rerank`), strict citation-validated answer generation, and a bounded LangGraph agent wrapper.

## 1) Core structure
- Source-of-truth scope and contracts: `reports/project_overview.md`.
- Core pipeline modules: `rag/ingest.py`, `rag/clean.py`, `rag/chunk.py`, `rag/embed.py`, `rag/index_faiss.py`, `rag/index_bm25.py`, `rag/retrieve.py`, `rag/rag_answer.py`, `rag/ask.py`.
- Retrieval abstraction: `rag/retriever/base.py` with backends `rag/retriever/faiss_retriever.py`, `rag/retriever/bm25_retriever.py`, selected via `rag/retriever/factory.py`.
- Agent layer: `rag/lc/graph.py`, `rag/lc/tools.py`, `rag/lc/state.py`, `rag/lc/state_utils.py`, `rag/lc/trace.py`, plus CLI `rag/agent/ask.py`.
- Utility and reporting scripts: `scripts/` and `eval/day2/`.

## 2) Current data/artifact state
- Raw docs in scope: `data/raw_pdfs/`
  - `NIST.FIPS.203.pdf`, `NIST.FIPS.204.pdf`, `NIST.FIPS.205.pdf`
  - `NIST.IR.8545.pdf`, `NIST.IR.8547.ipd.pdf`, `NIST.SP.800-227.pdf`
- Processed artifacts:
  - `data/processed/pages.jsonl`: 309 rows
  - `data/processed/pages_clean.jsonl`: 309 rows
  - `data/processed/chunks.jsonl`: 552 rows
  - `data/processed/chunk_store.jsonl`: 552 rows (`vector_id` 0..551)
  - `data/processed/emb_meta.json`: `BAAI/bge-base-en-v1.5`, dim `768`, normalized embeddings
- Chunk distribution by `doc_id`:
  - `NIST.FIPS.203`: 103
  - `NIST.FIPS.204`: 122
  - `NIST.FIPS.205`: 108
  - `NIST.IR.8545`: 56
  - `NIST.IR.8547.ipd`: 48
  - `NIST.SP.800-227`: 115
- Page-level citation fields are preserved on chunk records (`doc_id`, `start_page`, `end_page`).

## 3) Implemented features
### Ingestion
- `rag/ingest.py` parses each PDF with LlamaParse, validates parsed page count against `pypdf`, writes:
  - per-PDF parsed debug JSON (`*_parsed.json`)
  - unified `pages.jsonl`
- Ordering is deterministic via sorted PDF filenames.

### Cleaning
- `rag/clean.py` implements:
  - Unicode normalization, whitespace normalization, de-hyphenation
  - standalone page-number stripping
  - per-document boilerplate (header/footer) detection by frequency threshold
  - structure-preserving smart line joins (keeps table/code/math lines verbatim)
- Output: `pages_clean.jsonl`.

### Chunking
- `rag/chunk.py`:
  - splits pages into blank-line-delimited blocks
  - classifies blocks as verbatim-ish vs prose
  - greedy packs blocks into chunks with overlap
- Chunk ID contract is deterministic (`{doc_id}::p{page}::c{idx}`), and chunks include page span metadata.

### Embeddings and indexes
- `rag/embed.py` generates normalized embeddings and writes:
  - `embeddings.npy`
  - `chunk_store.jsonl` with `vector_id -> chunk metadata/text`
  - `emb_meta.json`
- `rag/index_faiss.py` builds `IndexFlatIP` on normalized vectors.
- `rag/index_bm25.py` builds a deterministic BM25 artifact (`bm25.pkl`) with technical-token-aware tokenization.

### Retrieval
- `rag/retrieve.py` supports:
  - `base` mode: single backend + optional query-fusion RRF
  - `hybrid` mode: FAISS + BM25 + RRF + optional rerank
- Deterministic query variants are generated for standards-specific patterns.
- Stable tie-breaking is used in fusion and rerank paths.

### Answer generation
- `rag/rag_answer.py` enforces citation-first generation:
  - evidence dedup + stable ordering + budgeting
  - strict inline marker validation (`[c#]`)
  - refusal contract when unsupported: exact `not found in provided docs`
- Deterministic fallback exists for algorithm-step questions when evidence contains numbered steps.

### Agent orchestration
- `rag/lc/graph.py` builds a bounded state graph:
  - `route -> do_tool -> answer -> verify_or_refuse -> END`
- Heuristic planner routes to `retrieve`, `resolve_definition`, `compare`, or `summarize`.
- Guardrails: `MAX_STEPS`, `MAX_TOOL_CALLS`.
- Run traces are serialized by `rag/lc/trace.py` and produced by `rag/agent/ask.py`.

## 4) Algorithms and decision process
### Query expansion (`rag/retrieve.py`)
- Rule-based deterministic rewrites for technical queries:
  - preserves compound tokens
  - adds known operation aliases (`ML-KEM.Decaps`, `ML-DSA.Sign`, etc.)
  - boosts algorithm-number forms (`Algorithm N` variants)

### Fusion (`rag/retrieve.py`)
- Reciprocal Rank Fusion:
  - score accumulation `1 / (k0 + rank)` per ranking
  - stable ordering by `(-rrf, doc_id, start_page, chunk_id)`

### Rerank (`rag/retrieve.py`)
- Lightweight rerank:
  - first criterion: exact technical token presence in hit text
  - second criterion: BM25 lexical score (`score_text`)
  - stable deterministic ties

### Evidence selection (`rag/rag_answer.py`)
- Dedup by `chunk_id`.
- Stable sort by `(-score, doc_id, start_page, end_page, chunk_id)`.
- Optional same-doc neighbor expansion using `vector_id` adjacency.
- Context limits: max chunks + max total chars.

### Validation (`rag/types.py`, `rag/rag_answer.py`)
- Refusal must have zero citations.
- Non-refusal must have citations when required.
- Every sentence must include a valid citation marker in strict mode.

## 5) Script functionality
- `scripts/clean_pages.py`
  - Runs `run_clean(...)` from `pages.jsonl` to `pages_clean.jsonl`.
- `scripts/make_chunks.py`
  - Runs `run_chunking_per_page(...)` from cleaned pages to `chunks.jsonl`.
- `scripts/mini_retrieval_sanity.py`
  - Compares base vs hybrid retrieval on fixed PQC queries.
  - Writes `reports/mini_retrieval_sanity.json` and `reports/mini_retrieval_sanity.md`.
- `eval/day2/run_baseline.py`
  - Runs `rag.ask` over `eval/day2/questions.txt`.
  - Writes JSONL outputs plus environment snapshots under `runs/day2_baseline/...`.

## 6) Tests and what they validate
- Retrieval/fusion/determinism:
  - `tests/test_query_fusion.py`
  - `tests/test_retrieve_rrf.py`
  - `tests/test_retrieve_determinism.py`
- BM25 and tokenization:
  - `tests/test_bm25_index.py`
- Answer and citation contracts:
  - `tests/test_rag_answer.py`
  - `tests/test_rag_answer_deterministic.py`
  - `tests/test_types.py`
- LangGraph tooling and state flow:
  - `tests/test_lc_tools.py`
  - `tests/test_lc_graph.py`
- Optional integration smoke:
  - `tests/test_integration_smoke.py` (env-gated)

## 7) Critique targets (high value)
1. **Doc/test vs implementation drift**
   - `tests/test_retrieve_eval_api.py` imports `retrieve_for_eval` from `rag/retrieve.py`, but this function is not currently defined.
   - `README.md` references `eval/day2/run.py` and `eval/day2/ablate.py`, but those files are not present.
2. **Agent is bounded but not iterative**
   - The current graph is a single tool pass then answer/verify. There is no retrieve-reassess loop edge yet.
3. **Tool hints/filters are mostly non-operative**
   - `rag/lc/tools.py` passes `mode_hint` and `filters`, but `rag/retrieve.py::retrieve` does not consume those parameters.
4. **Chunk-size behavior drift**
   - Chunking is configured with target size controls, but current artifacts include many large chunks (for example, >2200 chars), which may hurt retrieval precision.
5. **Packaging completeness**
   - Some imported runtime packages are not listed in `pyproject.toml`/`requirements.txt` (`faiss`, `numpy`, `sentence-transformers`, `pypdf`, `tqdm`).
6. **Eval maturity gap**
   - Current eval is baseline/sanity logging rather than a full metric harness (Recall@k, MRR, nDCG, citation coverage/faithfulness) as envisioned in project overview milestones.

## 8) Suggested critique questions for peers
1. Are chunking decisions giving enough retrieval precision for algorithm-heavy pages?
2. Should the agent add a second retrieval pass under explicit evidence insufficiency conditions?
3. What minimal eval harness should be implemented next to measure improvements rigorously?
4. Which retrieval knobs (`k0`, candidate multiplier, rerank pool, query fusion rules) are currently overfitting fixed sanity queries?
5. Where should strict citation policy be relaxed, if at all, for usability without reducing trust?
