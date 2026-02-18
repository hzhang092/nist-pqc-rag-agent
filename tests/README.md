# Tests Overview

This folder contains unit tests for retrieval, ranking, and answer-validation behavior, plus a couple of lightweight inspection scripts.

## How to run

- Run all tests: `pytest tests`
- Run one file: `pytest tests/test_retrieve_rrf.py`

## Test files

- `test_bm25_index.py`
  - Verifies BM25 tokenization preserves PQC technical tokens (for example `ML-KEM.KeyGen`) and that BM25 artifact generation is deterministic/stable for identical input.

- `test_query_fusion.py`
  - Verifies `query_variants` behavior: domain-specific expansions, generalized rewrite rules, empty-input handling, and deduplication.

- `test_rag_answer.py`
  - Verifies citation-first answer construction in `rag.rag_answer`: refusal conditions, citation-key validation, citation extraction, and deterministic citation-key assignment under reordered input hits.

- `test_retrieve_determinism.py`
  - Verifies deterministic retrieval pipeline behavior across query variant generation, RRF tie handling, and hybrid search output stability when backend result order changes.

- `test_retrieve_eval_api.py`
  - Verifies `retrieve_for_eval` output schema and parameter mapping into `hybrid_search` (eval knobs like `k0`, fusion toggle, rerank toggle).

- `test_retrieve_rrf.py`
  - Verifies retrieval fusion internals: Reciprocal Rank Fusion ranking behavior, stable tie-break ordering, and reranking preference for exact technical-token matches.

- `test_types.py`
  - Script-like sanity check for `rag.types` validation logic: refusal without citations is allowed, non-refusal without citations should raise.

## Utility scripts in this folder

- `check_clean.py`
  - Quick manual spot-check script for sampled rows in `data/processed/pages_clean.jsonl`.

- `count_chunks.py`
  - Quick script that prints the line count of `data/processed/chunks.jsonl`.

- `gemini-3-flash-preview-record.txt`
  - Saved model output/evidence transcript for manual reference; not a test.
