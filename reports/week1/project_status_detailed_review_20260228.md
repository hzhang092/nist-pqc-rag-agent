# Project Current Status Report for Detailed Peer Review (Code-Verified)

Date anchor: 2026-03-01 (verification refresh)  
Repository: `D:/Waterloo/terms/MDSAI/1B/projects/nist-pqc-rag-agent`  
Conflict policy: current code/artifacts are authoritative; older day notes are historical context only.

## Executive Summary (1-minute read)

What is working now:

- end-to-end PQC RAG pipeline is implemented with deterministic artifacts
- hybrid retrieval + citation-first answer generation are implemented
- bounded LangGraph controller is implemented and test-covered
- evaluation harness and baseline outputs exist

What is currently verified on this machine:

- `faiss` imports successfully in `eleven`
- full test suite runs: `47 passed, 2 skipped`
- `eval.run` executes and writes baseline artifacts

What reviewers should focus on:

- retrieval improvement strategy for strict page-level recall
- label strictness vs true retrieval failure on conceptual questions
- agent policy for balancing evidence sufficiency vs citation-producing answers
- smallest high-confidence ablation plan for next iteration

## How to Read This Report

If you are reviewing architecture:

- read Sections 2, 3, and the LangGraph subsection in Section 2

If you are reviewing evaluation quality:

- read Sections 5, 6, and Section 9 runbook/scenarios

If you are advising next steps:

- read Sections 8, 10, and 11

## Quick Glossary (for cross-discipline reviewers)

- RAG: retrieval-augmented generation (retrieve evidence first, then answer)
- FAISS: vector similarity search library used for semantic retrieval
- BM25: lexical retrieval algorithm for exact-term matching
- RRF: reciprocal rank fusion method to merge multiple ranked lists
- Recall@k: fraction of relevant evidence recovered in top-k results
- nDCG@k: ranking-quality metric that rewards relevant results appearing earlier
- refusal: explicit “do not answer” behavior when citation support is not reliable

## 1) Project Snapshot

This repo currently implements the Week-1 scoped PQC RAG system described in `reports/project_overview.md`:

- deterministic ingestion -> cleaning -> chunking -> embedding/indexing
- hybrid retrieval (`FAISS + BM25 + query fusion + RRF + optional rerank`)
- citation-first answer generation with refusal behavior
- bounded LangGraph agent controller (`retrieve -> assess -> refine/retrieve -> answer -> verify/refuse`)
- evaluation harness (`eval/run.py`) with retrieval metrics and diagnostics

In-scope source docs are present in `data/raw_pdfs/`:

- `NIST.FIPS.203.pdf`
- `NIST.FIPS.204.pdf`
- `NIST.FIPS.205.pdf`
- `NIST.IR.8545.pdf`
- `NIST.IR.8547.ipd.pdf`
- `NIST.SP.800-227.pdf`

Current processed artifact counts:

- `data/processed/pages.jsonl`: 309 rows
- `data/processed/pages_clean.jsonl`: 309 rows
- `data/processed/chunks.jsonl`: 552 rows
- `data/processed/chunk_store.jsonl`: 552 rows
- `eval/day4/questions.jsonl`: 13 rows

Snapshot boundaries:

- this report describes the local code/artifact state as of 2026-03-01
- optional scope expansion (additional PDFs/backends) is intentionally excluded unless already present

### Week-1 Milestone Status Matrix

Status legend:

- Complete: implemented and evidenced in code/artifacts
- Partial: implemented but missing measured delta or blocked validation
- Blocked: cannot be validated in current environment

| Milestone | Status | Evidence | Remaining Gap |
|---|---|---|---|
| M1 Deterministic ingestion/chunk/index | Complete | `rag/ingest.py`, `rag/clean.py`, `rag/chunk.py`, `rag/embed.py`, `rag/index_faiss.py`, `rag/index_bm25.py`; processed artifacts present | No open runtime blocker observed in current env; next improvement is adding environment preflight checks |
| M2 Citation-first RAG answers | Complete | `rag/rag_answer.py`, `rag/types.py`, `rag/ask.py` | Need stable answer-side evaluation runs (`--with-answers`) after env fix |
| M3 Bounded LangGraph controller | Complete | `rag/lc/graph.py`, `rag/lc/state.py`, `rag/lc/state_utils.py`, `tests/test_lc_graph.py` (8 passed) | Need more production-like run traces after env fix |
| M4 Eval baseline harness | Complete | `eval/dataset.py`, `eval/metrics.py`, `eval/run.py`, `reports/eval/day4_baseline/*` | Harness runs; remaining gap is systematic ablation reporting and answer-side tracking |
| M5 Measured retrieval improvement delta | Partial | `reports/mini_retrieval_sanity.md` (sanity gains on select queries) | Missing formal before/after ablation table on same eval set with primary metrics |

## 2) Implemented System End-to-End

### Ingestion

Files:

- `rag/ingest.py`

What is implemented:

- PDF parsing with LlamaParse (`result_type="markdown"`)
- per-PDF page count sanity check (`pypdf` pages vs parsed pages)
- deterministic PDF processing order (`sorted(RAW_DIR.glob("*.pdf"))`)
- outputs both per-doc parsed JSON and unified `data/processed/pages.jsonl`

### Cleaning

Files:

- `rag/clean.py`
- `scripts/clean_pages.py`

What is implemented:

- Unicode normalization, whitespace normalization, de-hyphenation
- repeated header/footer removal via frequency-based boilerplate detection
- structure-aware line joining that preserves table/code/math-like lines
- output contract includes cleaned text field (`text_clean`)

### Chunking

Files:

- `rag/chunk.py`
- `scripts/make_chunks.py`

What is implemented:

- block splitting by blank lines
- verbatim-ish block preservation for technical content
- greedy chunk packing with overlap and size controls
- deterministic chunk IDs (`{doc_id}::p{page}::c{idx}`)
- page-level citation fields preserved: `doc_id`, `start_page`, `end_page`

### Embedding and Index Construction

Files:

- `rag/embed.py`
- `rag/index_faiss.py`
- `rag/index_bm25.py`

What is implemented:

- embedding model: `BAAI/bge-base-en-v1.5`
- normalized embedding generation and `vector_id`-aligned chunk store
- FAISS exact index build (`IndexFlatIP` on normalized vectors)
- BM25 artifact build with technical compound-aware tokenizer

### Retrieval

Files:

- `rag/retrieve.py`
- `rag/retriever/base.py`
- `rag/retriever/faiss_retriever.py`
- `rag/retriever/bm25_retriever.py`
- `rag/retriever/factory.py`
- `rag/search.py`
- `rag/search_faiss.py`

What is implemented:

- swappable retriever contract (`ChunkHit`, `Retriever.search`)
- base mode and hybrid mode retrieval
- deterministic query variants for PQC technical phrasing
- reciprocal rank fusion (RRF)
- optional lightweight lexical rerank over fused candidates
- eval adapter `retrieve_for_eval(...)` with deterministic ranking key

### Answer Generation

Files:

- `rag/rag_answer.py`
- `rag/types.py`
- `rag/ask.py`
- `rag/llm/gemini.py`

What is implemented:

- evidence selection and budget controls
- strict inline citation contract (`[c#]` markers)
- refusal contract text: `not found in provided docs`
- citation validation and rejection of unsupported citations
- deterministic algorithm-step fallback when evidence clearly contains steps

### Agent Orchestration

Files:

- `rag/lc/graph.py`
- `rag/lc/tools.py`
- `rag/lc/state.py`
- `rag/lc/state_utils.py`
- `rag/lc/trace.py`
- `rag/agent/ask.py`

What is implemented:

- bounded LangGraph controller nodes:
  - `route`
  - `retrieve`
  - `assess_evidence`
  - `refine_query`
  - `answer`
  - `verify_or_refuse`
- loop budgets:
  - `AGENT_MAX_STEPS`
  - `AGENT_MAX_TOOL_CALLS`
  - `AGENT_MAX_RETRIEVAL_ROUNDS`
  - `AGENT_MIN_EVIDENCE_HITS`
- explicit `stop_reason` and `refusal_reason`
- trace artifact writing for each run

How the graph works (detailed, from `rag/lc/graph.py`):

1. Graph assembly and state machine shape
- Entry point is `route`.
- Compiled graph contains conditional edges:
  - `route -> (retrieve | answer | verify_or_refuse)`
  - `retrieve -> assess_evidence`
  - `assess_evidence -> (answer | refine_query | verify_or_refuse)`
  - `refine_query -> (retrieve | verify_or_refuse)`
  - `answer -> verify_or_refuse -> END`
- Runtime invoke path uses recursion guard:
  - `recursion_limit = max(20, MAX_STEPS * 4)`.

2. Node behavior and control intent
- `node_route`
  - bumps `steps` and logs step trace.
  - checks step budget first; if exhausted sets plan to refuse.
  - otherwise uses heuristic routing:
    - compare intent via regex patterns (`difference between`, `compare`, `vs/versus`)
    - definition intent (`what is`, `define`, `explain`)
    - algorithm intent (`algorithm`, `shake`)
    - default broad retrieval.
- `node_retrieve`
  - bumps `steps`.
  - enforces budget gates in sequence:
    - step limit
    - tool-call limit
    - retrieval-round limit
  - increments `tool_calls` and `retrieval_round`.
  - dispatches tools by action:
    - `retrieve` (query, `k=8`)
    - `resolve_definition` (`k=8`)
    - `compare` (`k=6`)
    - `summarize` (doc/page bounded)
  - merges incoming and existing evidence by `chunk_id` de-dup.
  - writes `last_retrieval_stats` (round, action, new hits, total hits, mode/tool stats).
- `node_assess_evidence`
  - computes sufficiency reasons:
    - `insufficient_hits`
    - `anchor_missing`
    - `compare_doc_diversity_missing`
  - anchor extraction is explicit:
    - regex anchors for `Algorithm N`, `Table N`, `Section x.y`
    - keyword anchors (`keygen`, `encaps`, `decaps`, `shake128`, `shake256`, `xof`)
  - sets:
    - `evidence_sufficient` boolean
    - `stop_reason`:
      - `sufficient_evidence` when pass
      - budget-exhausted reason when insufficient and a budget is hit
      - otherwise first insufficiency reason.
- `node_refine_query`
  - builds deterministic refinement from `stop_reason`:
    - `anchor_missing` -> append anchor tokens
    - `compare_doc_diversity_missing` -> add compare/topic and FIPS bias tokens
    - `insufficient_hits` -> add coverage/definition bias terms
  - replaces plan with refined `retrieve` action and logs `query_refined`.
- `node_answer`
  - only calls answer generator when:
    - `evidence_sufficient == True`
    - evidence list is non-empty
  - adapts evidence to `ChunkHit`, calls `build_cited_answer(...)`, maps citations back to state.
- `node_verify_or_refuse`
  - final guardrail; refuses if any of:
    - insufficient evidence
    - empty draft answer
    - empty evidence
    - zero citations
  - refusal path sets `refusal_reason`, clears citations, and sets refusal message.
  - success path promotes `draft_answer` to `final_answer`.

3. Edge decision logic (why the loop continues or stops)
- From `route`:
  - retrieval-like actions go to `retrieve`.
  - unknown/non-retrieval actions go to verification/refusal.
- From `assess_evidence`:
  - sufficient evidence -> `answer`.
  - insufficient + any budget exhausted -> `verify_or_refuse`.
  - insufficient + budget still available -> `refine_query`.
- From `refine_query`:
  - budget exhausted -> `verify_or_refuse`.
  - otherwise -> `retrieve` (next round).

4. Execution patterns reviewers should expect
- One-pass success:
  - `route -> retrieve -> assess_evidence -> answer -> verify_or_refuse -> END`
- Multi-round refinement:
  - `route -> retrieve -> assess_evidence -> refine_query -> retrieve -> assess_evidence -> answer -> verify_or_refuse -> END`
- Budget-protected refusal:
  - `route -> retrieve -> assess_evidence -> verify_or_refuse -> END`
  - triggered when evidence is still insufficient and step/tool/round budget is exhausted.

5. State/trace observability design
- `AgentState` carries both control and provenance fields:
  - counters: `steps`, `tool_calls`, `retrieval_round`
  - decisions: `evidence_sufficient`, `stop_reason`, `refusal_reason`
  - content: `plan`, `evidence`, `draft_answer`, `final_answer`, `citations`
  - diagnostics: `trace`, `last_retrieval_stats`, `errors`
- Every node appends structured trace events via `add_trace(...)`.
- `rag/agent/ask.py` can emit full state JSON and save run traces through `rag/lc/trace.py`.

6. Concrete trace walkthroughs (for reviewers)
- Case A: current controller refusal due missing citations after sufficient evidence
  - Artifact: `runs/agent/agent_20260219_101327_what_does_nist_say_about_pqc_for_wifi_9.json`
  - Path: `route -> retrieve -> assess_evidence -> answer -> verify_or_refuse`
  - Key state:
    - `evidence_sufficient=true`
    - `stop_reason=sufficient_evidence`
    - `citations=[]` after answer node
    - verify emits refusal with `refusal_reason=missing_citations`
  - Reviewer takeaway: the final citation gate is stricter than evidence sufficiency; this is intentional anti-hallucination behavior.
- Case B: current controller compare flow with same refusal mechanism
  - Artifact: `runs/agent/agent_20260219_100255_what_are_the_difference_between_mlkem_and_mldsa.json`
  - Path: `route(compare) -> retrieve(compare) -> assess_evidence -> answer -> verify_or_refuse`
  - Key state:
    - `tool_calls=1`, `retrieval_round=1`
    - `doc_diversity=4` and sufficient evidence
    - refusal still occurs because citations were not produced in drafted answer
  - Reviewer takeaway: compare retrieval can be sufficient while answer synthesis still fails citation compliance.
- Case C: success behavior reference (historical snapshot + current tests)
  - Historical artifact with successful verify: `runs/agent/agent_20260218_150604_what_are_the_steps_in_algorithm_2_shake128.json`
  - Current controller success path is also covered by unit tests (`tests/test_lc_graph.py`) where final answer path and refusal path are both asserted.
  - Reviewer takeaway: success behavior exists but current latest saved runs are refusal-heavy; this should be re-profiled after environment repair.

## 3) Models and Algorithms Used

### Models

- Parser: LlamaParse (`rag/ingest.py`)
- Embedding model: `BAAI/bge-base-en-v1.5` (`rag/embed.py`)
- Generation model wrapper: Gemini via `google-genai` (`rag/llm/gemini.py`, env `GEMINI_MODEL`)

### Retrieval and Ranking Algorithms

- Vector retrieval: FAISS inner product on L2-normalized vectors (`rag/index_faiss.py`, `rag/retriever/faiss_retriever.py`)
- Lexical retrieval: BM25 with persisted postings/idf (`rag/index_bm25.py`, `rag/retriever/bm25_retriever.py`)
- Query expansion: deterministic rule-based query variants (`query_variants` in `rag/retrieve.py`)
- Fusion: Reciprocal Rank Fusion `1 / (k0 + rank)` (`rrf_fuse` in `rag/retrieve.py`)
- Post-fusion rerank: exact technical token presence + BM25 lexical score (`rerank_fused_hits` in `rag/retrieve.py`)

### Answer/Citation Algorithms

- evidence dedupe and stable sorting
- inline citation enforcement per sentence
- refusal on missing/invalid citation support
- deterministic fallback extraction for algorithm-step questions

### Agent Decision Mechanisms

- heuristic route selection (compare/definition/algorithm/general)
- evidence sufficiency checks:
  - minimum evidence hit count
  - anchor token coverage
  - compare-mode doc diversity
- deterministic refine strategy keyed by insufficiency reason

## 4) Mechanisms and Guardrails

### Determinism

- stable chunk IDs and stable ordering in chunk/eval outputs
- deterministic qid ordering (`qid_sort_key`) in `eval/dataset.py`
- deterministic eval ranking tie-break key:
  - `(-score, doc_id, start_page, end_page, chunk_id)` in `rag/retrieve.py`
- deterministic JSON serialization (`sort_keys=True`) in eval outputs

### Citation Policy

- page-level citation metadata maintained on chunks (`doc_id`, `start_page`, `end_page`)
- non-refusal answers require valid citations
- refusal answers must have zero citations
- strict inline citation marker checks in `rag/rag_answer.py` and `rag/types.py`

### Bounded Agent Safety

- explicit step/tool/round budgets from `rag/config.py`
- loop stop routing to refusal when insufficient evidence and budget exhausted
- separate provenance fields:
  - `stop_reason` (loop outcome)
  - `refusal_reason` (final refusal cause)

### Data Contracts

- question dataset contract (`eval/dataset.py`)
- retrieval eval row contract (`rag/retrieve.py::retrieve_for_eval`)
- retriever interface contract (`rag/retriever/base.py`)
- agent state contract (`rag/lc/state.py`)

## 5) Evaluation Implementation

Files:

- `eval/dataset.py`
- `eval/metrics.py`
- `eval/run.py`
- `eval/day4/questions.jsonl`
- `reports/eval/day4_baseline/summary.json`
- `reports/eval/day4_baseline/per_question.jsonl`
- `reports/eval/day4_baseline/summary.md`

Implemented evaluation capabilities:

- dataset loading and validation (including labeled/unlabeled guardrails)
- retrieval metrics:
  - Recall@k
  - MRR@k
  - nDCG@k
- relaxed diagnostics:
  - doc-only hit rate
  - near-page hit rate (tolerance-based)
- multi-k metric computation in one run (`--ks`)
- deterministic eval artifact generation:
  - `per_question.jsonl`
  - `summary.json`
  - `summary.md`
- optional answer-side scoring (`--with-answers`) with citation/refusal checks

### Dataset Profile and Label Coverage

Source:

- `eval/day4/questions.jsonl` (13 rows)

Split:

- answerable: 12
- unanswerable: 1

Intent mix (heuristic categorization from question text):

- definition-style: 4
- algorithm/procedural: 4
- compare: 2
- general conceptual: 2
- why/policy: 1

Gold span distribution by doc_id:

- `NIST.FIPS.203`: 11 spans
- `NIST.FIPS.204`: 7 spans
- `NIST.FIPS.205`: 4 spans
- `NIST.IR.8547.ipd`: 1 span

Observed strict misses at k=8 from baseline per-question outputs:

- `q001`, `q002`, `q003`, `q005`, `q009`, `q010`, `q012`

Reviewer note:

- misses are concentrated in conceptual/definition/compare questions with broader evidence footprints, which may indicate both retriever ranking issues and strict page-label narrowness.

## 6) Evaluation Results (Day4 Baseline Snapshot)

Source:

- `reports/eval/day4_baseline/summary.json`

Run metadata:

- generated at: `2026-03-01T02:30:49.683700+00:00`
- dataset: `eval/day4/questions.jsonl`
- retrieval mode/backend: `hybrid` / `faiss`
- `k=8`, `ks=[1,3,5,8]`, fusion enabled, rerank enabled

Primary retrieval metrics (strict, answerable with non-empty gold only):

- Recall@8: `0.4167`
- MRR@8: `0.2986`
- nDCG@8: `0.3319`

By-k retrieval trend:

- k1: recall `0.2500`, mrr `0.2500`, ndcg `0.2500`
- k3: recall `0.3333`, mrr `0.2778`, ndcg `0.2917`
- k5: recall `0.3750`, mrr `0.2986`, ndcg `0.3137`
- k8: recall `0.4167`, mrr `0.2986`, ndcg `0.3319`

Secondary diagnostics at k8:

- strict page-overlap hit rate: `0.4167`
- near-page (+/-1) hit rate: `0.5833`
- doc-only hit rate: `0.8333`

Interpretation:

- retrieval often reaches the correct document family
- strict page-level localization remains the weaker part
- this aligns with current need for retrieval/rerank tuning and possible label-span calibration on broad conceptual questions

Metric interpretation (plain language):

| Metric | Current value | Plain-language meaning |
|---|---:|---|
| Recall@8 | 0.4167 | About 42% of labeled evidence spans are recovered in top-8 results |
| MRR@8 | 0.2986 | First relevant hit now tends to appear earlier (around rank 3-4 on average) |
| nDCG@8 | 0.3319 | Ranking quality improved but remains moderate |
| Doc-only hit-rate@8 | 0.8333 | System often finds the right document family |
| Strict hit-rate@8 | 0.4167 | Exact page overlap remains the main weakness |

### Suggested Quality Targets for Next Iteration

These targets are proposed to make reviewer feedback decision-oriented.

| Metric | Current | Target (next cycle) | Why this target |
|---|---:|---:|---|
| Recall@8 | 0.4167 | >= 0.55 | Raise strict evidence recall before deeper agent tuning |
| nDCG@8 | 0.3319 | >= 0.40 | Improve top-rank relevance quality |
| Strict hit-rate@8 | 0.4167 | >= 0.55 | Align with Recall@8 direction |
| Near-page hit-rate@8 | 0.5833 | >= 0.70 | Improve localization even before exact overlap |
| Doc-only hit-rate@8 | 0.8333 | >= 0.88 | Preserve strong document-family retrieval while tightening rank |
| Answer eval coverage | disabled | run `--with-answers` on full set | Track citation/refusal behavior with retrieval changes |

## 7) File Map for Reviewers

### Open-tab files explicitly included

- `reports/notes.md`
- `rag/lc/graph.py`
- `eval/run.py`
- `reports/Day4_summary.md`
- `reports/day4_langgraph_bounded_iterative_refinement_report_20260219.md`

### Core pipeline and contracts

- `reports/project_overview.md`
- `rag/ingest.py`
- `rag/clean.py`
- `rag/chunk.py`
- `rag/embed.py`
- `rag/index_faiss.py`
- `rag/index_bm25.py`
- `rag/retrieve.py`
- `rag/retriever/base.py`
- `rag/retriever/factory.py`
- `rag/retriever/faiss_retriever.py`
- `rag/retriever/bm25_retriever.py`
- `rag/rag_answer.py`
- `rag/types.py`
- `rag/config.py`
- `rag/lc/state.py`
- `rag/lc/state_utils.py`
- `rag/lc/tools.py`
- `rag/lc/trace.py`
- `rag/ask.py`
- `rag/agent/ask.py`
- `eval/dataset.py`
- `eval/metrics.py`
- `eval/run.py`

## 8) Current Problems and Risks

Classification:

- P0: runtime blockers
- P1: engineering/design risks
- P2: documentation drift
- Active P0 issues: none (as of 2026-03-01 verification refresh)

Important interpretation note:

- previous FAISS runtime blocker is resolved; remaining issues are engineering and documentation quality risks

### P1-1: Eager backend import path can reduce portability when optional dependencies are absent

Symptom:

- `rag.retriever.factory` imports `FaissRetriever` at module import time
- `rag.retrieve` imports the factory at module import time

Evidence:

- code path:
  - `rag/retrieve.py -> rag/retriever/factory.py -> rag/retriever/faiss_retriever.py`

Impact:

- tighter coupling to installed backend dependencies than necessary
- portability risk for future multi-backend deployments/tests

Suggested fix direction:

- lazy-import backend classes inside `get_retriever(...)` branches
- keep top-level modules free of backend-specific hard imports where possible

### P1-2: Retrieval tool hints and filters are currently best-effort and may be silently dropped

Symptom:

- `rag/lc/tools.py` passes `mode_hint` and `filters` into retrieval adapter
- `rag/retrieve.py::retrieve` does not accept these parameters directly

Evidence:

- `rag/lc/tools.py::_call_with_flexible_signature` intentionally drops unsupported kwargs

Impact:

- agent/tool behavior may appear configurable while some hints are no-op
- can make tuning/debugging less transparent

Suggested fix direction:

- either explicitly support `mode_hint`/`filters` in retrieval API or log dropped args in debug mode

### P2-1: README references non-existent Day2 eval modules

Symptom:

- README documents `eval/day2/run.py` and `eval/day2/ablate.py`
- current tree has `eval/day2/run_baseline.py` but no `run.py` or `ablate.py`

Evidence:

- files present in `eval/day2/`:
  - `answer_n_evidence.txt`
  - `questions.txt`
  - `run_baseline.py`
- README section "Scripts reference" includes outdated module names

Impact:

- reviewer onboarding friction
- command copy-paste failures

Suggested fix direction:

- align README script references with actual file names and entrypoints

### P2-2: Historical report narratives may conflict with current implementation state

Symptom:

- older reports/notes include earlier controller behavior or interim assumptions

Evidence:

- historical context files:
  - `reports/Day4_summary.md`
  - `reports/day4_langgraph_bounded_iterative_refinement_report_20260219.md`
  - `reports/notes.md`

Impact:

- review confusion if old narrative is read as current state

Suggested fix direction:

- tag historical reports as "as-of date snapshots"
- link this report as current canonical status snapshot

### Prioritized Action Plan (Owner / Effort / Impact)

| Action ID | Priority | Owner | Effort | Expected Impact | Success Signal |
|---|---|---|---|---|---|
| A1 Add environment preflight checks to CI/local smoke scripts | P1 | Project maintainer | S | Prevents dependency regressions from being discovered late | one-command preflight catches missing critical deps |
| A2 Lazy-import retriever backends | P1 | Project maintainer | S-M | Restores practical backend swappability | backend-specific deps only required when backend selected |
| A3 Make `mode_hint`/`filters` explicit in retrieval API | P1 | Project maintainer | M | Removes silent no-op behavior and improves agent tuning transparency | tool args either applied or explicitly logged |
| A4 Align README entrypoints with repo reality | P2 | Project maintainer | S | Reduces onboarding friction for reviewers | README commands all runnable as written |
| A5 Run formal retrieval ablations with fixed dataset | P1 | Project maintainer | M | Produces measurable improvement deltas | before/after table for Recall/MRR/nDCG and diagnostics |
| A6 Add answer-side eval tracking in baseline runs | P1 | Project maintainer | S-M | Quantifies citation/refusal outcomes | report includes citation_presence and refusal_accuracy trends |

## 9) Validation Scenarios Run for This Status Report

### Minimal Reviewer Runbook (Copy/Paste)

Prerequisites:

- use conda env: `eleven`

Commands and expected outcomes as of 2026-03-01:

1. `conda run -n eleven python -m pytest -q`
   - expected now: pass (`47 passed, 2 skipped`)
2. `conda run -n eleven python -m pytest -q tests/test_eval_dataset.py tests/test_eval_metrics.py tests/test_lc_graph.py`
   - expected now: pass (`18 passed`)
3. `conda run -n eleven python -m eval.run --dataset eval/day4/questions.jsonl --allow-unlabeled`
   - expected now: pass and write:
     - `reports/eval/day4_baseline/per_question.jsonl`
     - `reports/eval/day4_baseline/summary.json`
     - `reports/eval/day4_baseline/summary.md`
4. `conda run -n eleven python -m rag.search_faiss "ML-KEM key generation"`
   - expected now: pass and print top cited chunks

Reviewer interpretation rule:

- if (1), (2), (3), and (4) pass, environment baseline is healthy; feedback should focus on retrieval quality, evaluation design, and agent policy.

### Scenario 1: Environment check (pass)

Command:

- `conda run -n eleven python -m pytest -q`

Observed:

- `47 passed, 2 skipped, 3 warnings`

### Scenario 2: Partial quality gate (pass)

Commands:

- `conda run -n eleven python -m pytest -q tests/test_eval_dataset.py tests/test_eval_metrics.py`
- `conda run -n eleven python -m pytest -q tests/test_lc_graph.py`

Observed:

- eval dataset/metrics tests: `10 passed`
- LangGraph tests: `8 passed`

### Scenario 3: Eval runner execution (pass)

Command:

- `conda run -n eleven python -m eval.run --dataset eval/day4/questions.jsonl --allow-unlabeled`

Observed:

- writes:
  - `reports/eval/day4_baseline/per_question.jsonl`
  - `reports/eval/day4_baseline/summary.json`
  - `reports/eval/day4_baseline/summary.md`

### Scenario 4: Artifact availability check (pass)

Checked:

- processed data artifacts exist (`pages`, `clean pages`, `chunks`, `chunk_store`, `faiss.index`, `bm25.pkl`)
- baseline eval artifacts exist:
  - `reports/eval/day4_baseline/summary.json`
  - `reports/eval/day4_baseline/per_question.jsonl`
  - `reports/eval/day4_baseline/summary.md`

## 10) Recommended Next Validation Steps (Eval-Driven)

1. Keep environment validation in the regular loop (no current FAISS blocker):
   - `conda run -n eleven python -m pytest -q`
   - `conda run -n eleven python -m rag.search_faiss "ML-KEM key generation"`
   - `conda run -n eleven python -m eval.run --dataset eval/day4/questions.jsonl --allow-unlabeled`
2. Treat current Day4 baseline as refreshed reference and track deltas from this run.
3. Run small retrieval ablations (candidate multiplier, `k0`, rerank pool, fusion/rerank toggles), and report changes in Recall@k/MRR/nDCG.
4. Resolve README drift so reviewers can execute commands without path mismatches.

## 11) Feedback Requested from Reviewers

Please focus feedback on these decision points:

1. Retrieval strategy: should next effort prioritize query-variant refinement, rerank redesign, or chunking adjustments first for strict page overlap gains?
2. Eval protocol: are the proposed next-cycle metric targets realistic for this 13-question baseline, or should targets be reshaped by intent buckets?
3. Label quality: which baseline misses look like retriever failure versus strict-label span mismatch that should be widened?
4. Agent policy: should `assess_evidence` be tightened so `sufficient_evidence` more strongly predicts citation-producing answers?
5. Refusal behavior: should the current strict refusal on missing citations remain unchanged, or should there be one bounded auto-repair retry before refusal?
6. Interface design: do you agree with making retrieval tool args (`mode_hint`, `filters`) explicit and enforced rather than best-effort?
7. Backend portability: is lazy import sufficient, or should retriever backend modules be fully isolated behind runtime plugin loading?
8. Next experiment design: what is the smallest ablation set that would give high-confidence direction within one iteration?

---

This report is intentionally code-verified against current repository state as of 2026-03-01 and is intended to be the current review baseline.
