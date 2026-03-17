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

## 2026-03-16 - Graph runtime upgrade: algorithm lookup + section priors

### Goal
Promote graph-lite from a definition-only analyze-query helper into one bounded runtime feature that measurably affects `/ask-agent`, while keeping `/ask` unchanged and keeping Neo4j out of the serving path.

### Implementation
- added `lookup_algorithm(...)` in `rag/graph/query.py` with:
  - exact algorithm-name matching
  - exact `Algorithm N` matching
  - scoped multi-match behavior for ambiguous bare algorithm numbers
  - deterministic term fallback for identifier-only queries such as `ML-DSA.KeyGen`
- replaced the definition-only graph hook in `rag/lc/graph.py` with a mode-aware graph lookup step for:
  - `definition`
  - `algorithm`
- threaded `candidate_section_ids` into planner retrieval only and used them as soft section-aware rerank priors in `rag/retrieve.py`
- extended `/ask-agent` summaries so:
  - `trace_summary` includes a compact `graph_lookup` block
  - `analysis` includes the full structured `graph_lookup` payload
- kept direct `/ask` unchanged as the runtime control path

### Scope boundaries
- Neo4j remains a dev/debug/export surface rather than a live dependency
- graph signals bias retrieval inside the bounded agent path only; they do not replace hybrid retrieval
- no new extraction pass or offline LLM-driven graph expansion was added in this update

### Eval and artifacts
- added a focused graph-runtime eval slice at `eval/graph_runtime_ablation.jsonl`
- added `python -m eval.graph_runtime_ablation --out <path>` to compare:
  - direct `/ask`
  - `/ask-agent` with graph lookup only
  - `/ask-agent` with graph lookup plus section priors

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

## 2026-03-09 - Day 2 planner upgrade: bounded retrieval planner for `analyze_query`

### Goal
Promote `analyze_query` from a structured query normalizer into a bounded retrieval planner for the LangGraph path, while keeping:
- the deterministic analyzer as the guardrail and fallback layer
- direct `/ask`, `/search`, and eval adapters on the existing query-centric retrieval contract
- retrieval changes thin and graph-scoped rather than turning into a hidden retriever rewrite

This update supersedes the execution description in the earlier Day 2 entry where:
- graph routing still branched to `compare` and `resolve_definition`
- analysis output was still centered on `canonical_query` + `required_anchors`

### High-Level Before/After Planner Shape

### Before
`user question -> deterministic hints -> LLM-corrected JSON -> canonical_query -> graph retrieve`

Behavior before this upgrade:
- the LLM mostly validated or lightly rewrote deterministic analysis
- `canonical_query` remained the main retrieval input
- graph execution still depended on specialized compare / definition branches
- evidence assessment remained coarse (`anchor_missing`, compare doc diversity, insufficient hits)

### After
`user question -> protected spans + guardrails -> LLM retrieval plan -> planner-aware retrieve -> bounded assess/refine`

Behavior after this upgrade:
- `analyze_query` emits retriever-oriented planner fields rather than just a normalized query
- `canonical_query` remains for trace/debug/fallback, not as the only executable retrieval query
- graph execution always routes through `retrieve` for QA requests
- compare and definition remain visible as planner metadata rather than graph actions
- assessment/refinement now operate on cheap planner-aware failure categories

### Implementation

#### 1) Planner schema and state
Extended `QueryAnalysis`, LangGraph state, and serialized `query_analysis` with:
- `rewrite_needed`
- `protected_spans`
- `sparse_query`
- `dense_query`
- `subqueries`
- `confidence`

Existing fields were intentionally preserved:
- `original_query`
- `canonical_query`
- `mode_hint`
- `required_anchors`
- `compare_topics`
- `doc_ids`
- `doc_family`
- `analysis_notes`
- `answer_prompt_question`

Role split after this change:
- `canonical_query` = trace/debug/fallback
- `protected_spans` = primary guardrail concept
- `required_anchors` = compatibility mirror of assessable protected spans
- `sparse_query` = BM25 input
- `dense_query` = dense retrieval input
- `answer_prompt_question` = wording passed to answer generation
- `confidence` = trace/debug signal only in this phase

#### 2) Deterministic guardrail layer + planner prompt
Kept deterministic extraction for:
- `Algorithm N`
- `Table N`
- `Section x.y`
- dotted identifiers such as `ML-KEM.Decaps`
- explicit doc references such as `FIPS 203` and `SP 800-227`
- compare topic pairs

The deterministic analyzer now also:
- builds `protected_spans`
- derives assessable `required_anchors`
- computes deterministic fallback `sparse_query`, `dense_query`, `subqueries`, and `confidence`
- remains the fallback if planner JSON is invalid or unavailable

The planner prompt was reframed from “correct this JSON” to “plan retrieval for a bounded hybrid PQC standards retriever.”

Prompt/validation rules now enforce:
- preserve `protected_spans` exactly
- do not invent document IDs
- emit one allowed `mode_hint`
- emit separate `sparse_query` and `dense_query`
- emit `subqueries` only for compare mode
- cap compare `subqueries` at 2
- prefer `rewrite_needed = false` when the raw query is already retrieval-ready

One additional hardening fix was added after a live agent smoke:
- when deterministic doc scope is empty, the LLM can no longer invent `doc_ids` or `doc_family`

#### 3) Graph simplification
The graph was simplified intentionally:
- `node_route` now always emits `Plan(action=\"retrieve\")` for query-answering requests
- `mode_hint` and `compare_topics` remain in analysis state and traces
- graph no longer depends on `compare` / `resolve_definition` as primary execution branches

Important non-change:
- the `compare` and `resolve_definition` tools remain in-repo
- this is a graph-level simplification, not a repo-wide statement that those tools are removed

#### 4) Thin planner-aware retrieval adapter
Added a thin graph-side retrieval adapter without rewriting the shared retrieval API surface used by direct `/ask` and eval.

Execution policy in this phase:
- BM25 runs on `sparse_query`
- dense retrieval runs on `dense_query`
- compare mode is the only mode allowed to use `subqueries`
- each compare subquery runs once through each backend
- all rankings are fused with existing RRF
- existing filtering and rerank behavior are reused
- graph-path retrieval still keeps query fusion and retrieval-side mode-variant expansion disabled

Backward compatibility:
- the graph-facing `retrieve` tool accepts planner fields optionally
- if planner fields are absent, the legacy `query`-based behavior still works

#### 5) Planner-aware assessment and refinement
Assessment was tightened to cheap operational categories only:
- `missing_protected_span`
- `wrong_doc_scope`
- `one_sided_comparison`
- `insufficient_hits`

This avoids a larger scoring subsystem while making failure traces more actionable.

Refinement remains bounded:
- missing protected span -> append missing spans to the next sparse query and lightly tighten dense phrasing
- wrong doc scope -> retry with existing scoped doc constraints
- one-sided comparison -> bias toward the missing side’s topic/doc tokens
- insufficient hits -> reuse light coverage-bias behavior

### Regression Coverage
Added or updated tests for:
- planner-shaped analysis output fields
- deterministic fallback when planner JSON is invalid
- graph routing through `retrieve` while preserving compare/definition metadata
- graph retrieval plumbing for `canonical_query`, `sparse_query`, `dense_query`, `subqueries`, and `protected_spans`
- compare-mode-only subquery execution with a hard cap of 2
- planner-aware assessment categories
- `/ask-agent` service and API analysis payload compatibility

### Verification
Verified with:
- `python -m py_compile rag/lc/state.py rag/lc/state_utils.py rag/lc/graph.py rag/lc/tools.py rag/retrieve.py rag/service.py tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py tests/test_retrieve_eval_api.py`
- `conda run -n pyt python -m pytest -q tests/test_lc_graph.py tests/test_lc_tools.py tests/test_api.py tests/test_retrieve_eval_api.py`

Observed results:
- targeted planner slice: `39 passed, 1 skipped`

Additional live check performed:
- one real `rag.agent.ask ... --json --no-trace` smoke was run for an anchored algorithm query
- this surfaced planner over-scoping of `doc_ids` / `doc_family`
- validation was tightened so deterministic doc-scope guardrails now win

Not completed in this update:
- full-suite regression rerun
- stable live smoke coverage for broader compare/general prompts under the configured local backend

### Status
- planner-shaped `analyze_query`: complete
- deterministic protected-span guardrail layer: complete
- graph-level simplification to retrieve-only execution: complete
- thin planner-aware retrieval adapter: complete
- planner-aware assessment/refinement categories: complete
- `/ask-agent` analysis payload upgrade: complete
- targeted planner verification slice: complete
- full-suite rerun: not completed in this update
- broader live backend smoke set: not completed in this update

## 2026-03-09 - Day 2 trace upgrade: readable LangGraph traces + shared summarizer

### Goal
Upgrade the LangGraph trace path from a mostly raw final-state dump into a readable observability layer that makes the bounded controller easier to inspect in demos, API responses, and saved artifacts.

This update was intentionally scoped to:
- keep `state["trace"]` as the canonical flat event stream
- derive human-facing trace views from that stream rather than storing multiple competing trace formats in graph state
- unify saved trace files and `/ask-agent` `trace_summary` behind one shared summarizer
- improve trace readability without changing retrieval, answer policy, or graph budgets

### High-Level Before/After Trace Shape

### Before
`final AgentState -> write_trace(...) -> single JSON dump`

Behavior before this upgrade:
- `rag/lc/trace.py` deep-copied final state and wrote one JSON file
- evidence text was truncated, but the artifact still remained mostly a raw state dump
- mutation helpers emitted coarse events such as `analysis`, `plan`, `evidence`, `answer`, and `final_answer`
- `/ask-agent` used a separate ad hoc service-side `trace_summary` builder
- older `verify` events existed only as compact node-level terminal events

### After
`flat trace events -> summarize_trace(state) -> readable summary JSON + sibling raw JSON`

Behavior after this upgrade:
- `write_trace(...)` now writes:
  - `agent_<ts>_<slug>.json` as the readable summary artifact
  - `agent_<ts>_<slug>.raw.json` as the raw final-state dump
- one shared `summarize_trace(state)` function now drives:
  - saved trace summaries
  - `/ask-agent` `trace_summary`
  - `/ask-agent` `analysis`
- the trace is now presented through:
  - `run`
  - `analysis`
  - `retrieval_summary`
  - `answer_summary`
  - `trace_by_node`
  - `timeline`
  - `evidence_preview`

### Implementation

#### 1) Trace artifact split and shared summarizer
Reworked `rag/lc/trace.py` so the trace writer is no longer a simple final-state serializer.

Added:
- `summarize_trace(state)` as the canonical trace summarizer
- readable summary JSON output
- sibling raw state output for debugging and replay

Readable summary sections now include:
- `run`: question, entry node, mode, rewrite flag, doc scope, result, stop/refusal reason, steps, tool calls, retrieval rounds, evidence hit count, citation count, top chunk IDs, timing totals
- `analysis`: planner-shaped analysis payload from graph state
- `retrieval_summary`: one card per retrieval round with query inputs, doc scope, hit counts, top docs, assessment outcome, refinement reason, and retrieval timing
- `answer_summary`: answer prompt wording, draft/final lengths, citation keys, refusal/template usage, and final result
- `trace_by_node`: ordered node-visit cards so repeated `retrieve` and `assess_evidence` visits remain distinct
- `timeline`: compact execution-order event list
- `evidence_preview`: top evidence hits with short previews only

The summary file intentionally no longer repeats the full evidence blobs; the raw file preserves the full final state.

#### 2) Mutation-level trace events in state helpers
Updated `rag/lc/state_utils.py` so major state mutations emit compact, explicit trace events directly from the mutator helpers.

New mutation event names:
- `analysis_applied`
- `plan_applied`
- `evidence_updated`
- `answer_drafted`
- `final_answer_set`

This replaced the older coarser helper events:
- `analysis`
- `plan`
- `evidence`
- `answer`
- `final_answer`

Compact payload design:
- `analysis_applied` records mode, rewrite flag, protected spans, doc scope, sparse/dense query, subquery count, and analysis notes
- `plan_applied` records action, reason, query, mode hint, and summarized args
- `evidence_updated` records retrieval round, total hits, top chunk IDs, and top doc IDs
- `answer_drafted` records prompt wording, draft length, citation count/keys, and generation timing
- `final_answer_set` records final result, final length, refusal-template usage, stop reason, and refusal reason

#### 3) Node attribution and terminal verification trace
Added a private observability-only `_trace_active_node` field in graph state.

Behavior after this change:
- `_bump_step(...)` sets `_trace_active_node`
- `add_trace(...)` attaches `node` automatically when the caller does not provide one
- the field is used only for trace attribution, not for control flow

The terminal verification path was also upgraded:
- graph now emits `verification_decision` instead of only `verify`
- `summarize_trace(...)` accepts both legacy `verify` and new `verification_decision` events during the transition
- `set_final_answer(...)` now emits `final_answer_set` with explicit result metadata

#### 4) Sparse timing surfacing
Timing was kept intentionally sparse and high-signal.

No generic per-mutation `timing_recorded` event was added.

Instead:
- retrieval round timing is attached to `retrieval_round_result`
- generation timing is attached to `answer_drafted`
- aggregate stage timing remains in `state["timing_ms"]` and is surfaced in `run`

This kept the timeline readable while still showing where time went.

#### 5) Shared `/ask-agent` summary source
Removed the service-layer trace duplication by making `rag/service.py` depend on `summarize_trace(state)`.

`/ask-agent` now returns:
- `trace_summary = summarize_trace(state)["run"]`
- `analysis = summarize_trace(state)["analysis"]`

This keeps saved trace files and the API surface aligned.

### Follow-up Hardening Fix
After the trace upgrade, a real `python -m rag.agent.ask "What are the differences between ML-KEM and ML-DSA?"` smoke surfaced a trace serialization bug.

Observed issue:
- `assessment_decision["missing_compare_topics"]` in the current graph is list-shaped
- the new trace compactor briefly assumed it was dict-shaped and called `dict(...)` on it
- this caused `write_trace(...)` to fail after the answer had already been produced

Fix applied:
- hardened `rag/lc/trace.py` preview logic so `missing_compare_topics` now accepts:
  - current list-shaped payloads
  - legacy dict-shaped payloads
  - simple string payloads defensively

This restored trace writing for real compare runs while keeping legacy trace compatibility.

### Regression Coverage
Added `tests/test_lc_trace.py` as the source of truth for the trace contract.

Coverage added or updated for:
- mutation helper event names and node attribution
- `verification_decision` plus `final_answer_set` ordering for success and refusal paths
- legacy `verify` normalization in `summarize_trace(...)`
- `_trace_active_node` remaining observability-only
- per-round `retrieval_summary` grouping
- sparse timing surfacing in round / answer / run summaries
- summary + raw trace file writing
- legacy dict-shaped and current list-shaped `missing_compare_topics` payload compatibility
- `/ask-agent` using the shared trace summarizer rather than a separate ad hoc builder

### Verification
Verified with:
- `conda run -n pyt python -m pytest -q tests/test_lc_trace.py tests/test_lc_graph.py tests/test_api.py`
- `conda run -n pyt python -m pytest -q tests/test_lc_tools.py tests/test_retrieve_eval_api.py`

Observed results:
- trace / graph / API slice: `33 passed, 1 warning`
- tool / retrieve-eval compatibility slice: `13 passed, 1 skipped, 1 warning`

Live smoke performed after the hardening fix:
- `conda run -n pyt python -m rag.agent.ask "What are the differences between ML-KEM and ML-DSA?"`

Observed result:
- command completed successfully
- trace summary was written to `runs/agent/agent_20260309_205312_what_are_the_differences_between_mlkem_and_mldsa.json`
- sibling raw state was written to `runs/agent/agent_20260309_205312_what_are_the_differences_between_mlkem_and_mldsa.raw.json`

Note:
- the `torchvision` image-extension warning still appears in the environment but is unrelated to trace serialization

### Status
- readable summary + raw trace artifact split: complete
- shared `summarize_trace(state)` for file + API paths: complete
- mutation-level trace events in state helpers: complete
- ordered `trace_by_node` / `retrieval_summary` / `answer_summary`: complete
- verification-event migration with legacy compatibility: complete
- post-smoke `missing_compare_topics` hardening fix: complete
- targeted trace regression coverage: complete
- broader full-suite rerun: not completed in this update
