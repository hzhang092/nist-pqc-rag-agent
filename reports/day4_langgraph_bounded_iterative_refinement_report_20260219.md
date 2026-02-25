# Technical Report: Bounded Iterative LangGraph Controller Upgrade

Date: 2026-02-19
Project: `nist-pqc-rag-agent`

## 1. Executive Summary

This upgrade replaced a single-pass LangGraph controller with a bounded iterative controller that follows the project scope contract:

`retrieve -> assess evidence -> (optional) refine/retrieve again -> answer -> verify/refuse`

The implementation introduces explicit loop budgets, deterministic stop rules, structured refusal semantics, and improved traceability.

Primary outcomes:

- Agent flow is now multi-step and bounded.
- Evidence refinement is deterministic and testable.
- The agent avoids LLM answer generation when evidence is insufficient and budgets are exhausted.
- `stop_reason` and `refusal_reason` are now separated for unambiguous diagnostics.

---

## 2. Old vs New Agent (LangChain/LangGraph)

### 2.1 Old controller (single-pass)

Observed in historical run artifact:

- `runs/agent/agent_20260218_113844_what_are_the_differences_between_mlkem_and_mldsa.json`
- Trace path: `route -> do_tool -> answer -> verify_or_refuse -> END`
- Only one tool pass (`tool_calls = 1`) and no evidence refinement loop.

Operational limitations:

- No iterative evidence collection.
- Budgets existed but were not driving loop decisions.
- Verification could only post-check answer; retrieval strategy was not adaptive.

### 2.2 New controller (bounded iterative)

Implemented in:

- `rag/lc/graph.py`

Current graph nodes:

- `route`
- `retrieve`
- `assess_evidence`
- `refine_query`
- `answer`
- `verify_or_refuse`

Current control path:

`route -> retrieve -> assess_evidence -> (answer | refine_query -> retrieve | verify_or_refuse) -> verify_or_refuse -> END`

Safety behavior:

- If evidence is insufficient and any budget is exhausted, graph bypasses answer generation and goes directly to refusal via `verify_or_refuse`.

---

## 3. Files Improved (What and Why)

### 3.1 `rag/lc/graph.py`

Why changed:

- Core orchestration logic needed to match scope (`retrieve -> assess -> optionally retrieve again -> answer`).

What changed:

- Added bounded loop constants from config:
  - `MAX_STEPS`
  - `MAX_TOOL_CALLS`
  - `MAX_RETRIEVAL_ROUNDS`
  - `MIN_EVIDENCE_HITS`
- Added budget checks:
  - `_step_limit_hit`
  - `_tool_limit_hit`
  - `_round_limit_hit`
  - `_budget_limit_reason`
- Added evidence assessment logic:
  - hit threshold checks
  - anchor-aware checks (`Algorithm/Table/Section`, plus keywords like `keygen`, `encaps`, `decaps`, `shake128`, `xof`)
  - compare-mode doc diversity check
- Added deterministic query refinement:
  - anchor-token biasing
  - compare-topic biasing (including FIPS hints)
  - coverage/definition biasing
- Added safer verify semantics:
  - refusal when no reliable citation support
  - explicit `refusal_reason` generation (`_derive_refusal_reason`)
- Added recursion guard:
  - `GRAPH.invoke(..., config={"recursion_limit": ...})`

Concrete implementation mechanisms:

- Budget constants are bound once from config:
  - `MAX_STEPS = SETTINGS.AGENT_MAX_STEPS`
  - `MAX_TOOL_CALLS = SETTINGS.AGENT_MAX_TOOL_CALLS`
  - `MAX_RETRIEVAL_ROUNDS = SETTINGS.AGENT_MAX_RETRIEVAL_ROUNDS`
  - `MIN_EVIDENCE_HITS = SETTINGS.AGENT_MIN_EVIDENCE_HITS`
- Budget gate helpers are explicit:
  - `_step_limit_hit`, `_tool_limit_hit`, `_round_limit_hit`
  - `_budget_limit_reason` prioritizes step/tool/round exhaustion reasons.
- Compare intent parsing uses 3 regex families:
  - `difference(s) between A and B`
  - `compare/comparison of A and B`
  - `A vs/versus B`
  - with topic cleanup and identical-topic rejection.
- Evidence sufficiency includes three concrete checks:
  - minimum evidence count (`len(evidence) < MIN_EVIDENCE_HITS`)
  - anchor presence (`Algorithm N`, `Table N`, `Section x.y`, and keywords like `keygen`, `shake128`)
  - compare-specific doc diversity (`>=2` distinct `doc_id` values).
- Query refinement strategy is deterministic from `stop_reason`:
  - `anchor_missing` -> append missing anchor tokens
  - `compare_doc_diversity_missing` -> append topic/doc bias tokens (e.g., `FIPS 203`, `FIPS 204`)
  - `insufficient_hits` -> add coverage or definition bias terms.
- Answer adapter `_call_rag_answer(...)` converts `EvidenceItem` -> `ChunkHit`, calls `build_cited_answer(...)`, and preserves optional citation `key` into graph state.
- Verification/refusal node derives refusal via `_derive_refusal_reason(...)` and emits user-facing refusal text via `_refusal_message(...)` while preserving loop provenance (`stop_reason`).

### 3.2 `rag/config.py`

Why changed:

- Budgets needed to be configurable and validated centrally.

What changed:

- Added agent loop settings:
  - `AGENT_MAX_STEPS`
  - `AGENT_MAX_TOOL_CALLS`
  - `AGENT_MAX_RETRIEVAL_ROUNDS`
  - `AGENT_MIN_EVIDENCE_HITS`
- Added `_env_int_any(...)` to allow backward-compatible fallback to `ASK_MIN_EVIDENCE_HITS`.
- Extended `validate_settings()` with strict bounds checks.

Concrete implementation mechanisms:

- Agent settings are env-backed with defaults:
  - `AGENT_MAX_STEPS` (default `8`)
  - `AGENT_MAX_TOOL_CALLS` (default `3`)
  - `AGENT_MAX_RETRIEVAL_ROUNDS` (default `2`)
  - `AGENT_MIN_EVIDENCE_HITS` (default/fallback `2` via `_env_int_any`).
- `_env_int_any(("AGENT_MIN_EVIDENCE_HITS", "ASK_MIN_EVIDENCE_HITS"), 2)` keeps old knobs compatible.
- `validate_settings()` hard-fails on non-positive step/tool/round budgets or negative min-evidence values, preventing invalid runtime loop configs.

### 3.3 `rag/lc/state.py`

Why changed:

- Iterative controller requires explicit loop/provenance fields.

What changed:

- Added:
  - `retrieval_round`
  - `evidence_sufficient`
  - `stop_reason`
  - `refusal_reason`
  - `last_retrieval_stats`

Concrete implementation mechanisms:

- `AgentState` (`TypedDict`) now carries loop counters (`steps`, `tool_calls`, `retrieval_round`) plus control/diagnostic flags (`evidence_sufficient`, `stop_reason`, `refusal_reason`).
- `EvidenceItem` is normalized to retrieval contract fields: `score`, `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`.
- `Citation` includes optional `key` so inline citation keys (`c1`, `c2`) survive graph transitions.
- `Plan` includes `action`, `reason`, optional `query`, optional `args`, and optional `mode_hint` (`general/definition/algorithm/symbolic`) for deterministic routing + retrieval behavior.

### 3.4 `rag/lc/state_utils.py`

Why changed:

- New state contract must be deterministically initialized.

What changed:

- `init_state(...)` now initializes all new fields, including `refusal_reason` and loop counters.

Concrete implementation mechanisms:

- `init_state(question)` explicitly zero-initializes counters and empty-initializes trace/provenance fields, so each run starts from a deterministic state shape.
- `set_plan(...)`, `set_evidence(...)`, `set_answer(...)`, and `set_final_answer(...)` are the only mutation helpers used by nodes for major state transitions.
- Every helper appends a structured trace event (`type=plan/evidence/answer/final_answer`) through `add_trace(...)`, guaranteeing trace consistency.

### 3.5 `tests/test_lc_graph.py`

Why changed:

- Needed direct coverage for iterative behavior, budget stop semantics, and refusal diagnostics.

What changed:

- Added/updated tests for:
  - refine-then-answer multi-round behavior
  - compare-doc-diversity sufficiency gate
  - tool-budget stop with no answer LLM call
  - round-budget stop behavior
  - `stop_reason` vs `refusal_reason` correctness when evidence was sufficient but citations missing

Concrete implementation mechanisms:

- `test_loop_refines_then_answers` monkeypatches retrieval tool to return sparse then enriched hits, validating two retrieval rounds and final cited answer path.
- `test_assess_compare_requires_doc_diversity` asserts compare questions fail sufficiency when evidence comes from one doc only.
- `test_budget_stop_refuses_without_calling_answer_llm` monkeypatches `_call_rag_answer` to crash if called; asserts budget stop prevents answer LLM invocation.
- `test_round_limit_stops_and_refuses` validates `MAX_RETRIEVAL_ROUNDS` hard-stop behavior.
- `test_verify_uses_refusal_reason_when_stop_reason_is_sufficient` verifies semantic split:
  - loop stop provenance remains `sufficient_evidence`
  - refusal reason becomes `missing_citations`.

---

## 4. Node Mechanisms and Decisions

### 4.1 `node_route`

Purpose:

- Classify query into initial plan (`retrieve`, `compare`, `resolve_definition`, etc.).
- Early-stop if step budget already exhausted.

Decision policy:

- Uses regex and heuristics for compare parsing and intent routing.

Concrete implementation mechanisms:

- Increments `steps` via `_bump_step(state, "route")` and records a step-trace event.
- If step budget is already exhausted, sets:
  - `stop_reason = "step_budget_exhausted"`
  - plan action `refuse`
  - trace event `type=loop_stop`.
- Otherwise computes deterministic plan via `_heuristic_route(question)` and stores it with `set_plan(...)`.

### 4.2 `node_retrieve`

Purpose:

- Execute one retrieval round.

Mechanics:

- Increments `tool_calls` and `retrieval_round`.
- Calls appropriate tool based on `plan.action`.
- Merges evidence deterministically (dedupe by `chunk_id`, stable order).
- Stores per-round retrieval telemetry in `last_retrieval_stats`.

Concrete implementation mechanisms:

- Enforces budget gates in this order before tool call:
  - step, tool-call, retrieval-round.
- On pass, increments:
  - `tool_calls += 1`
  - `retrieval_round += 1`
  and emits `retrieval_round_started` trace.
- Tool dispatch is action-specific:
  - `retrieve` -> `lc_tools.retrieve.invoke({"query": ..., "k": 8})`
  - `resolve_definition` -> `lc_tools.resolve_definition.invoke(...)`
  - `compare` -> `lc_tools.compare.invoke({"topic_a": ..., "topic_b": ..., "k": 6})`
  - `summarize` -> `lc_tools.summarize.invoke(...)`.
- Normalizes incoming tool evidence to `EvidenceItem`, merges with prior state evidence using `_merge_evidence(...)` (first-seen `chunk_id` kept), and writes with `set_evidence(...)`.
- Persists structured round telemetry (`new_hits`, `total_hits`, `tool_stats`, `mode_hint`) into `last_retrieval_stats` and trace.

### 4.3 `node_assess_evidence`

Purpose:

- Decide whether current evidence is sufficient to answer.

Mechanics:

- Computes reasons list:
  - `insufficient_hits`
  - `anchor_missing`
  - `compare_doc_diversity_missing`
- Sets `evidence_sufficient`.
- Sets `stop_reason`:
  - budget reason when budget-bound insufficient state occurs
  - first insufficiency reason otherwise
  - `sufficient_evidence` when pass condition met

Concrete implementation mechanisms:

- Converts state evidence dicts to `EvidenceItem` and computes:
  - `anchors = _extract_anchor_terms(question)`
  - `anchor_match = _evidence_contains_any_anchor(...)`
  - `doc_diversity = _doc_diversity(evidence)`.
- Constructs insufficiency reasons in deterministic order:
  - `insufficient_hits`
  - `anchor_missing`
  - `compare_doc_diversity_missing`.
- Marks `evidence_sufficient = (len(reasons) == 0)`.
- Emits full decision trace payload including reasons, budget reason, hit counts, doc diversity, anchors, and counters.

### 4.4 `node_refine_query`

Purpose:

- Produce deterministic query revision for next retrieval round.

Mechanics:

- Uses `stop_reason` + query structure to apply a refinement strategy:
  - anchor token bias
  - compare doc/topic bias
  - coverage/definition bias
- Writes new retrieval plan with updated query.

Concrete implementation mechanisms:

- Increments step and checks step budget before refinement.
- Calls `_build_refined_query(state)` to produce `(refined_query, strategy)`.
- Rebuilds plan as:
  - `action="retrieve"`
  - `reason=f"Refined retrieval query via {strategy}."`
  - updated `query` and inferred `mode_hint`.
- Emits `query_refined` trace with previous query, refined query, and strategy tag.

### 4.5 `node_answer`

Purpose:

- Call answer generator only when evidence is marked sufficient.

Mechanics:

- Skips on insufficient evidence (`answer_skip` trace).
- Converts normalized citations back to state citations.

Concrete implementation mechanisms:

- If `evidence_sufficient` is false or evidence list is empty:
  - does not call answer LLM path
  - writes `answer_skip` trace reason.
- Otherwise calls `_call_rag_answer(question, evidence)` and maps returned citation dicts into `Citation` objects (preserving optional `key`).
- Stores draft answer + citations via `set_answer(...)` (final answer still not committed here).

### 4.6 `node_verify_or_refuse`

Purpose:

- Final guardrail before output.

Mechanics:

- Refuses if any of:
  - evidence insufficient
  - empty draft
  - empty evidence
  - zero citations
- Sets `refusal_reason` explicitly.
- Keeps `stop_reason` unchanged as loop-stop provenance.
- Emits trace with both fields:
  - `stop_reason`
  - `refusal_reason`

Concrete implementation mechanisms:

- Refusal predicate is explicit boolean OR over:
  - `not evidence_sufficient`
  - empty draft answer
  - empty evidence set
  - empty citations.
- On refusal:
  - computes `refusal_reason = _derive_refusal_reason(...)`
  - clears citations (`state["citations"] = []`)
  - writes user-facing refusal into `final_answer` via `set_final_answer(...)`
  - emits `verify` trace with `result="refuse"`, `stop_reason`, `refusal_reason`.
- On pass:
  - sets `refusal_reason=""`
  - promotes `draft_answer -> final_answer`
  - emits `verify` trace with `result="ok"` and citation count.

---

## 5. `stop_reason` vs `refusal_reason` Semantics

### 5.1 Problem fixed

Previously, refusal diagnostics could be ambiguous when:

- evidence assessment succeeded (`stop_reason = "sufficient_evidence"`), but
- verification refused due to missing citations.

### 5.2 Current contract

- `stop_reason`: why the control loop stopped and proceeded to answer/verify.
- `refusal_reason`: why verify refused the final output.

Example observed in:

- `runs/agent/agent_20260219_101327_what_does_nist_say_about_pqc_for_wifi_9.json`

State outcome:

- `stop_reason = "sufficient_evidence"`
- `refusal_reason = "missing_citations"`

This split gives precise postmortem signals and avoids semantic overload.

---

## 6. Evidence from Run Artifacts

### 6.1 Historical single-pass behavior

Artifact:

- `runs/agent/agent_20260218_113844_what_are_the_differences_between_mlkem_and_mldsa.json`

Trace characteristics:

- `do_tool` node present.
- No assess/refine loop nodes.
- Straight path to answer.

### 6.2 Current bounded-controller behavior

Artifact:

- `runs/agent/agent_20260219_101327_what_does_nist_say_about_pqc_for_wifi_9.json`

Trace characteristics:

- `retrieve` + `assess_evidence` nodes present.
- Verification refusal includes both `stop_reason` and `refusal_reason`.
- Final refusal message is citation-contract compliant.

---

## 7. Validation and Test Status

Executed:

- `conda run -n eleven python -m pytest -q tests/test_lc_graph.py`

Result:

- `8 passed`

What this covers:

- iterative refinement loop
- compare diversity gate
- budget stop behavior
- anti-hallucination refusal path
- citation-key preservation
- `stop_reason`/`refusal_reason` split correctness

---

## 8. Design Tradeoffs

Benefits:

- Stronger deterministic control over agent behavior.
- Better explainability via trace events and explicit reasons.
- Reduced hallucination risk by skipping answer generation when evidence is inadequate.

Costs:

- Heuristic sufficiency logic may reject answerable questions in edge cases.
- More state fields and transitions increase controller complexity.
- Refinement remains heuristic (not learned/planned), so recall gains vary by query class.

---

## 9. Future Improvements (Recommended)

### 9.1 Evaluation-backed tuning

- Run `python -m eval.run` before/after with fixed seed/config and report deltas for:
  - Recall@k, MRR, nDCG
  - citation coverage/compliance
  - refusal rate vs answer quality

### 9.2 Better sufficiency calibration

- Replace fixed thresholds with intent-aware thresholds.
- Add confidence score from retrieval distribution entropy / top-k margin.

### 9.3 Retrieval refinement improvements

- Incorporate section metadata (`section_path`) more aggressively in refine stage.
- Add deterministic rerank features using symbol/section exact matching.

### 9.4 Compare query specialization

- Explicit dual-branch retrieval per topic with balanced quotas.
- Require paired evidence structure before answer generation.

### 9.5 Verification hardening

- Sentence-level citation alignment check (claim-to-citation coverage).
- Auto-repair pass that requests one extra retrieval round when only citation coverage fails.

### 9.6 Observability

- Aggregate trace analytics (per-node latency, refusal reason distribution, budget-hit distribution).
- Add report script to summarize run artifacts over batch evals.

---

## 10. Conclusion

The agent now matches the Week-1 bounded-controller scope materially better than the previous single-pass implementation. The controller is loop-capable, budget-bounded, refusal-safe, and easier to debug. The most impactful next step is to quantify retrieval and citation quality deltas using the eval harness and tune thresholds against that data.
