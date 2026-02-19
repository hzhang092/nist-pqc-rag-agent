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

### 3.4 `rag/lc/state_utils.py`

Why changed:

- New state contract must be deterministically initialized.

What changed:

- `init_state(...)` now initializes all new fields, including `refusal_reason` and loop counters.

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

---

## 4. Node Mechanisms and Decisions

### 4.1 `node_route`

Purpose:

- Classify query into initial plan (`retrieve`, `compare`, `resolve_definition`, etc.).
- Early-stop if step budget already exhausted.

Decision policy:

- Uses regex and heuristics for compare parsing and intent routing.

### 4.2 `node_retrieve`

Purpose:

- Execute one retrieval round.

Mechanics:

- Increments `tool_calls` and `retrieval_round`.
- Calls appropriate tool based on `plan.action`.
- Merges evidence deterministically (dedupe by `chunk_id`, stable order).
- Stores per-round retrieval telemetry in `last_retrieval_stats`.

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

### 4.4 `node_refine_query`

Purpose:

- Produce deterministic query revision for next retrieval round.

Mechanics:

- Uses `stop_reason` + query structure to apply a refinement strategy:
  - anchor token bias
  - compare doc/topic bias
  - coverage/definition bias
- Writes new retrieval plan with updated query.

### 4.5 `node_answer`

Purpose:

- Call answer generator only when evidence is marked sufficient.

Mechanics:

- Skips on insufficient evidence (`answer_skip` trace).
- Converts normalized citations back to state citations.

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
