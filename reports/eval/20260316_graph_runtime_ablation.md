# Graph Runtime Ablation

- generated_at: 2026-03-17 02:14 UTC
- dataset: `eval/graph_runtime_ablation.jsonl`
- primary_k: 8
- ks: `1,3,5,8`
- agent_planner_mode: deterministic fallback pinned during eval
- note: `/ask` remains the control path; graph section priors only affect `/ask-agent` rerank behavior

## Summary

| config | Recall@k | MRR@k | nDCG@k | citation compliance | refusal rate | answer modes |
| --- | --- | --- | --- | --- | --- | --- |
| 1. /ask baseline | 1.0000 | 0.7167 | 0.7937 | 1.0000 | 0.0000 | deterministic=12 |
| 2. /ask-agent + graph lookup | 0.9000 | 0.6667 | 0.7223 | 0.9000 | 0.1667 | deterministic=12 |
| 3. /ask-agent + graph lookup + section priors | 1.0000 | 0.9000 | 0.9262 | 1.0000 | 0.0833 | deterministic=12 |

## Default

- selected_default: 3. /ask-agent + graph lookup + section priors
- rationale: best retrieval score at the primary k, with citation compliance used as a tie-breaker.

## Examples

### graph-runtime-001: What are the steps in Algorithm 19?

- 1. /ask baseline: first_gold_rank=3; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p9-p9; r2 NIST.FIPS.204 p9-p9; r3 NIST.FIPS.203 p44-p44
- 2. /ask-agent + graph lookup: first_gold_rank=2; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p9-p9; r2 NIST.FIPS.205 p45-p45; r3 NIST.FIPS.203 p35-p35; graph=algorithm_number/exact_algorithm_number; section_priors=False
- 3. /ask-agent + graph lookup + section priors: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.205 p45-p45; r2 NIST.FIPS.203 p44-p44; r3 NIST.FIPS.203 p44-p44; graph=algorithm_number/exact_algorithm_number; section_priors=True

### graph-runtime-002: What are the steps in Algorithm 19 in FIPS 203?

- 1. /ask baseline: first_gold_rank=2; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p9-p9; r2 NIST.FIPS.203 p44-p44; r3 NIST.SP.800-227 p43-p43
- 2. /ask-agent + graph lookup: first_gold_rank=miss; refusal=True; citations=0; top_hits=r1 NIST.FIPS.203 p6-p6; r2 NIST.FIPS.203 p4-p4; r3 NIST.FIPS.203 p56-p56; graph=algorithm_number/exact_algorithm_number; section_priors=False
- 3. /ask-agent + graph lookup + section priors: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p44-p44; r2 NIST.FIPS.203 p6-p6; r3 NIST.FIPS.203 p4-p4; graph=algorithm_number/exact_algorithm_number; section_priors=True

### graph-runtime-007: What does ML-DSA.KeyGen do?

- 1. /ask baseline: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.204 p27-p27; r2 NIST.FIPS.204 p19-p19; r3 NIST.FIPS.204 p23-p23
- 2. /ask-agent + graph lookup: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.204 p27-p27; r2 NIST.FIPS.204 p19-p19; r3 NIST.FIPS.204 p27-p27; graph=algorithm_name/term_fallback:exact_normalized_term; section_priors=False
- 3. /ask-agent + graph lookup + section priors: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.204 p27-p27; r2 NIST.FIPS.204 p19-p19; r3 NIST.FIPS.204 p27-p27; graph=algorithm_name/term_fallback:exact_normalized_term; section_priors=True

### graph-runtime-009: What is little-endian?

- 1. /ask baseline: first_gold_rank=2; refusal=False; citations=2; top_hits=r1 NIST.FIPS.204 p38-p38; r2 NIST.FIPS.203 p12-p12; r3 NIST.FIPS.204 p39-p39
- 2. /ask-agent + graph lookup: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p12-p12; r2 NIST.FIPS.203 p12-p12; r3 NIST.FIPS.203 p30-p30; graph=term/exact_normalized_term; section_priors=False
- 3. /ask-agent + graph lookup + section priors: first_gold_rank=1; refusal=False; citations=2; top_hits=r1 NIST.FIPS.203 p12-p12; r2 NIST.FIPS.203 p12-p12; r3 NIST.FIPS.203 p30-p30; graph=term/exact_normalized_term; section_priors=True

## Tradeoffs

- This ablation isolates graph runtime effects by pinning the agent planner to the deterministic fallback.
- If any config used `deterministic_fallback` answer mode, answer-side metrics should be treated as lower confidence than retrieval metrics.
- Neo4j remains a dev/debug/export path only; this runtime ablation uses the checked-in graph-lite JSONL artifacts.
