# Evaluation Summary

- generated_at_utc: 2026-03-07T16:52:42.055688+00:00
- run_id: questions_20260307T165242Z
- dataset: eval/day4/questions.jsonl
- total_questions: 24
- answerable_questions: 21
- unanswerable_questions: 3
- labeled_answerable_questions: 21
- unlabeled_answerable_questions: 0

## Retrieval
- scoring_scope: answerable_with_non_empty_gold_only
- primary_k: 8
- Recall@k: 0.6548
- MRR@k: 0.3529
- nDCG@k: 0.4147

### Retrieval By K
- k1: recall=0.1667, mrr=0.1905, ndcg=0.1905
- k3: recall=0.4048, mrr=0.3175, ndcg=0.3220
- k5: recall=0.4286, mrr=0.3175, ndcg=0.3333
- k8: recall=0.6548, mrr=0.3529, ndcg=0.4147

### Secondary Diagnostics
- near_page_tolerance: 1
- diagnostic_pre_rerank_depth: 60
- rerank_pool: 40
- k1: strict=0.1905, doc_only=0.7143, near_page=0.2381
- k3: strict=0.4762, doc_only=0.9048, near_page=0.5714
- k5: strict=0.4762, doc_only=0.9048, near_page=0.5714
- k8: strict=0.7143, doc_only=0.9048, near_page=0.7619
- miss_cause_counts: rerank_demotion:1, rerank_insufficient_promotion:3, upstream_missing:2

### Metric Definitions
- Retrieval By K:
  Recall@k = average fraction of gold spans recovered per question; MRR@k = average reciprocal first strict-hit rank; nDCG@k = rank-aware gain over unique gold spans.
- Secondary Diagnostics:
  hit-rate style metrics (per-question success rate): at least one matching hit appears in top-k.
- strict:
  doc_id match + page overlap (same relevance rule as primary retrieval metrics).
- doc_only:
  doc_id must match; page overlap is ignored.
- near_page:
  doc_id match + page overlap with +-near_page_tolerance slack.
- Why numbers differ:
  Retrieval By K is span-coverage/rank quality; Secondary Diagnostics is question-level any-hit rate, so strict can be higher when questions have multiple gold spans.

### Questions Missing Gold In Top-k
- count: 6
- q002: why do we need the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.205:p11-p11 | top_hits=r1 NIST.IR.8547.ipd p9-p9; r2 NIST.IR.8547.ipd p12-p12; r3 NIST.IR.8545 p25-p25 | diag=pre_fused_rank=20, pre_pool_rank=20, post_rank=20, delta_pre_pool_to_post=0, cause=rerank_insufficient_promotion
- q003: What are the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p3-p3, NIST.FIPS.205:p3-p3 | top_hits=r1 NIST.IR.8547.ipd p24-p24; r2 NIST.IR.8547.ipd p12-p12; r3 NIST.IR.8547.ipd p16-p16 | diag=pre_fused_rank=None, pre_pool_rank=None, post_rank=None, delta_pre_pool_to_post=None, cause=upstream_missing
- q007: give me Algorithm 2 SHAKE128 | gold=NIST.FIPS.203:p28-p28 | top_hits=r1 NIST.FIPS.204 p50-p50; r2 NIST.FIPS.205 p25-p25; r3 NIST.FIPS.203 p31-p31 | diag=pre_fused_rank=13, pre_pool_rank=13, post_rank=13, delta_pre_pool_to_post=0, cause=rerank_insufficient_promotion
- q010: Define ML-KEM encapsulation key and where it’s used. | gold=NIST.FIPS.203:p4-p4, NIST.FIPS.203:p11-p11 | top_hits=r1 NIST.FIPS.203 p44-p44; r2 NIST.FIPS.203 p3-p3; r3 NIST.FIPS.203 p10-p10 | diag=pre_fused_rank=30, pre_pool_rank=30, post_rank=30, delta_pre_pool_to_post=0, cause=rerank_insufficient_promotion
- q012: How do ML-DSA and SLH-DSA differ in intended use-cases? | gold=NIST.FIPS.204:p5-p5, NIST.FIPS.205:p5-p5 | top_hits=r1 NIST.FIPS.204 p29-p29; r2 NIST.FIPS.205 p18-p18; r3 NIST.FIPS.205 p48-p48 | diag=pre_fused_rank=None, pre_pool_rank=None, post_rank=None, delta_pre_pool_to_post=None, cause=upstream_missing
- q014: What underlying mathematical hardness assumptions form the security basis for ML-KEM, ML-DSA, and SLH-DSA? | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.204:p11-p11 | top_hits=r1 NIST.FIPS.204 p7-p7; r2 NIST.IR.8545 p8-p8; r3 NIST.FIPS.204 p7-p7 | diag=pre_fused_rank=16, pre_pool_rank=16, post_rank=18, delta_pre_pool_to_post=2, cause=rerank_demotion

## Answer
- enabled: False
- model_dependent: True
- note: Answer metrics are model-dependent and less stable than retrieval metrics; use retrieval metrics as primary regression signals.
- answer_evaluated: 0
- citation_presence_rate: n/a
- inline_citation_sentence_rate: n/a
- refusal_accuracy: n/a
