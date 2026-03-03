# Evaluation Summary

- generated_at_utc: 2026-03-03T16:57:54.217790+00:00
- run_id: questions_20260303T165754Z
- dataset: eval\day4\questions.jsonl
- total_questions: 24
- answerable_questions: 21
- unanswerable_questions: 3
- labeled_answerable_questions: 21
- unlabeled_answerable_questions: 0

## Retrieval
- scoring_scope: answerable_with_non_empty_gold_only
- primary_k: 8
- Recall@k: 0.5000
- MRR@k: 0.3829
- nDCG@k: 0.3955

### Retrieval By K
- k1: recall=0.2619, mrr=0.2857, ndcg=0.2857
- k3: recall=0.3810, mrr=0.3571, ndcg=0.3488
- k5: recall=0.4286, mrr=0.3690, ndcg=0.3693
- k8: recall=0.5000, mrr=0.3829, ndcg=0.3955

### Secondary Diagnostics
- near_page_tolerance: 1
- diagnostic_pre_rerank_depth: 60
- rerank_pool: 40
- k1: strict=0.2857, doc_only=0.6667, near_page=0.3333
- k3: strict=0.4286, doc_only=0.8095, near_page=0.5238
- k5: strict=0.4762, doc_only=0.9048, near_page=0.5714
- k8: strict=0.5714, doc_only=0.9048, near_page=0.7143
- miss_cause_counts: rerank_suppression:7, upstream_missing:2

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
- count: 9
- q002: why do we need the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.205:p11-p11 | top_hits=r1 NIST.IR.8547.ipd p12-p12; r2 NIST.IR.8545 p25-p25; r3 NIST.IR.8547.ipd p12-p12 | diag=pre_fused_rank=22, pre_pool_rank=22, post_rank=34, cause=rerank_suppression
- q003: What are the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p3-p3, NIST.FIPS.205:p3-p3 | top_hits=r1 NIST.IR.8547.ipd p16-p16; r2 NIST.IR.8547.ipd p18-p18; r3 NIST.IR.8547.ipd p12-p12 | diag=pre_fused_rank=None, pre_pool_rank=None, post_rank=None, cause=upstream_missing
- q009: What are the difference between ML-KEM and ML-DSA? | gold=NIST.FIPS.203:p3-p3, NIST.FIPS.203:p21-p21, NIST.FIPS.204:p3-p4, NIST.FIPS.204:p19-p19 | top_hits=r1 NIST.SP.800-227 p43-p43; r2 NIST.FIPS.204 p38-p38; r3 NIST.FIPS.204 p8-p8 | diag=pre_fused_rank=6, pre_pool_rank=6, post_rank=13, cause=rerank_suppression
- q010: Define ML-KEM encapsulation key and where it’s used. | gold=NIST.FIPS.203:p4-p4, NIST.FIPS.203:p11-p11 | top_hits=r1 NIST.FIPS.203 p44-p44; r2 NIST.FIPS.203 p3-p3; r3 NIST.SP.800-227 p16-p16 | diag=pre_fused_rank=30, pre_pool_rank=30, post_rank=36, cause=rerank_suppression
- q012: How do ML-DSA and SLH-DSA differ in intended use-cases? | gold=NIST.FIPS.204:p5-p5, NIST.FIPS.205:p5-p5 | top_hits=r1 NIST.FIPS.205 p18-p18; r2 NIST.FIPS.205 p48-p48; r3 NIST.FIPS.205 p47-p47 | diag=pre_fused_rank=None, pre_pool_rank=None, post_rank=None, cause=upstream_missing
- q014: What underlying mathematical hardness assumptions form the security basis for ML-KEM, ML-DSA, and SLH-DSA? | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.204:p11-p11 | top_hits=r1 NIST.FIPS.203 p22-p22; r2 NIST.IR.8545 p8-p8; r3 NIST.FIPS.204 p7-p7 | diag=pre_fused_rank=23, pre_pool_rank=23, post_rank=16, cause=rerank_suppression
- q019: Which specific message authentication code (MAC) algorithms are approved for use during the key confirmation process of a KEM? | gold=NIST.SP.800-227:p11-p11 | top_hits=r1 NIST.SP.800-227 p28-p28; r2 NIST.SP.800-227 p28-p28; r3 NIST.SP.800-227 p29-p29 | diag=pre_fused_rank=5, pre_pool_rank=5, post_rank=12, cause=rerank_suppression
- q020: What are the required steps to construct a composite key-encapsulation mechanism (KEM) from two existing component KEMs? | gold=NIST.SP.800-227:p35-p36 | top_hits=r1 NIST.FIPS.203 p22-p22; r2 NIST.FIPS.203 p8-p8; r3 NIST.FIPS.203 p3-p3 | diag=pre_fused_rank=9, pre_pool_rank=9, post_rank=25, cause=rerank_suppression
- q023: What does MLWE mean? | gold=NIST.FIPS.203:p13-p13 | top_hits=r1 NIST.SP.800-227 p9-p9; r2 NIST.FIPS.203 p22-p22; r3 NIST.SP.800-227 p20-p20 | diag=pre_fused_rank=17, pre_pool_rank=17, post_rank=9, cause=rerank_suppression

## Answer
- enabled: False
- model_dependent: True
- note: Answer metrics are model-dependent and less stable than retrieval metrics; use retrieval metrics as primary regression signals.
- answer_evaluated: 0
- citation_presence_rate: n/a
- inline_citation_sentence_rate: n/a
- refusal_accuracy: n/a
