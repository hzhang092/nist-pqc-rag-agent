# Project Overview — nist-pqc-rag-agent

## Why this exists

This repo builds a citation-grounded, agentic RAG assistant over NIST Post-Quantum Cryptography (PQC) documents. It answers questions with page-cited evidence, can use tools (retrieve, summarize, compare), and includes an evaluation harness so changes are measured rather than described informally.

The project is intentionally scoped to align with AI/ML roles that emphasize agentic RAG, retrieval quality, deterministic pipelines, NLP/data processing, and evaluation discipline.

## Timeline anchor

### Week 1 goal — working citation-grounded RAG baseline

Build a compact, recruiter-legible mini system with:

- deterministic ingestion, cleaning, and chunking
- hybrid retrieval (`FAISS + BM25 + query fusion + RRF`)
- citation-first answer generation with refusal behavior
- a bounded LangGraph controller
- a reproducible evaluation baseline

### Week 2 progress — completed work actually shipped

Week 2 should be described as completed progress, not as a future roadmap.

#### Day 1 — ingestion / chunking upgrade

Implemented and validated:

- parser abstraction with dual backends (`llamaparse` and `docling`)
- markdown-aware chunking v2 behind `CHUNKER_VERSION`
- manifest-based artifact versioning and retriever compatibility checks
- deterministic ingestion controls, preflight checks, and chunk-structure hardening

Important status note:

- the safe default parser remains `llamaparse`
- chunking v2 exists and was hardened, but conservative rollout principles still apply when selecting defaults

#### Day 2 — retrieval recovery and rerank fix

Implemented and validated:

- stage-aware retrieval diagnostics to localize where gold evidence is lost
- identifier-scoped normalization for index/query symmetry
- deterministic, intent-aware do-no-harm reranker
- stricter rerank diagnostics and safer default behavior

Best recovered run from Week 2 Day 2:

- Recall@8: `0.6548`
- nDCG@8: `0.4334`
- MRR@8: `0.3755`

Recommended Week 2 retrieval default from the report:

- keep the intent-aware do-no-harm reranker enabled
- keep mode-aware query-variant expansion disabled by default unless revalidated under current pool limits

### Week 3 goal — turn the project into a small deployable internal AI system

Week 3 is the forward-looking plan.

Primary objective:

Turn the NIST PQC RAG assistant into a more recruiter-legible ML engineering project by adding:

- a local/on-prem style serving path
- FastAPI endpoints and Docker packaging
- stronger agent query analysis and retrieval plumbing
- answer-side citation repair / refusal robustness
- one concise before/after ablation artifact
- a scoped graph-lite knowledge organization layer

## Current project state after Week 2

What is already demonstrated:

- deterministic ingestion -> cleaning -> chunking -> indexing
- hybrid retrieval with fusion and measured rerank recovery
- citation-first answering with refusal guardrails
- bounded LangGraph orchestration
- reproducible retrieval evaluation with diagnostics
- structure-aware processing tailored to standards-style technical PDFs

What is still under-signaled relative to stronger ML engineering roles:

- deployment and packaging
- API surface and demo ergonomics
- explicit query-analysis node and better agent traces
- answer-side robustness when retrieval succeeds but citation generation fails
- graph / knowledge-organization story
- security-adjacent runbook and deployment assumptions

## Why this corpus is a good RAG testbed

The NIST PQC corpus is not generic prose. It contains:

- dense technical definitions
- algorithms and stepwise procedures
- tables, notation, and math-like spans
- exact identifiers and operation names
- cross-references by section, table, and algorithm number

That means the project needs:

- structure-aware chunking
- lexical + semantic retrieval together
- careful identifier handling
- citation-preserving page metadata
- evaluation that distinguishes document-family retrieval from exact localization

## Core user stories

- As a user, I can ask: “What does FIPS 203 specify for ML-KEM key generation?” and receive a concise cited answer.
- As a user, I can ask: “How do ML-DSA and SLH-DSA differ in intended use-cases?” and receive a structured cited comparison.
- As a developer, I can run eval and inspect whether a change improved retrieval and answer behavior on a fixed set.
- As an engineer, I can inspect deterministic artifacts and trace why the system answered or refused.

## Document scope

Included in core scope:

- FIPS 203 (ML-KEM)
- FIPS 204 (ML-DSA)
- FIPS 205 (SLH-DSA)
- SP 800-227
- NIST IR 8545
- NIST IR 8547

Optional documents should remain outside default eval scope unless explicitly promoted.

## Non-goals

These remain outside the honest current scope:

- cryptographic proof-level guarantees
- full LLM fine-tuning pipeline
- full production auth / multi-tenant system design
- perfect PDF parsing across every edge case
- full knowledge graph platform
- cloud-native production deployment in the current state

## System architecture (high level)

Pipeline:

`PDFs -> parse -> clean -> structured chunks -> embed/index -> hybrid retrieval -> citation-first generation -> verify/refuse -> evaluation`

### LangChain / LangGraph role

LangChain is used for:

- model and structured-output interfaces
- tool definitions
- retriever and agent plumbing where useful

LangGraph is used for a bounded controller:

- retrieve -> assess evidence -> optional refine/retrieve -> answer -> verify/refuse

Design principle:

- ingestion, chunking, and evaluation stay framework-independent through deterministic JSONL contracts
- the agent layer is bounded and inspectable rather than open-ended

## Implemented components after Week 2

### 1) Ingestion (deterministic, parser-abstracted)

- input: raw PDFs
- output: `data/processed/pages.jsonl`
- parser abstraction supports `llamaparse` and `docling`
- current conservative default remains `llamaparse`, with `docling` available for controlled use and ablation
- page-count and page-coverage checks protect citation mapping

### 2) Cleaning

- removes repeated boilerplate and whitespace issues
- preserves technical content needed for retrieval

### 3) Chunking

- page-level citation spans are preserved on every chunk
- v1 remains available for stable fallback
- v2 adds structure-aware splitting and metadata such as `section_path`, `block_type`, and `chunker_version`
- chunking work is tailored to technical PDFs containing algorithms, tables, and math-like content

### 4) Indexes

- FAISS vector index for semantic retrieval
- BM25 artifacts for lexical matching
- chunk store with stable chunk/page metadata

### 5) Retrieval

- deterministic query fusion + RRF
- optional reranking now upgraded with a do-no-harm, intent-aware policy
- retrieval diagnostics can identify whether misses occur upstream, outside rerank pool, or during reranking

### 6) Answer generation

- citation-first answers
- explicit refusal when support is insufficient or citation requirements fail
- answer reliability is still stronger on retrieval than on answer-side recovery, which is a Week 3 target

### 7) Agent orchestration

- bounded LangGraph controller
- explicit budgets for steps / retrieval rounds / tool calls
- inspectable traces and refusal reasons
- further query-analysis strengthening is planned for Week 3

## Evaluation harness

The eval harness supports:

- Recall@k, MRR, nDCG
- strict and near-page retrieval checks
- per-question diagnostics
- fixed-set before/after comparisons

Week 2 changed the project from “we think reranking is the issue” to “we can localize failure stages and show measured recovery.”

## Milestone status

### Milestone 1 — Deterministic ingestion / cleaning / chunking / indexing

Status: complete

### Milestone 2 — Citation-first RAG answers

Status: complete at baseline level

### Milestone 3 — Bounded LangGraph controller

Status: complete at baseline level

### Milestone 4 — Week 2 upgrade

Status: complete for the two shipped items:

- structure-aware chunking / ingestion hardening
- retrieval recovery via rerank diagnostics and fix

### Milestone 5 — Week 3 upgrade

Status: next iteration

See `week3_plan.md` for the forward-looking execution plan.

## Week 3 roadmap summary

Week 3 should prioritize the highest-signal gaps for ML engineering roles within one week:

1. productize the system with FastAPI + Docker + a local/on-prem style serving path
2. strengthen the agent story with explicit query analysis and real `mode_hint` plumbing
3. improve answer-side robustness with citation repair and clearer refusal categories
4. publish one concise ablation report
5. add a scoped graph-lite layer for knowledge organization

## What this project still will not fully cover, and how to work toward it

These are the major skill gaps that will still remain even after a strong Week 3.

### 1) Full LLM fine-tuning / embedding-model training

The project can show strong retrieval and answer engineering, but it still will not honestly represent substantial post-training work by itself.

Practical next step:

- run one narrow LoRA/QLoRA or reranker fine-tuning experiment on a fixed eval-derived task
- report dataset, objective, baseline, and measured delta

### 2) Real graph databases and query languages (`Neo4j`, `Cypher`, `SQL`)

A graph-lite layer is useful, but it is not the same as a true graph database workflow.

Practical next step:

- export graph-lite entities/relations into Neo4j
- write a small set of Cypher queries over standards entities
- add PostgreSQL + pgvector as a persistent retrieval backend to strengthen SQL relevance

### 3) Docker / Kubernetes / cloud-native deployment depth

Docker is realistic in the near term. Full Kubernetes or GCP depth is not yet honestly covered.

Practical next step:

- deploy the API container once to a minimal local or cloud environment
- keep the story concrete and reproducible instead of broad

### 4) Production MLOps depth

This project can show evaluation discipline, packaging, and clean contracts, but not a full MLOps platform.

Practical next step:

- add CI smoke tests
- log retrieval/answer metrics
- introduce workflow tooling only when there is a real operational need

### 5) Security engineering integration beyond basic secure-coding hygiene

The project is domain-aligned with PQC and security, but that is not the same as proving deep security-systems integration.

Practical next step:

- add a short threat-model note
- document trust boundaries, secret handling, and evidence-source assumptions
- add a simple repo security checklist
