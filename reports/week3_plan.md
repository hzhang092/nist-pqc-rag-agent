# Week 3 plan — nist-pqc-rag-agent

## Starting point

Week 3 starts from the real Week 2 state, not from an idealized roadmap.

Completed in Week 2:

- Day 1: parser abstraction, chunking v2, manifest versioning, Docling/chunk-structure hardening
- Day 2: retrieval diagnostics, identifier normalization, and an intent-aware do-no-harm rerank fix

This means Week 3 should not spend its limited budget repeating chunking work. It should build on the retrieval core and close the biggest remaining ML-engineering gaps.

## Why this plan is shaped this way

The strongest employer-facing gaps now are not “more retrieval tweaks” in isolation. They are:

- deployability and packaging
- stronger agent/query-analysis behavior
- answer-side robustness
- a clean ablation story
- a scoped knowledge-organization feature
- explicit security-minded documentation

This plan stays within a one-week constraint and avoids trying to force in a full fine-tuning platform, a full knowledge graph, or a serious Kubernetes deployment.

## Main Week 3 objective

Turn the NIST PQC RAG assistant into a small deployable internal AI system that demonstrates:

- agentic RAG over technical standards documents
- inspectable and bounded retrieval / answer behavior
- better answer reliability under citation constraints
- a local/on-prem style service path
- one concise proof artifact showing before/after improvement
- an honest partial story for knowledge organization

## Success criteria

By the end of Week 3, the project should be able to show:

- one-command or near-one-command local startup for a demo path
- FastAPI endpoints for `/ask`, `/search`, and `/health`
- a graph trace that clearly shows `analyze_query -> retrieve -> assess -> optional refine -> answer -> verify/refuse`
- deterministic citation repair or explicit refusal when free-form generation fails citation checks
- one concise dated ablation report with a chosen best config
- one graph-lite artifact over standards structure
- updated README / runbook / architecture notes that are easy to defend in interviews

## Constraints and design rules

- preserve deterministic contracts and page-level citation metadata
- do not widen document scope just to look bigger
- do not claim improvements without fixed-set eval deltas
- prefer one solid demo path over multiple half-finished integrations
- keep the work aligned with technical-document characteristics: algorithms, tables, math-like content, and exact identifiers

## Priority order for Week 3

### Priority 1 — Productize the project

Add a local/on-prem style serving path, FastAPI surface, Docker packaging, and a clear runbook.

### Priority 2 — Make the agent story stronger

Add explicit query analysis, make `mode_hint` truly flow through retrieval, and improve inspectability.

### Priority 3 — Make answer-side robustness visible

Reduce avoidable refusals when evidence is already adequate.

### Priority 4 — Publish measurable proof

Write one clean ablation report instead of leaving improvements scattered across notes.

### Priority 5 — Add a scoped KG-lite feature

Add a lightweight standards-navigation layer rather than attempting a full knowledge graph.

## Day-by-day execution plan

### Day 1 — Deployment foundation: local model path + FastAPI skeleton

Goal:

Make the system look deployable rather than notebook-only.

Work:

- add a configurable local model adapter path for one practical serving option
- prefer one realistic backend only:
  - `vLLM` if GPU access is actually available
  - otherwise an OpenAI-compatible local server, Ollama, or `llama.cpp`-style path
- keep the rest of the RAG pipeline behind a stable model interface
- scaffold FastAPI endpoints:
  - `/ask`
  - `/search`
  - `/health`
- return answer, citations, refusal reason, and trace summary where appropriate
- log retrieval, rerank, and generation latency separately

Acceptance:

- at least one end-to-end query works through FastAPI
- model backend is swappable without changing pipeline logic
- latency fields are visible in traces or logs

### Day 2 — Query analysis node + mode-aware retrieval plumbing

Goal:

Make the LangGraph controller look like a real agent pipeline, not a thin retrieve-and-answer wrapper.

Work:

- add `analyze_query` as the first graph node
- emit structured fields such as:
  - `canonical_query`
  - `mode_hint` (`definition | algorithm | compare | general`)
  - `required_anchors`
  - optional document-family filters when safe
- make `mode_hint` actually flow through retrieval APIs rather than being partial or best-effort
- keep the implementation deterministic:
  - schema-constrained outputs
  - bounded variants
  - stable ordering
- reuse Week 2 retrieval lessons:
  - acronym anchors
  - identifier-safe normalization
  - algorithm / table / section anchors

Acceptance:

- traces show query analysis before retrieval
- tests cover deterministic structured analysis and bounded variants
- retrieval behavior changes in a controlled way by query mode

### Day 3 — Evidence packets + citation repair

Goal:

Fix the failure mode where retrieval is decent but the answer still fails citation requirements.

Work:

- group retrieved hits into evidence packets by document, section path, and local neighborhood
- update `assess_evidence` to reason over packet-level signals instead of only raw hit counts
- add deterministic citation-repair fallback:
  - if generation has unsupported claims or zero citations, produce a short extractive / semi-extractive cited answer
  - otherwise refuse explicitly
- make refusal reasons explicit and inspectable, for example:
  - no strong evidence
  - anchor missing
  - comparison evidence one-sided
  - citation generation failed

Acceptance:

- more answerable questions end with cited output rather than avoidable refusal
- refusal reasons are explicit in traces
- evidence packets are visible in debug output

### Day 4 — Eval pass + recruiter-visible ablation report

Goal:

Convert implementation work into one clean proof artifact.

Work:

- freeze a baseline config
- run a small, targeted ablation set on the same eval set
- recommended ablations:
  1. baseline hybrid retrieval
  2. + query analysis / mode-aware retrieval plumbing
  3. + evidence packets
  4. + citation repair
  5. + graph-lite signals if ready
- report:
  - Recall@k / MRR / nDCG
  - strict and near-page overlap
  - citation coverage / citation compliance
  - refusal rate and refusal reason breakdown
- publish one concise dated report under `reports/eval/<date>/ablation.md`

Acceptance:

- one table compares baseline vs upgraded configs
- one best configuration is selected with justification
- the report is usable in interviews and resume bullets

### Day 5 — Docker packaging + secure local demo path

Goal:

Close the packaging story.

Work:

- add Dockerfile and `docker-compose` setup for the API path
- document environment variables and model/backend assumptions cleanly
- write a short runbook for startup, index usage, and common failures
- add a security-minded checklist appropriate for this project:
  - no secrets in code
  - no secrets in logs
  - pinned dependencies where practical
  - local/offline assumptions documented
  - allowlisted document corpus
  - auditable trace logs

Acceptance:

- containerized startup is reproducible
- another reader could follow the runbook without guessing
- deployment and security assumptions are explicit

### Day 6 — KG-lite / standards navigation layer

Goal:

Partially address the knowledge-organization side of the job without derailing the week.

Work:

- extract lightweight entities and relations such as:
  - Document
  - Section
  - Algorithm
  - Table
  - Term / symbol / operation
- record simple relations such as:
  - `defined_in`
  - `appears_in`
  - `near_algorithm`
  - `referenced_by_section`
  - `same_document_as`
- use this only where it creates clear value:
  - navigation for definition questions
  - algorithm lookup
  - compare-query evidence grouping
- keep implementation simple:
  - JSON / adjacency artifact first
  - no full graph DB requirement this week

Acceptance:

- one or two flows visibly benefit from graph-lite structure
- a concrete artifact exists showing extracted relations
- README explains the scope honestly

### Day 7 — Polish and interview-facing packaging

Goal:

Make the repo easy to explain and demo.

Work:

- update README and architecture notes
- add one clear system diagram
- prepare a short demo script with:
  - one definition question
  - one algorithm question
  - one compare question
  - one insufficient-evidence / refusal case
- write a short engineering-decisions note covering:
  - why no full KG yet
  - why deterministic query analysis instead of open-ended query rewriting
  - why citation repair matters
  - why Docker/FastAPI were prioritized over more retrieval tuning alone

Acceptance:

- the repo is easy to understand quickly
- the project is easy to demo and defend in an interview

## What is intentionally not in Week 3

These are valuable, but not the highest-ROI use of one week:

- full knowledge graph construction
- serious LLM fine-tuning pipeline
- Kubernetes beyond a brief stretch mention
- GCP deployment as a main deliverable
- full MLOps stack with orchestration tools
- major parser replacement or broad corpus expansion

## Stretch goals only if ahead of schedule

- add a tiny Streamlit or Gradio demo in front of FastAPI
- add `pgvector` as a second retrieval backend
- export graph-lite output into a tiny Neo4j prototype
- add CI smoke tests for API + one retrieval query

## Skills this project will still not fully cover, and how to work toward them

### 1) Full LLM fine-tuning / embedding-model training

Why it still falls short:

- Week 3 can improve retrieval and answer quality, but it still will not honestly demonstrate substantial post-training work

How to work toward it:

- run one small LoRA/QLoRA experiment on a narrowly scoped task after Week 3
- or fine-tune a lightweight reranker / embedding model on relevance pairs from the eval set
- report dataset, objective, baseline, and before/after metrics
- use HuggingFace + PyTorch so the work maps directly to common ML engineering expectations

### 2) Real graph databases and query languages (`Neo4j`, `Cypher`, `SQL`)

Why it still falls short:

- graph-lite helps, but it is not the same as a true graph database workflow

How to work toward it:

- export graph-lite entities/relations into Neo4j after Week 3
- write 5–10 concrete Cypher queries over standards entities
- add one small demo query over algorithms/sections/operations
- separately add PostgreSQL + `pgvector` as a persistent retrieval backend

### 3) Kubernetes / cloud-native deployment / GCP

Why it still falls short:

- Docker is realistic this week; Kubernetes and cloud deployment are not the best primary use of limited time

How to work toward it:

- deploy the FastAPI container once to a small local or cloud environment
- write a minimal deployment manifest and short ops note
- keep the deployment story concrete rather than broad

### 4) Production MLOps depth

Why it still falls short:

- the project can show evaluation discipline and packaging, but not a full production ML platform

How to work toward it:

- add CI for tests and one smoke evaluation
- log retrieval and answer metrics over runs
- introduce orchestration tools only if a real scaling or scheduling need appears

### 5) Security engineering integration beyond secure-coding hygiene

Why it still falls short:

- the project can become security-aware, but it will not fully demonstrate collaboration with security engineers or enterprise security architecture

How to work toward it:

- add a short threat-model note covering prompt injection, unsafe evidence, secrets handling, and trace retention
- document trust boundaries between corpus, retriever, and model
- add dependency review / pinning and a small security checklist

### 6) Large-scale search infrastructure

Why it still falls short:

- the corpus is small and standards-focused, so it does not prove distributed or billion-scale retrieval

How to work toward it:

- later add a backend such as `pgvector`, OpenSearch, or Elasticsearch behind the same retrieval interface
- keep eval constant while swapping backends so the systems story is stronger
- describe the project honestly as architecture-ready, not scale-proven

## Interview framing

A clean way to describe this iteration is:

> I already had a citation-grounded RAG core over technical standards. In Week 3, I prioritized the gaps that mattered most for ML engineering roles: deployment, stronger agent control, answer robustness, measurable ablations, and a lightweight knowledge-organization layer. I deliberately avoided overreaching into full fine-tuning, full knowledge graphs, or a heavy cloud stack in one week.
