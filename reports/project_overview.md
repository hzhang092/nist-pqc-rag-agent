# Project Overview — nist-pqc-rag-agent

## environment:
- conda environment: pyt


## Why this exists

This repo builds a citation-grounded, agentic RAG assistant over NIST Post-Quantum Cryptography (PQC) documents. It answers questions with page-cited evidence, can use tools (retrieve, summarize, compare), and ships with an evaluation harness so improvements are measurable rather than anecdotal.

The project is intentionally scoped to align with current ML engineering and applied AI expectations for agentic RAG systems: LangChain/LangGraph orchestration, tool calling, hybrid retrieval, deterministic ingestion and evaluation, citation-safe answering, and a clear path from local prototype to small deployable internal AI service.

The corpus choice is deliberate. NIST PQC standards are technical, symbol-heavy, and structure-rich documents with algorithms, tables, notation, section references, and standards-specific terminology. That makes them a strong testbed for retrieval over technical PDFs and a good match for post-quantum / security-adjacent AI roles.

### Week-1 goal: a compact, recruiter-legible “mini production system”

- deterministic ingestion + cleaning + chunking over standards PDFs,
- hybrid retrieval (FAISS + BM25 + fusion) with citation-preserving chunk metadata,
- strict citation policy (“no claim without evidence”),
- a bounded LangGraph controller (no runaway loops),
- eval suite + baseline + at least one measured retrieval improvement.

### Week-2 goal: upgrade the project into a small deployable internal AI system

Week 2 is no longer framed as “retrieval recovery only.” The goal is to keep the strong retrieval core, then close the most visible ML-engineering gaps:

- add a local/on-prem style serving path and API surface,
- strengthen the agent story with explicit query analysis and bounded refinement,
- improve answer-side robustness with citation repair / extractive fallback,
- publish clean before/after ablations for both retrieval and answer-side behavior,
- add a scoped graph-lite layer for standards navigation and knowledge organization,
- document security-minded assumptions, runbooks, and limitations honestly.

## Core user stories

- As a user, I can ask: “What does FIPS 203 specify for ML-KEM key generation?” and get a concise answer with citations (doc + page range + chunk ids).
- As a user, I can ask: “How do ML-DSA and SLH-DSA differ in intended use-cases?” and get a structured comparison with citations.
- As a user, I can inspect retrieved evidence and refusal reasons rather than receiving an unsupported answer.
- As a developer, I can run an eval suite and see whether a change improved:
  - retrieval quality (Recall@k, MRR, nDCG),
  - localization quality (strict/near-page overlap),
  - citation coverage / compliance,
  - refusal behavior and answer-side robustness.
- As an engineer, I can run everything locally (CLI) and through an API, with reproducible outputs and explicit configuration.

## Document scope

Included (current main corpus):

- FIPS 203 (ML-KEM)
- FIPS 204 (ML-DSA)
- FIPS 205 (SLH-DSA)
- SP 800-227 (migration/guidance)
- NIST IR 8545 (4th-round status)
- NIST IR 8547 (transition to PQC standards)

Optional (kept separate unless explicitly promoted):

- HQC and other extras under `raw_pdf_optional/` to avoid widening eval scope or diluting retrieval.

## Scope guardrails and non-goals

These are intentional, not omissions.

### In scope

- deterministic document ingestion and chunk generation,
- structure-aware technical retrieval,
- citation-grounded RAG answers,
- bounded tool-using agent behavior,
- local/API demo path,
- evaluation-driven iteration,
- lightweight knowledge organization over standards structure.

### Out of scope for the current project window

- cryptographic proof-level correctness guarantees,
- full fine-tuning pipeline for a base LLM,
- full production auth/rate limiting/multi-tenant storage,
- perfect PDF parsing across every standards edge case,
- full knowledge graph platform,
- full MLOps / cloud orchestration stack,
- Kubernetes or GCP as a core deliverable for Week 2.

## System architecture (high-level)

Pipeline:

PDFs → parser extraction → page text → cleaned pages → structured chunks → indexes → query analysis → hybrid retrieval → evidence packets → citation-first answer generation → verify/refuse → evaluation / API / traces

### LangChain/LangGraph role

LangChain provides standardized interfaces for:

- LLM + structured outputs,
- tool calling definitions,
- retriever wrappers / model adapters.

LangGraph provides the bounded controller as a state machine:

- analyze_query → retrieve → assess evidence → optionally refine/retrieve → answer → verify/refuse
- explicit step budget, tool-call budget, and stop rules.

Crucially, ingestion, chunking, storage artifacts, and evaluation remain framework-independent JSONL/JSON contracts. That keeps the system inspectable, deterministic, and easier to test.

## Components

### 1) Ingestion (deterministic, parser-backed)

- Input: `raw_pdf/*.pdf`
- Output: `data/processed/pages.jsonl`
- Parsing: parser abstraction with Docling and LlamaParse backends

Guarantees:

- page-level records: `doc_id`, `page_number`, `text`
- deterministic ordering
- page-count sanity checks
- warnings for empty-page spikes / parser anomalies
- explicit fallback behavior instead of silent page drops
- manifest/version metadata for artifact traceability

### 2) Cleaning

Removes repeated headers/footers, fixes whitespace/hyphenation, and preserves technical semantics.

- Output: `pages_clean.jsonl`

### 3) Chunking (structure-aware; citations preserved)

Converts cleaned pages into retrieval units while preserving page spans.

Current direction:

- target chunk size ~250–400 tokens with overlap,
- prefer headings / algorithm blocks / tables / lists when detectable,
- preserve `start_page` / `end_page` for every chunk,
- attach optional `section_path`, `block_type`, and chunker version metadata,
- allow embedding-time breadcrumb context without mutating evidence text.

- Output: `chunks.jsonl`

### 4) Indexes (hybrid by default)

Standards PDFs benefit heavily from hybrid retrieval:

- BM25 (lexical) for exact tokens, symbols, section numbers, and algorithm names,
- vector retrieval for semantic recall,
- deterministic fusion / rerank behavior for stable evaluation.

Outputs:

- vector index (FAISS in the current local backend),
- BM25 artifacts,
- chunk store mapping `chunk_id -> metadata/text`.

### 5) Retrieval layer (fusion + mode-aware rerank)

Given a query:

- optionally normalize / canonicalize technical phrasing,
- generate deterministic query variants under bounded rules,
- retrieve via BM25 + vector backend,
- merge results with Reciprocal Rank Fusion (RRF),
- apply a do-no-harm, intent-aware rerank policy when enabled,
- return citation-bearing evidence objects.

This retrieval layer is designed for technical PDFs where exact anchors such as `Algorithm 2`, `Section 6.2`, `ML-KEM.Decaps`, or acronym definitions often matter as much as general semantics.

### 6) Query analysis layer

The agent begins with explicit query analysis rather than immediately retrieving.

Structured outputs include:

- `canonical_query`
- `mode_hint` (`definition`, `algorithm`, `compare`, `general`)
- `required_anchors` (for example Algorithm/Table/Section references or operation names)
- optional filters / document-family hints when justified.

Design constraints:

- deterministic / schema-constrained where possible,
- bounded variant generation,
- no open-ended query rewriting loops.

### 7) Evidence packets and standards-aware context grouping

Retrieved hits can be grouped into evidence packets to improve answer quality.

Packeting may combine:

- same `doc_id`,
- nearby `section_path`,
- neighboring chunks/pages when an algorithm, definition, or comparison spans local context.

This helps keep standards-style evidence together instead of treating all top-k chunks as isolated fragments.

### 8) RAG answer generator (citation-first)

Takes query + evidence and produces:

- answer (concise, structured),
- citations: `{doc_id, start_page, end_page, chunk_id}`,
- optional notes about what was or was not supported.

Policy:

- no factual claim without citation,
- post-check for unsupported or uncited sentences,
- deterministic citation-repair / extractive fallback when free-form generation fails,
- refusal when reliable citation support is not available.

### 9) Agent layer (bounded LangGraph controller)

Primary graph shape:

- `analyze_query`
- `retrieve`
- `assess_evidence`
- optional `refine_query`
- `answer`
- `verify_or_refuse`

Tooling direction:

- `retrieve(query, k, filters)` → hybrid retriever
- `summarize(doc_id, page_range)` → citation-grounded summary
- `compare(topic_a, topic_b)` → structured comparison + citations
- `resolve_definition(term_or_symbol)` → definition/notation-biased retrieval

Graph policy:

- explicit step limit, tool-call limit, and retrieval-round limit,
- explicit refusal reasons,
- no silent fallback from unsupported answering to uncited prose.

### 10) Service surface and packaging

Target Week-2 service surface:

- `POST /ask` → answer, citations, refusal reason, trace summary
- `GET /search` → retrieved chunks and metadata for debugging/demo
- `GET /health` → service sanity check

Packaging direction:

- local/on-prem style generation backend abstraction,
- FastAPI service entrypoint,
- Dockerized run path,
- reproducible runbook.

### 11) Graph-lite knowledge organization layer

This is intentionally scoped smaller than a full knowledge graph.

Planned graph-lite entities:

- `Document`
- `Section`
- `Algorithm`
- `Table`
- `Term / symbol / operation`

Planned lightweight relations:

- `defined_in`
- `appears_in`
- `referenced_by_section`
- `near_algorithm`
- `same_document_as`

Purpose:

- better standards navigation,
- stronger definition / algorithm retrieval support,
- clearer comparison evidence grouping,
- partial demonstration of intelligent data organization without overclaiming a full KG platform.

## Evaluation harness

A labeled set of technical questions with expected doc/page targets.

Metrics:

- Retrieval: Recall@k, MRR, nDCG
- Localization: strict page overlap, near-page overlap, doc-only hit rate
- Answer-side: citation coverage, citation compliance, refusal rate, refusal reason breakdown
- Faithfulness proxy: claims supported by retrieved evidence

Outputs:

- JSON report
- Markdown summary
- baseline / ablation snapshots for regression comparison

Principle:

- do not claim improvement without eval deltas on a fixed comparison set.

## Repo structure (recommended / evolving)

```text
nist-pqc-rag-agent/
  data/
    processed/
    raw_pdf/
    raw_pdf_optional/

  rag/
    config.py
    ingest.py
    clean.py
    chunk.py
    embed.py
    index_faiss.py
    search_faiss.py
    index_bm25.py
    retrieve.py
    rag_answer.py

    parsers/
      base.py
      factory.py
      llamaparse_backend.py
      docling_backend.py

    retriever/
      base.py
      factory.py
      faiss_retriever.py
      pgvector_retriever.py   # later / optional
      chroma_retriever.py     # later / optional

    lc/
      llm.py
      tools.py
      graph.py
      state.py
      trace.py

  eval/
    dataset.py
    metrics.py
    run.py

  api/
    main.py                  # FastAPI service surface

  scripts/
  tests/
  README.md
  pyproject.toml (or requirements.txt)
```

## Interfaces and data contracts

### `pages.jsonl` record

- `doc_id` (string)
- `source_path` (string)
- `page_number` (int, 1-indexed)
- `text` (string)
- optional: `markdown`, `parser_backend`

### `chunks.jsonl` record

- `chunk_id` (stable / deterministic)
- `doc_id`
- `source_path`
- `start_page`, `end_page` (int)
- `text` (string)
- optional: `section_path`, `block_type`, `chunker_version`, `token_est`

### Retrieval return object

- `chunk_id`
- `score`
- `doc_id`
- `start_page`, `end_page`
- `text` (or preview)
- optional: `highlights`, `matched_terms`, packet / trace metadata

### Final answer format

- `answer` (string)
- `citations` (list of `{doc_id, start_page, end_page, chunk_id}`)
- optional:
  - `notes`
  - `refusal_reason`
  - `trace_summary`

## Milestones

### Milestone 1 — Deterministic ingestion / chunk / index

```bash
python -m rag.ingest
python -m rag.chunk
python -m rag.index_faiss
python -m rag.index_bm25
python -m rag.search_faiss "ML-KEM key generation"
```

Acceptance:

- page spans correct and stable
- relevant results for sanity queries
- artifact/version checks are explicit

### Milestone 2 — Citation-grounded RAG answers

```bash
python -m rag.ask "What is ML-KEM?"
```

Acceptance:

- cited answer or explicit refusal
- no unsupported factual prose

### Milestone 3 — Bounded agent behavior

Acceptance:

- multi-step questions can trigger bounded retrieval refinement
- traces show analyze → retrieve → assess → answer/refuse behavior
- budgets prevent runaway loops

### Milestone 4 — Evaluation baseline and diagnostics

Acceptance:

- stable baseline reports across runs
- per-question diagnostics are inspectable

### Milestone 5 — Measured improvement

Acceptance:

- before/after report shows improvement in retrieval and/or answer-side quality on the same eval set

### Milestone 6 — Packaging and demoability

Acceptance:

- API surface works
- local serving path is reproducible
- Docker runbook exists

## Week-2 roadmap (tailored next iteration)

Timeline anchor: Week 2 starts on Tuesday, March 3, 2026.

### Scope guardrails (carry forward)

- Keep deterministic ingestion/chunking and page-level citation contracts intact.
- Keep optional PDFs out of default eval scope unless explicitly promoted.
- Prefer measurable upgrades with clear employer signal over broad experimental sprawl.
- Do not overclaim: no full KG, no full fine-tuning platform, no full cloud-native stack in one week.

### Week-2 demo deliverables

1. Local/on-prem style serving path with reproducible startup.
2. FastAPI endpoints for `/ask`, `/search`, and `/health`.
3. Bounded LangGraph trace showing `analyze_query -> retrieve -> assess -> optional refine -> answer/refuse`.
4. Stronger citation-safe answering, including deterministic fallback behavior.
5. A concise ablation report with baseline vs upgraded configs.
6. A small graph-lite artifact over standards entities/relations.
7. Updated README / runbook / architecture notes.

### Day-by-day execution plan

#### Day 1 — Deployment foundation: local model path + FastAPI skeleton

- add a local model adapter path for one practical serving option,
- scaffold `/ask`, `/search`, `/health`,
- expose timing hooks for retrieval / rerank / generation.

Acceptance:

- one end-to-end query works through FastAPI
- backend is configurable without changing pipeline code

#### Day 2 — Query analysis node + mode-aware retrieval plumbing

- add `analyze_query` as the first graph node,
- emit structured query-analysis fields,
- ensure `mode_hint` actually flows through retrieval APIs,
- keep variants deterministic and bounded.

Acceptance:

- traces clearly show query analysis before retrieval
- tests cover structured analysis and bounded variants

#### Day 3 — Evidence packets + citation repair

- group hits into evidence packets,
- update `assess_evidence` to reason over packet-level signals,
- add deterministic citation-repair / extractive fallback,
- make refusal reasons explicit and inspectable.

Acceptance:

- more answerable questions end with cited answers rather than avoidable refusals

#### Day 4 — Eval pass + recruiter-visible ablation report

- freeze a baseline config,
- run a small targeted ablation set,
- compare retrieval + answer-side metrics,
- publish one concise dated report.

Acceptance:

- one table clearly compares baseline vs upgraded configs
- one best config is chosen with justification

#### Day 5 — Docker packaging + secure local demo path

- add Dockerfile / compose setup,
- document environment variables and secrets handling,
- add a simple security-minded runbook and assumptions note.

Acceptance:

- reproducible containerized startup
- deployment / security assumptions are explicit

#### Day 6 — KG-lite / standards navigation layer

- extract lightweight entities and relations,
- use them where they add retrieval or navigation value,
- keep implementation simple (JSON / adjacency store first).

Acceptance:

- one or two flows visibly benefit from graph-lite structure
- README explains the scope honestly

#### Day 7 — Polish and interview-facing packaging

- update README and architecture notes,
- add one system diagram,
- prepare a short demo script,
- document engineering tradeoffs and remaining gaps.

Acceptance:

- the repo is easy to understand and demo quickly

## Configuration knobs (centralized)

In `rag/config.py`:

- `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP`
- `TOP_K`, `FUSION_NUM_QUERIES`, `RRF_K`
- embedding model name
- `VECTOR_BACKEND` (`faiss` now; later `pgvector` or `chroma`)
- BM25 on/off
- reranker on/off
- `PARSER_BACKEND`
- local/API model backend selection
- paths (`raw_pdf`, `data/processed`)
- agent budgets / refusal thresholds

## Engineering guidelines

- Determinism: stable chunk IDs, stable ordering, stable eval comparisons.
- No silent failures: explicit warnings for parser mismatches, stale artifacts, and citation failures.
- Security hygiene: no secrets in code or logs; document trust boundaries and local/offline assumptions.
- Documentation: every module begins with a short contract comment where useful.
- Evaluation-first iteration: no improvement claims without fixed-set comparisons.

Minimum tests:

- ingest produces non-empty pages
- chunks preserve valid page spans
- retrieval returns stable results under fixed settings
- answer format always includes citations or an explicit refusal
- graph remains bounded under multi-step queries

## What this project still will not fully cover, and how to work toward it

These are the most important remaining gaps after the Week-2 upgrade.

### 1) Full LLM fine-tuning / embedding-model training

This project can show strong retrieval and answer engineering, but it will still not honestly represent substantial post-training work by itself.

Practical next step:

- run one small LoRA/QLoRA or reranker fine-tuning experiment on a narrow task,
- use HuggingFace + PyTorch,
- document training objective, data, and before/after metrics.

### 2) Real graph databases and query languages (`Neo4j` / `Cypher` / SQL)

Graph-lite improves knowledge organization, but it is not the same as a full graph DB workflow.

Practical next step:

- export graph-lite entities/relations to Neo4j,
- write a small set of concrete Cypher queries,
- optionally add PostgreSQL + pgvector as a persistent backend.

### 3) Kubernetes / cloud-native deployment / GCP

Docker is realistic for this project window; Kubernetes and cloud deployment are not central Week-2 deliverables.

Practical next step:

- deploy the FastAPI container once to a small k3d/minikube or GCP environment,
- write one minimal deployment manifest and ops note.

### 4) Production MLOps depth

The repo can show evaluation discipline and packaging, but not a full production ML operations stack.

Practical next step:

- add CI for tests + a smoke eval,
- log retrieval / answer metrics over runs,
- add workflow tooling only when a real scaling need appears.

### 5) Deeper security engineering integration

The project is domain-aligned with PQC and can be security-aware, but it will not fully demonstrate enterprise security integration on its own.

Practical next step:

- add a short threat-model note,
- document trust boundaries between corpus, retriever, and model,
- add a small allowlisting / source-trust feature.

### 6) Large-scale search infrastructure

The current corpus is small and standards-focused, so the system is architecture-ready rather than scale-proven.

Practical next step:

- keep the retrieval interface stable,
- later add `pgvector`, OpenSearch, or Elasticsearch behind that interface,
- preserve eval comparability while swapping backends.

## Suggested README contents (recruiter-friendly)

- 10-second description + screenshot/GIF of `/ask` output with citations
- architecture diagram (pipeline + LangGraph controller + API)
- how to run (CLI + API + Docker)
- evaluation: dataset size, metrics, baseline, and what changed them
- limitations + next steps (fine-tuning, graph DB, deployment depth, MLOps)

A concise intro line that helps position the repo well:

> Agent orchestration is implemented with LangGraph as a bounded tool-using controller, while ingestion, chunking, artifacts, and evaluation remain deterministic and framework-independent.

For demo ergonomics, prefer a local model path when practical so the project is not blocked by API keys during interviews or reviews.
