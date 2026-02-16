# Project Overview — nist-pqc-rag-agent

## Why this exists

This repo builds a citation-grounded, agentic RAG assistant over NIST Post-Quantum Cryptography (PQC) documents. It answers questions with page-cited evidence, can use tools (retrieve, summarize, compare), and ships with an evaluation harness so improvements are measurable—not vibes.

The project is intentionally scoped to align with common “AI agent / RAG / retrieval / evaluation / pipelines” expectations in current co-op roles: LangChain/LangGraph orchestration, tool calling, hybrid retrieval, vector search, deterministic pipelines, and reproducible evaluation.

### Week-1 goal: a compact, recruiter-legible “mini production system”:

- deterministic ingestion + chunking (standards PDFs are picky),
- hybrid retrieval + query fusion (high ROI),
- strict citation policy (“no claim without evidence”),
- a bounded LangGraph agent controller (no runaway loops),
- eval suite + baseline + one measured improvement.

## Core user stories

- As a user, I can ask: “What does FIPS 203 specify for ML-KEM key generation?” and get a concise answer with citations (doc + page range + chunk ids).
- As a user, I can ask: “How do ML-DSA and SLH-DSA differ in intended use-cases?” and get a structured comparison with citations.
- As a developer, I can run an eval suite and see whether a change improved:
  - retrieval quality (Recall@k, MRR, nDCG),
  - citation coverage / compliance,
  - answer faithfulness (claims supported by retrieved evidence).
- As an engineer, I can run everything locally (CLI) and optionally via an API (FastAPI), with reproducible outputs.

## Document scope (Week-1)

Included (already downloaded):

- FIPS 203 (ML-KEM)
- FIPS 204 (ML-DSA)
- FIPS 205 (SLH-DSA)
- SP 800-227 (migration/guidance)
- NIST IR 8545 (4th-round status)
- NIST IR 8547 (transition to PQC standards)

Optional (kept separate until after Week-1):

- HQC and other extras under raw_pdf_optional/ to avoid widening eval scope or diluting retrieval.

## Non-goals (Week-1)

- Cryptographic proof-level correctness guarantees (we cite sources, not prove theorems)
- Fine-tuning a base LLM
- Production auth/rate limiting/multi-tenant storage
- Perfect PDF parsing across every edge case (we add fallbacks + sanity checks)
- Full knowledge graph (possible later)

## System architecture (high-level)

Pipeline: PDFs → page text → cleaned pages → structured chunks → indexes → retrieval (hybrid + fusion + optional rerank) → generation (citation-first) → evaluation

### LangChain/LangGraph role:

LangChain provides standardized interfaces for:

- LLM + structured outputs
- tool calling definitions
- retriever wrappers (vector + BM25)

LangGraph provides the agent controller as a bounded state machine/graph:

- retrieve → assess evidence → optionally retrieve again → answer
- step budget + tool-call budget + stop rules

Crucially: ingestion, chunking, and evaluation remain framework-independent (clean JSONL contracts), so your system stays deterministic and inspectable.

## Components

1) Ingestion (deterministic)

- Input: raw_pdf/*.pdf
- Output: data/processed/pages.jsonl

Guarantees:

- page-level records: doc_id, page_number, text
- deterministic ordering
- sanity-check PDF page count vs extracted pages
- warnings for empty-page spikes

2) Cleaning

Removes repeated headers/footers, fixes whitespace/hyphenation, preserves semantics.

- Output: pages_clean.jsonl

3) Chunking (structure-aware; citations preserved)

Converts pages into retrieval units while preserving page spans.

Week-1 approach:

- target chunk size ~250–400 tokens with overlap
- prefer splitting on headings / algorithm blocks when detectable
- store start_page / end_page for every chunk
- optional section_path when available

- Output: chunks.jsonl

4) Indexes (hybrid by default)

Standards PDFs benefit heavily from hybrid retrieval:

- BM25 (lexical) for exact matches (symbols, parameter names, section numbers)
- Vector index for semantic recall

Outputs:

- vector index (FAISS in Week-1; later pgvector/Chroma)
- BM25 artifacts
- chunk store mapping chunk_id → metadata/text

### Swappable vector store backend (planned from Day 1)

We will keep the retrieval *API* stable while swapping the underlying vector index.

- Week-1 backend: FAISS (fast, local, zero services)
- Later backend (optional): PostgreSQL + pgvector (persistent, “real” VectorDB feel, plays well with deployment)
- Later backend (optional): Chroma (developer-friendly local VectorDB; nice for demos)

Design rule: “app code depends on a tiny interface, not on FAISS.”

Create a minimal interface:

- rag/retriever/base.py
- ChunkHit: {score, chunk_id, doc_id, start_page, end_page, text}
- Retriever protocol: search(query, k) -> [ChunkHit]

Implement FAISS behind that interface:

- rag/retriever/faiss_retriever.py
- loads the FAISS index from disk
- embeds queries with the same embedding model used at ingest
- returns ChunkHit objects

Later, add vector DB backends without touching your agent/eval code:

- rag/retriever/pgvector_retriever.py
- same Retriever interface
- uses SQL + pgvector similarity search

Backend selection via config:

```
VECTOR_BACKEND = "faiss"  # later: "pgvector" | "chroma"
```

Why this is employer-appealing:

- it demonstrates evolution-ready system design (interfaces + config)
- it keeps evaluation stable while you change storage/search implementations
- it’s exactly how teams migrate from local prototypes to production infrastructure

5) Retrieval layer (fusion + optional rerank)

Given a query:

- Multi-query rewriting (RAG-Fusion)
- Retrieve for each variant (BM25 + vector via the configured Retriever backend)
- Merge results with Reciprocal Rank Fusion (RRF)
- Optional rerank toggle for top-N candidates (stretch)

Returns evidence objects with citations.

6) RAG answer generator (citation-first)

Takes query + retrieved evidence and produces:

- answer (concise, structured)
- citations: {doc_id, start_page, end_page, chunk_id}

Policy:

- No claim without citation for factual statements.
- Post-check: flag/repair sentences lacking citations; if evidence is insufficient, respond with “not found in provided docs.”

7) Agent layer (LangGraph controller)

The agent is implemented as a LangGraph graph with a bounded loop.

Tools (LangChain tool calling):

- retrieve(query, k, filters) → hybrid+fusion retriever
- summarize(doc_id, page_range) → citation-grounded summary
- compare(topic_a, topic_b) → structured comparison + citations
- resolve_definition(term_or_symbol) → forces retrieval from definitions/notation first

Graph policy:

- step limit, tool-call limit, and stop conditions
- refuses to answer if citations cannot be produced

## Evaluation harness (Week-1 deliverable)

A labeled set of 30–80 questions with expected doc/page targets.

Metrics:

- Retrieval: Recall@k, MRR, nDCG
- Answer: citation coverage, formatting compliance, faithfulness proxy (sentences supported by evidence)

Outputs:

- JSON report
- Markdown summary
- baseline snapshot for regression tests

## Repo structure (recommended)

```
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

    embed.py                 # builds embeddings for chunks (or part of indexing)
    index_faiss.py           # Week-1 vector index backend
    search_faiss.py          # sanity-check search CLI
    index_bm25.py

    retriever/
      base.py                # Retriever interface + ChunkHit schema
      factory.py             # get_retriever() based on config
      faiss_retriever.py     # Week-1 backend
      pgvector_retriever.py  # later (optional)
      chroma_retriever.py    # later (optional)

    retrieve.py              # hybrid retrieval entrypoint (uses retriever + bm25)
    fusion.py                # RAG-Fusion + RRF merge
    rerank.py                # optional toggle

    rag_answer.py            # citation-first generation + post-check

    lc/                      # LangChain/LangGraph integration
      llm.py                 # model init + structured output schemas
      tools.py               # tool definitions (retrieve/summarize/compare/resolve_definition)
      graph.py               # LangGraph state machine controller

  eval/
    dataset.py
    metrics.py
    run.py

  api/
    main.py                  # optional FastAPI

  scripts/
  tests/
  README.md
  pyproject.toml (or requirements.txt)
```

## Interfaces and data contracts

### pages.jsonl record

- doc_id (string)
- source_path (string)
- page_number (int, 1-indexed)
- text (string)

### chunks.jsonl record

- chunk_id (stable/deterministic)
- doc_id
- source_path
- start_page, end_page (int)
- text (string)
- optional: section_path, token_est

### Retrieval return object

- chunk_id
- score
- doc_id
- start_page, end_page
- text (or preview)
- optional: highlights, matched_terms

### Final answer format

- answer (string)
- citations (list of {doc_id, start_page, end_page, chunk_id})
- optional: notes (“what I used / what I couldn’t find”)

## Milestones (fit to one week)

### Milestone 1 — Index works (Day 1)

```
python -m rag.ingest → pages.jsonl
python -m rag.chunk → chunks.jsonl
python -m rag.index_faiss and python -m rag.index_bm25
python -m rag.search_faiss "ML-KEM key generation" prints top chunks with citations
```

Acceptance:

- page spans correct and stable
- relevant results for 3–5 sanity queries

### Milestone 2 — RAG answers with citations (Day 2)

```
python -m rag.ask "What is ML-KEM?" returns answer + citations to FIPS 203 pages
optional: POST /ask
```

Acceptance:

- every factual claim has ≥1 citation (prompt + post-check)

### Milestone 3 — Agent tool use with LangGraph (Day 3)

- LangGraph controller decides when to retrieve more vs answer
- tools available via LangChain tool calling

Acceptance:

- multi-step questions trigger multiple retrievals
- output is structured + cited
- step/tool budgets prevent runaway loops

### Milestone 4 — Evaluation baseline (Day 4)

- 30–80 labeled questions
- python -m eval.run outputs JSON + Markdown report

Acceptance:

- stable scores across runs (small tolerance)

### Milestone 5 — Measured improvement (Day 5–6)

Implement one and show improvement:

- Hybrid + Fusion (RRF) (recommended if not already baseline)
- or reranker toggle (stretch)
- or better chunking/section parsing

Acceptance:

- report shows improved retrieval metrics and/or citation faithfulness

Optional (if time): add pgvector backend via the same Retriever interface (no agent/eval rewrites)

### Milestone 6 — Packaging (Day 7)

- README demo outputs + architecture diagram
- optional Docker + GitHub Actions
- centralized config knobs

## Configuration knobs (centralized)

In rag/config.py:

- CHUNK_TARGET_TOKENS, CHUNK_OVERLAP
- TOP_K, FUSION_NUM_QUERIES, RRF_K
- embedding model name
- VECTOR_BACKEND ("faiss" now; later "pgvector" or "chroma")
- BM25 on/off
- reranker on/off
- parsing backend
- paths (raw_pdf, data/processed)

## Engineering guidelines

- Determinism: stable chunk IDs and ordering.
- No silent failures: warn on page-count mismatch, too many empty pages, etc.

Minimum tests:

- ingest produces non-empty pages
- chunks include valid page spans
- retrieval returns k results
- answer format always includes citations (or explicit “not found”)

- Security hygiene: .env not committed; no API keys in logs.
- Documentation: every module begins with a short contract comment.

## Suggested README contents (recruiter-friendly)

- 10-second description + screenshot/GIF of /ask output with citations
- architecture diagram (pipeline + LangGraph tools)
- how to run (CLI + optional API)
- evaluation: dataset size, metrics, baseline scores, what improved them
- limitations + next steps (reranking, richer section parsing, KG-lite)

> If you want to make the LangChain/LangGraph usage pop even more in the README, add one explicit line in the intro like: “Agent orchestration is implemented with LangGraph (bounded tool-using controller), while ingestion/chunking/eval remain deterministic and framework-independent.” That reads like an engineer wrote it.
