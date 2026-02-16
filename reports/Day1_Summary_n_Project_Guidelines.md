# Day 1 Summary & Project Guidelines — NIST PQC RAG Agent

This document consolidates the **Day 1 technical summary** and the ongoing **project guidelines** into a single reference: scope, tooling choices, pipeline artifacts, repository structure, and the plan for a **swappable retrieval backend** (FAISS now; optional VectorDB later). :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## 1. Day 1 Objective

Stand up a reproducible **document → retrieval** pipeline for NIST Post-Quantum Cryptography standards with:

- **Page-level traceability** for citations (doc_id + page span)
- A working **semantic retrieval baseline** (search CLI) to validate retrieval quality
- A modular design that supports a **swappable retrieval backend** without rewriting downstream code

---

## 2. Corpus Selection & Scope

### Included (public, citable, high-signal)
- **FIPS 203 / 204 / 205** (ML-KEM, ML-DSA, SLH-DSA standards)
- **SP 800-227** (implementation considerations / usage guidance)
- **NIST IR 8545** (status / fourth-round context)
- **NIST IR 8547 (IPD)** (transition / migration guidance)

### Deferred (intentional)
- **HQC documentation** and other adjacent/non-standardized documents were deferred for Day 1 to avoid widening scope early and to keep retrieval + evaluation clean.

Rationale: the current set already supports both (1) algorithm-definition questions (FIPS) and (2) real-world transition questions (SP/IR), which is the best “portfolio ROI” for the first week.

---

## 3. What Was Built (Day 1 Deliverables)

### 3.1 Ingestion (PDF → page-level text)
**Tooling choice:** **LlamaParse** for robust parsing of technical PDFs (tables/layout).  
**Outputs:**
- `data/processed/pages.jsonl` — page-addressable records: `(doc_id, page_number, text, ...)`
- `data/processed/*_parsed.json` — intermediate per-document parse artifacts for traceability/debugging

### 3.2 Cleaning (normalize PDF artifacts)
- `pages.jsonl` → `pages_clean.jsonl`
- Purpose: improve chunk boundaries and embedding signal by reducing layout noise.

### 3.3 Chunking (page-aware RAG-ready chunks)
- `pages_clean.jsonl` → `chunks.jsonl`
- Each chunk preserves citation provenance:
  - `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`
  - plus diagnostics (`char_len`, `approx_tokens`) for tuning.

### 3.4 Embedding (chunks → dense vectors)
**Tooling choice:** `sentence-transformers` (local embeddings; reproducible; avoids API costs/rate limits).  
**Outputs:**
- `embeddings.npy` — float32 matrix (N × dim)
- `emb_meta.json` — embedding config (model_name, dim, normalization flags, etc.)
- `chunk_store.jsonl` — maps `vector_id` → metadata + **text** (RAG-ready)

### 3.5 Indexing + Search (FAISS baseline retrieval)
**Tooling choice:** **FAISS (faiss-cpu)**, chosen for:
- strong employer recognition
- fast similarity search + simple persistence
- good fit for current scale and timeline

**Outputs:**
- `faiss.index`
- CLI search validated retrieval quality for real queries (e.g., ML-KEM definition and IR 8547 migration considerations).

---

## 4. Tools and APIs Used

### Libraries / Components
- **LlamaParse**: PDF parsing to page-addressable text
- **sentence-transformers**: embedding runtime (local)
- **FAISS (faiss-cpu)**: vector similarity index
- **NumPy**: embedding persistence (`.npy`)
- **JSONL**: inspectable stores for pages/chunks/chunk_store

### Not chosen (yet) — and why
- **VectorDB backend (pgvector / Chroma / Qdrant)**  
  Deferred due to Windows + time constraints (Docker/WSL2/service overhead). Day 1 focus was proving retrieval correctness quickly.
- **Hybrid BM25 retrieval**  
  Useful later for exact-token matching (e.g., “IND-CCA2”, “decapsulation failure”), but not required to validate the dense baseline to start.
- **Expanding corpus (e.g., HQC)**  
  Deferred to avoid adding retrieval noise before baseline eval is established.

---

## 5. Swappable Backend Architecture (Implemented on Day 1)

A minimal retriever interface was implemented so the rest of the project (RAG answering, agent tools, evaluation) depends on **Retriever.search()**, not on FAISS-specific code.

### 5.1 Interface
- `rag/retriever/base.py`
  - `ChunkHit` (score + doc_id + page span + text)
  - `Retriever` protocol

### 5.2 FAISS implementation
- `rag/retriever/faiss_retriever.py`
  - wraps FAISS search, optional dedupe, returns `List[ChunkHit]`

### 5.3 Factory selection
- `rag/retriever/factory.py`
  - selects backend via config (default `faiss`)
  - future: add `pgvector` or `chroma` retrievers without changing downstream code

This design enables backend evolution with minimal churn and is aligned with “production-minded” engineering expectations.

---

## 6. Planned Upgrade: Add VectorDB Backend (Optional)

### Why add later?
Once the system is end-to-end (RAG + eval), adding a VectorDB backend is an **incremental** improvement that strengthens “production readiness” without jeopardizing timeline.

### Candidate options
1) **pgvector (Postgres)** — most production-ish
   - Pros: SQL metadata filtering, durability, industry credibility
   - Cons: Docker/WSL2 overhead on Windows
2) **Chroma** — fastest dev experience
   - Pros: simple integration, common RAG backend
   - Cons: less traditional infra credibility than Postgres

### How the swap works
Only add a new retriever module and wire it into the factory:
- `rag/retriever/pgvector_retriever.py` (or `chroma_retriever.py`)
- update `rag/retriever/factory.py` to route by `BACKEND`

Everything else remains stable.

---

## 7. How to Run (Day 1 artifacts)

Typical workflow:
1) Ingest PDFs → `pages.jsonl`
2) Clean pages → `pages_clean.jsonl`
3) Chunk → `chunks.jsonl`
4) Embed → `embeddings.npy`, `chunk_store.jsonl`, `emb_meta.json`
5) Index → `faiss.index`
6) Search → prints top hits with citations (doc_id + page span)

CLI entry points (current):
- `python -m rag.ingest`
- `python -m rag.clean`
- `python -m rag.chunk`
- `python -m rag.embed`
- `python -m rag.index_faiss`
- `python -m rag.search_faiss "query"` (backend-specific)
- `python -m rag.search "query"` (backend-agnostic; uses retriever interface)

---

## 8. Repository Structure (Updated)
.
├── pyproject.toml
├── README.md
├── requirements.txt
├── data/
│ ├── processed/
│ │ ├── chunk_store.jsonl
│ │ ├── chunks.jsonl
│ │ ├── emb_meta.json
│ │ ├── embeddings.npy
│ │ ├── faiss.index
│ │ ├── NIST.FIPS.203_parsed.json
│ │ ├── NIST.FIPS.204_parsed.json
│ │ ├── NIST.FIPS.205_parsed.json
│ │ ├── NIST.IR.8545_parsed.json
│ │ ├── NIST.IR.8547.ipd_parsed.json
│ │ ├── NIST.SP.800-227_parsed.json
│ │ ├── pages_clean.jsonl
│ │ └── pages.jsonl
│ └── raw_pdfs/
│ └── readme.md
├── rag/
│ ├── init.py
│ ├── chunk.py
│ ├── clean.py
│ ├── config.py
│ ├── embed.py
│ ├── index_faiss.py
│ ├── ingest.py
│ ├── search_faiss.py
│ ├── search.py
│ ├── utils.py
│ └── retriever/
│ ├── base.py
│ ├── factory.py
│ └── faiss_retriever.py
├── reports/
│ ├── progress.md
│ └── day1_summary_guidelines.md
├── scripts/
│ ├── check_clean.py
│ ├── clean_pages.py
│ ├── count_chunks.py
│ └── make_chunks.py
└── tests/

> Note: this combined file is intended to replace the separate `day1_summary.md` and `day1_project_guidelines.md` sources.

---

## 9. Next Steps (Day 2+)

### Day 2: RAG answering (citation-grounded)
- Implement `/ask` flow: retrieve → assemble context → generate answer → return citations
- Enforce citation discipline (answers must be grounded in retrieved chunks)

### Day 3+: Agent tools + evaluation
- Agent tools: retrieve, summarize section, compare algorithms, extract definitions
- Evaluation harness: retrieval metrics + faithfulness checks + regression set
- Optional: add VectorDB backend via `Retriever` interface

---
