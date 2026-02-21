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

**Detailed mechanism implemented:**
- Enumerated input PDFs with deterministic ordering via `sorted(data/raw_pdfs/*.pdf)`.
- For each PDF, used `pypdf` to read true page count, then parsed with `LlamaParse(result_type="markdown")`.
- Set parsing instruction to preserve technical structure (tables + LaTeX math).
- Ran a page-count sanity check (`parsed_pages` vs true PDF pages) and emitted warnings on mismatch.
- Wrote both:
  - per-document parsed JSON (`*_parsed.json`) for debugging,
  - unified `pages.jsonl` records with `doc_id`, `source_path`, `page_number`, `text`.

### 3.2 Cleaning (normalize PDF artifacts)
- `pages.jsonl` → `pages_clean.jsonl`
- Purpose: improve chunk boundaries and embedding signal by reducing layout noise.

**Detailed mechanism implemented:**
- Unicode normalization with `NFKC`.
- Removed soft hyphens / zero-width characters and replaced ligatures (`ﬁ`→`fi`, `ﬂ`→`fl`).
- Normalized whitespace: standardized line endings and collapsed repeated spaces/tabs (including extra spaces).
- De-hyphenated wrapped words across line breaks (e.g., `algo-\nrithm` → `algorithm`).
- Removed standalone page-number lines (e.g., `12`, `Page 12`).
- Detected repeated headers/footers per document using top/bottom line frequency (canonicalized lines, threshold ratio).
- Applied structure-aware line joining:
  - joined only wrapped prose lines,
  - preserved line structure for tables, algorithms/pseudocode, and math blocks.
- Collapsed multiple blank-line runs to a single blank line.

### 3.3 Chunking (page-aware RAG-ready chunks)
- `pages_clean.jsonl` → `chunks.jsonl`
- Each chunk preserves citation provenance:
  - `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`
  - plus diagnostics (`char_len`, `approx_tokens`) for tuning.

**Detailed mechanism implemented:**
- Split each page into blocks by blank lines.
- Classified blocks with heuristics (table / algorithm-code / math):
  - mostly verbatim blocks kept line-by-line,
  - prose blocks joined to undo PDF hard wraps.
- Packed blocks greedily into chunks with configured bounds:
  - `target_chars=1400`, `min_chars=250`, `max_chars=2200`, `overlap_blocks=1`.
- Used stable chunk IDs per page: `{doc_id}::p{page:04d}::c{chunk:03d}`.
- Preserved citation fields on every chunk: `doc_id`, `start_page`, `end_page` (Day 1: page-local chunks so `start_page=end_page=page_number`).

### 3.4 Embedding (chunks → dense vectors)
**Tooling choice:** `sentence-transformers` (local embeddings; reproducible; avoids API costs/rate limits).  
**Outputs:**
- `embeddings.npy` — float32 matrix (N × dim)
- `emb_meta.json` — embedding config (model_name, dim, normalization flags, etc.)
- `chunk_store.jsonl` — maps `vector_id` → metadata + **text** (RAG-ready)

**Detailed mechanism implemented:**
- Model: `BAAI/bge-base-en-v1.5`, batch size `64`.
- Encoded non-empty chunk text with `normalize_embeddings=True` and cast to `float32`.
- Built deterministic `vector_id` mapping aligned to embedding row index (`vector_id == row_id`).
- Persisted:
  - `embeddings.npy` (dense matrix),
  - `chunk_store.jsonl` (vector-to-chunk/text mapping),
  - `emb_meta.json` (model name, dim, normalization flag, paths).

### 3.5 Indexing + Search (FAISS baseline retrieval)
**Tooling choice:** **FAISS (faiss-cpu)**, chosen for:
- strong employer recognition
- fast similarity search + simple persistence
- good fit for current scale and timeline

**Outputs:**
- `faiss.index`
- CLI search validated retrieval quality for real queries (e.g., ML-KEM definition and IR 8547 migration considerations).

**Detailed mechanism implemented:**
- Loaded embeddings + dimension metadata, asserted shape consistency.
- Applied L2 normalization before indexing.
- Built FAISS with `faiss.IndexFlatIP(dim)`:
  - exact brute-force nearest-neighbor search (no ANN approximation),
  - inner product on normalized vectors to approximate cosine similarity.
- Added all vectors and persisted `faiss.index`.
- Search path embedded query with the same model + normalization, retrieved an expanded candidate pool, then deduped by `(doc_id, start_page, end_page)` before returning top hits.

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

# evaluation
## you asked

I am building a rag system. does the following FAISS baseline retrieval match the expectation:
(eleven) D:\Waterloo\terms\MDSAI\1B\projects\nist-pqc-rag-agent>python -m rag.search "What is ML-KEM"
Query: What is ML-KEM
[1] score=0.7240  NIST.FIPS.203  p44-p44  (NIST.FIPS.203::p0044::c000)
7. The ML-KEM Key-Encapsulation Mechanism  This section describes the three main algorithms of the ML-KEM scheme:  1. Key generation (ML-KEM.KeyGen)  2. Encapsulation (ML-KEM.Encaps)  3. Decapsulation (ML-KEM.Decaps) To instantiate ML-KEM, one must select a parameter set. Each parameter set is assoc...
[2] score=0.7199  NIST.SP.800-227  p43-p43  (NIST.SP.800-227::p0043::c001)
5.1.3. ML-KEM ML-KEM is a high-performance, general-purpose, lattice-based key-encapsulation mechanism. It is a NIST-approved KEM and was standardized in FIPS 203 [3]. ML-KEM is based on CRYSTALS-Kyber [26], which was a candidate in the NIST PQC standardization process. It is believed to satisfy IND...
[3] score=0.7119  NIST.FIPS.203  p10-p10  (NIST.FIPS.203::p0010::c000)
1. Introduction  1.1 Purpose and Scope This standard specifies the Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM). A key-encapsulation mechanism (KEM) is a set of algorithms that can be used to establish a shared secret key between two parties communicating over a public channel. A KEM is...
[4] score=0.7073  NIST.FIPS.203  p46-p46  (NIST.FIPS.203::p0046::c000)
nor with every execution of ML-KEM.Encaps. Instead, assurance that these checks have been performed can be acquired through other means (see SP 800-227 [1]).  Algorithm 20 ML-KEM.Encaps(ek) Uses the encapsulation key to generate a shared secret key and an associated ciphertext. Checked input: encaps...
[5] score=0.7006  NIST.FIPS.203  p48-p48  (NIST.FIPS.203::p0048::c001)
Table 2. Approved parameter sets for ML-KEM  n q k η1 η2 du dv required RBG strength (bits) ML-KEM-512 256 3329 2 3 2 10 4 128 ML-KEM-768 256 3329 3 2 2 10 4 192 ML-KEM-1024 256 3329 4 2 2 11 5 256  Table 3. Sizes (in bytes) of keys and ciphertexts of ML-KEM  encapsulation key decapsulation key ciph...

the critic of the retrieval 
Based on the new sources provided, the baseline FAISS retrieval was good but incomplete. It successfully captured the definition (what it is) and the mechanics (algorithms and parameters), but it missed the theoretical basis and security model which are fundamental to defining a cryptographic primitive.
Here is a critique of the retrieved chunks and examples of better information available in the source documents.

Critique of Retrieved Chunks
Strong on Definition & Mechanics: The retrieved chunks correctly identified ML-KEM as a lattice-based KEM standardized in FIPS 203, listed its three main algorithms (KeyGen, Encaps, Decaps), and provided the parameter sets (512, 768, 1024).
Weak on Security Basis: The retrieved chunks did not explicitly capture the Module Learning With Errors (MLWE) problem. Defining ML-KEM without mentioning MLWE is like defining RSA without mentioning integer factorization; it explains what it does, but not why it is secure.
Weak on Security Properties: The retrieval did not highlight IND-CCA2 (Indistinguishability under Adaptive Chosen Ciphertext Attack) security. This is the gold standard for general-purpose KEMs and a critical differentiator between ML-KEM and other primitives (like the raw K-PKE component it uses).
Better Information Not Retrieved
The following sections from the provided sources offer deeper insight into "What is ML-KEM" and would improve the answer:
1. The Theoretical Foundation (MLWE)

Source: FIPS 203, Abstract and Section 3.2
Why it's better: It explains the mathematical hardness assumption.
Excerpt: "The security of ML-KEM is based on the presumed hardness of the so-called Module Learning with Errors (MLWE) problem... which is a generalization of the Learning With Errors (LWE) problem."
2. The Security Model (IND-CCA2)

Source: FIPS 203, Section 3.2
Why it's better: It explains the specific security guarantee provided, which is crucial for protocol designers.
Excerpt: "ML-KEM is believed to satisfy so-called IND-CCA2 security... The cryptographic transformation from K-PKE to ML-KEM is crucial for achieving IND-CCA2 security."
3. The Construction (Fujisaki-Okamoto Transform)

Source: FIPS 203, Section 3.2
Why it's better: It explains how the scheme is built from a simpler component.
Excerpt: "ML-KEM proceeds in two steps. First... construct a public-key encryption (PKE) scheme... Second, this PKE scheme is converted into a key-encapsulation mechanism using the so-called Fujisaki-Okamoto (FO) transform."
4. Context and Origin

Source: FIPS 203, Section 1.2
Why it's better: It provides the historical context of the NIST PQC process.
Excerpt: "This standard specifies a variant of the selected algorithm CRYSTALS-KYBER... a submission in the NIST Post-Quantum Cryptography Standardization project."
Summary of Missing Key Information
To fully answer "What is ML-KEM," a RAG system should ideally retrieve information that allows it to say:

"ML-KEM is a lattice-based key encapsulation mechanism derived from CRYSTALS-Kyber. Its security is based on the Module Learning With Errors (MLWE) problem. It is constructed by applying the Fujisaki-Okamoto transform to a CPA-secure encryption scheme (K-PKE) to achieve IND-CCA2 security."
The original retrieval missed the bolded technical details.

---

## gemini response

Yes, this retrieval baseline **absolutely matches (and arguably exceeds) the expectation for a Milestone 1 deliverable.**

For a "Day 1" system using standard chunking and basic vector search, this is a **strong pass**. You have successfully retrieved the "What", the "How" (algorithms), and the "Specs" (parameters).

Here is the breakdown of why this is a success for your current stage, and how to use the critique to drive the rest of your week.

### 1. Why this is a "Pass" for Milestone 1
Your goal for Milestone 1 is defined in your specs as: *"relevant results for 3–5 sanity queries."*

Look at what your baseline captured:
* **The Definition (Chunk 3):** *"This standard specifies the Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM)."* — This is a direct, factual answer.
* **The Context (Chunk 2):** *"ML-KEM is a high-performance, general-purpose, lattice-based key-encapsulation mechanism... based on CRYSTALS-Kyber."* — This is excellent context from a secondary source (SP 800-227).
* **The Components (Chunk 1):** Lists `KeyGen`, `Encaps`, and `Decaps`.

**Verdict:** The system works. It is retrieving semantically relevant text. If you were demoing this to a recruiter on Day 1, they would say, "Great, it reads the docs."

### 2. Contextualizing the Critique (The "Gap")
The critique you provided is harsh for a baseline, but **perfect for your project narrative**.

The critique correctly identifies that you missed the *deep* properties:
* **The Security Basis:** MLWE (Module Learning With Errors)
* **The Guarantee:** IND-CCA2
* **The Construction:** Fujisaki-Okamoto Transform

**Why the baseline missed these:**
Vector search optimizes for semantic similarity. The sentence *"ML-KEM is a lattice-based KEM"* (Chunk 2) is semantically closer to the query *"What is ML-KEM"* than a sentence like *"The security is based on the presumed hardness of MLWE."*

**How to use this "failure":**
Do *not* try to fix this by hacking the chunker right now. Instead, use this gap to justify the complexity of your **Milestone 3 (Agent)** and **Milestone 5 (Improvements)**.

This is your "Before" snapshot.
* **Problem:** Naive vector search answers "What is it?" superficially but misses the "Why is it secure?" unless the user explicitly asks for security proofs.
* **Solution (Your Week-1 Goals):**
    * **Hybrid Search:** BM25 might catch specific terms if the query contained them, but...
    * **Query Expansion (Fusion):** If your agent expands "What is ML-KEM" into multiple sub-queries like *"ML-KEM security basis"*, *"ML-KEM theoretical hardness"*, and *"ML-KEM construction"*, you will instantly hit those missing chunks.
    * **Agent Logic:** A `resolve_definition` tool could be prompted to specifically look for "hardness assumptions" or "security models."

### 3. Immediate Recommendation
You are ready to move to **Milestone 2 (RAG Answers)**.

When you generate the answer using these chunks, your LLM will likely say:
> "ML-KEM is a high-performance, lattice-based key-encapsulation mechanism standardized in FIPS 203. It is based on the CRYSTALS-Kyber algorithm and consists of three main algorithms: KeyGen, Encaps, and Decaps."

This is a perfectly acceptable baseline answer.

**Strategic Tip for your Evaluation (Milestone 4):**
Add the question *"What is the hardness assumption behind ML-KEM?"* to your eval set.
* **Current Score:** 0 (Fail) - The current retrieval didn't find MLWE.
* **Future Score:** 1 (Pass) - Once you implement Query Fusion or Hybrid Search.

This measurable improvement is exactly what makes the project "recruiter-legible." **Proceed with confidence.**
