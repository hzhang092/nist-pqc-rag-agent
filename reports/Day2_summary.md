# Day 2 Technical Summary — nist-pqc-rag-agent

### Goal shipped
Day 2 turned the Day 1 “index + retriever” foundation into a **citation-grounded, deterministic RAG answering flow** runnable from the CLI:

- `python -m rag.ask "<question>"` → **answer + inline citations** + structured citation objects
- `python -m rag.search "<query>"` → debug retrieval view (now using the shared retrieval pipeline)

The key product change is that answers are now **evidence-bound**: every sentence must end with an inline marker like **`[c1]`**, and those markers must match `citations[].key`. If the system can’t support the question with retrieved evidence, it refuses with a fixed contract:  
`"not found in provided docs"` and empty citations.

---

## What I built / changed

### 1) Centralized runtime knobs in `rag/config.py`
I added a single settings surface for Day 2 behavior, including:

- Retrieval knobs: backend selection (FAISS vs BM25 vs hybrid), `TOP_K`, candidate multipliers, RRF constants (e.g., `k0`), rerank toggles/pool sizes.
- Answering knobs: max evidence chunks, max context chars, minimum evidence hits, strict citation enforcement.
- LLM knobs: default Gemini model (`gemini-3-flash-preview`), temperature 0, plus debugging flags for `--json` and `--show-evidence`.

**Why:** This makes the pipeline tunable without editing core logic, supports a “swappable backend” architecture, and keeps behavior reproducible across runs/environments.

---

### 2) Output contract + strict validation in `rag/types.py`
I defined a structured output contract:

- `Citation(key, doc_id, start_page, end_page, chunk_id)`
- `AnswerResult(answer, citations, notes?)`

And added validation utilities to enforce invariants:

- refusal ⇒ exact refusal string + `citations=[]`
- non-refusal ⇒ citations required
- optional but enabled: inline markers `[c#]` required and must correspond to real citation keys

**Why:** This makes “citation groundedness” testable and machine-checkable (for later eval harness + agent tool calls).

---

### 3) Citation-grounded answer builder in `rag/rag_answer.py`
This is the core Day 2 engine.

**Evidence selection**
- Deduplicate by `chunk_id`
- Deterministic sort key: `(-score, doc_id, start_page, end_page, chunk_id)`
- Enforce context budgets:
  - max chunks
  - max chars (guards against big tables/algorithms blowing up prompts)

**Evidence windowing**
- After selecting the main evidence chunks, I expanded context using **neighbor chunks** (prev/next in doc order) when available and within the context budget.
- This targets a classic FIPS failure mode: algorithms and tables often span chunk boundaries.

**Prompting policy**
- “Prompt specificity guard”: ban details not explicitly supported by evidence (e.g., algorithm numbers, byte lengths, variable names) unless present in evidence text.
- “Evidence-driven answering mode”: answers must be concise and sentence-level cited.

**Inline citation enforcement**
- Every sentence must contain at least one marker `[c#]`
- Markers must exist in the evidence-derived `key_to_citation` map
- If the model violates rules, the system refuses (safe default)

**Why:** This turns the model into a *summarizer of evidence*, not a “freeform explainer,” which is essential for standards PDFs.

---


---

### 3.1) Bugfix: “Algorithm N” questions refusing despite correct evidence (SHAKE128example)

**Repro / symptom**

A concrete failure showed up with algorithm-style questions, e.g.:

- `python -m rag.ask "What are the steps in Algorithm 2 SHAKE128?" --k 5 --show-evidence`

Retrieval **did return** the correct chunk (`NIST.FIPS.203::p0028::c001`) containing the full Algorithm 2 pseudocode, but the system could still refuse with the fixed contract (`"not found in provided docs"`). This was initially confusing because `--show-evidence` truncates each chunk for readability, so the “top hits” preview didn’t visibly include the numbered steps even when the full chunk did.

**Debugging instrumentation added**

- Printed the **actual context block** sent to the LLM (the `[c1] …` evidence payload). This confirmed that the “retriever hits” list and the “context to LLM” list can differ because `select_evidence()` applies **dedup + neighbor expansion + context budgeting** (expected behavior).
- Printed **RAW model output** + parsed citation keys + validated keys to confirm whether the refusal was coming from retrieval, the model, or the validator.

**Root cause**

In `rag_answer.py`, the prompt assembly had a formatting bug: the Rule 6 line was concatenated directly into `Question:` (missing a newline), which reduced compliance with the citation/answering rules and increased refusals/low-quality outputs for algorithm-heavy evidence.

**Fix shipped**

Updated `rag/rag_answer.py` to make algorithm answers reliable:

- Fixed prompt formatting (explicit newline separation before `Question:`).
- Added evidence prettification for standards pseudocode (insert newlines before `1:`, `2:`, `for (...)` etc.) so algorithm steps are readable to the model.
- Added a deterministic safety net for Algorithm questions: if the model refuses but the evidence chunk contains numbered steps, extract and return the steps directly **with valid inline citations**, still passing the strict `enforce_inline_citations` contract.

This turns “algorithm blocks + PDF layout weirdness” into a stable, testable path rather than a prompt-luck issue.

### 4) Gemini LLM integration (free tier) in `rag/llm/gemini.py`
I integrated Gemini via the `google-genai` SDK and set the default model to:

- `gemini-3-flash-preview`

This avoided the earlier quota pitfall where `gemini-2.5-pro` effectively had a free-tier quota limit of 0.

I also made the CLI output include the model name (in `--show-evidence` and/or `--json`) for reproducibility.

**Why:** Day 2 needed a working free-tier LLM backend to complete the end-to-end “ask” flow.

---

## Retrieval quality improvements (major Day 2 work)

### 5) BM25 indexing + retriever
I added lexical retrieval because dense embeddings struggle with spec-heavy tokens:

- Built a BM25 index artifact: `data/processed/bm25.pkl`
- Implemented `rag/index_bm25.py` and `rag/retriever/bm25_retriever.py`
- Used a tokenizer that preserves “technical compounds”:
  - dot tokens (`ML-KEM.KeyGen`)
  - hyphen tokens (`ML-KEM-768`)
  - acronyms / constants (`SHAKE128`, `NTT`)

**Why:** Standards PDFs depend heavily on exact identifiers; lexical retrieval recovers precise sections that dense vectors often miss.

---

### 6) Hybrid retrieval + fusion in `rag/retrieve.py`
I implemented hybrid search combining:

- FAISS dense hits
- BM25 lexical hits

Then fused results using **Reciprocal Rank Fusion (RRF)**:

- `rrf_score += 1 / (k0 + rank)`
- Final stable sort by:
  - `-rrf_score`, then `(doc_id, start_page, chunk_id)`

**Why:** Hybrid retrieval improves recall of exact spec sections while retaining semantic matching.

---

### 7) Query fusion (generalized + deterministic)
I added deterministic query rewriting/expansion (“query fusion”) to improve retrieval for standards questions, including:

- Generalized rules across ML-KEM / ML-DSA / SLH-DSA style patterns
- Added `"Algorithm N"` as its own explicit variant when detected
- Included “dot-name” variants (e.g., `X.KeyGen`, `X.Sign`, `X.Verify`) when the query implies those operations

Then ran hybrid retrieval across variants and merged again via RRF.

**Why:** This is a lightweight, deterministic version of RAG-Fusion: it increases recall of the right spec blocks without using an LLM for rewriting (keeps cost + nondeterminism down).

---

### 8) Cheap reranker toggle
After fusion, I added an optional lightweight rerank stage:
- prioritize hits containing **exact technical tokens** from the query
- then apply a lexical scoring signal (BM25-style score or BM25’s `score_text`)
- stable tie-breaking maintained

**Why:** This nudges “actual algorithm text” above “overview/differences” sections, which is common when dense search returns semantically related but non-authoritative chunks.

---

## Tests added (what + why)

### Answer contract + citation correctness
- `tests/test_types.py`  
  Validates `AnswerResult`/`Citation` invariants (refusal rules, page span sanity).
- `tests/test_rag_answer.py`  
  Verifies:
  - refusal behavior when evidence insufficient
  - inline markers required
  - markers map to `citations[].key`
  - evidence selection is stable (dedup + deterministic sorting)

**Why:** Prevent regressions where answers “look cited” but violate your strict rules.

### Retrieval tests (determinism + correctness)
- `tests/test_bm25_index.py`  
  Checks BM25 artifact build and basic retriever behavior.
- `tests/test_query_fusion.py`  
  Ensures query variants are deterministic, deduped, and include key expansions.
- `tests/test_retrieve_rrf.py`  
  Tests RRF behavior and stable tie-breaking.
- `tests/test_retrieve_determinism.py`  
  Ensures:
  - query variants order stability
  - RRF stable output under ties
  - hybrid pipeline deterministic even if retrievers return different orders (with rerank enabled)

**Why:** Retrieval order instability causes citation key instability (`c1..cN`) which breaks reproducibility and makes evaluation impossible.

### Tooling sanity outputs
- `scripts/mini_retrieval_sanity.py` + `reports/mini_retrieval_sanity.{json,md}`  
  A small snapshot suite of fixed queries that records top hits (doc + pages), acting as a regression baseline.

**Why:** Simple “does retrieval still look sane” check without needing a full eval harness yet.

---

## End-of-Day 2 results
You now have:

- **Working RAG CLI** with strict inline citations + refusal contract (`rag.ask`)
- **Hybrid retrieval** (FAISS + BM25) with **RRF fusion**
- **Deterministic query fusion** rules tuned for technical spec language
- **Evidence windowing** to capture multi-chunk algorithm blocks
- **Cheap reranker** toggle to prioritize exact-match spec content
- A growing **test suite** that locks down determinism, citation validity, and retrieval behavior

---

## What needs to be done next (Day 3+)

### 1) Add an agent loop (LangGraph) that uses your stable tool boundaries
Now that retrieval and answer generation are clean functions, wrap them as tools:
- `retrieve(query) -> hits`
- `answer(question, hits) -> AnswerResult`

Then build a bounded agent loop:
- retrieve → assess evidence sufficiency → (optional) reformulate query → retrieve again → answer

This matches your project’s “agentic, tool-using RAG” direction and makes improvements measurable rather than prompt-only.

### 2) Add evaluation harness beyond sanity snapshots
Build a small eval set (10–30 questions) and score:
- citation compliance rate
- refusal correctness (should refuse when evidence truly absent)
- retrieval hit-rate for “gold pages”
- answer faithfulness via simple heuristics (e.g., claims per sentence must cite, no uncited numbers)

### 3) Improve faithfulness checks
Right now you enforce citation *format* strongly. Next step is checking citation *support* more directly:
- sentence-level evidence overlap heuristic
- “quote-backed” mode for critical claims (optionally include short quoted spans)
- optional lightweight verifier pass (still cheap model)

### 4) Backend swap readiness (vector DB later)
Your retriever interface is now strong enough to introduce:
- pgvector / Chroma backend behind the same interface
- the hybrid+fusion logic should remain unchanged (only the dense backend changes)

---

## File structure at end of Day 2
You ended Day 2 with a coherent repo layout including:
- `rag/` core pipeline (`config.py`, `types.py`, `rag_answer.py`, `retrieve.py`, `ask.py`, `search.py`)
- `rag/retriever/` swappable retrievers (FAISS, BM25, factory)
- `data/processed/` artifacts: FAISS index, embeddings, BM25 pickle, parsed pages/chunks
- `scripts/` reproducible utilities (`mini_retrieval_sanity.py`, chunking/cleaning)
- `tests/` unit tests for contracts + retrieval determinism and fusion correctness
- `reports/` day summaries + progress + sanity snapshots

That’s a strong Day 2. You’ve moved from “I have an index” to “I have an auditable, reproducible RAG system with hybrid retrieval and tests.”