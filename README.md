# nist-pqc-rag-agent
PQC Standards Navigator — Agentic RAG over NIST PDFs with citations + eval harness.

[NIST's Post-Quantum Cryptography Standardization Project](https://csrc.nist.gov/projects/post-quantum-cryptography)

[NIST Releases First 3 Finalized Post-Quantum Encryption Standards](https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards?utm_source=chatgpt.com)

[NIST Selects HQC as Fifth Algorithm for Post-Quantum Encryption](https://www.nist.gov/news-events/news/2025/03/nist-selects-hqc-fifth-algorithm-post-quantum-encryption?utm_source=chatgpt.com)

[HQC](https://pqc-hqc.org/resources.html)

what characteristics of the NIST doc  make this project special?
- technical content: dense with math, tables, algorithms, and domain-specific language
- structured layout: clear sections, subsections, and page numbers that can be leveraged for precise citations
- authoritative source: as the official standards body, NIST docs are definitive references, so accurate retrieval and citation is crucial for trustworthiness

## Retrieval CLI quickstart

Use the project conda environment:

```powershell
conda activate eleven
```

### 1) Build BM25 artifact (needed for hybrid retrieval)

```powershell
python -m rag.index_bm25
```

### 2) Run retrieval-only search

```powershell
python -m rag.search "ML-KEM key generation"
python -m rag.search "Algorithm 19" --mode hybrid --k 8
python -m rag.search "ML-DSA verification" --mode base --backend bm25
```

### 3) Run shared retrieval module directly

```powershell
python -m rag.retrieve "ML-KEM.KeyGen" --mode hybrid --k 5
python -m rag.retrieve "SLH-DSA parameters" --no-query-fusion
```

### 4) Run question answering with citations

```powershell
python -m rag.ask "What does FIPS 203 specify for ML-KEM key generation?"
python -m rag.ask "Compare ML-DSA and SLH-DSA use-cases" --show-evidence
python -m rag.ask "What is Algorithm 19?" --mode hybrid --k 8 --json
```

### 5) Run retrieval sanity script

```powershell
python scripts/mini_retrieval_sanity.py
```

Outputs:
- reports/mini_retrieval_sanity.json
- reports/mini_retrieval_sanity.md

### Script/module dependency map

- rag.search uses rag.retrieve.retrieve
- rag.ask uses rag.retrieve.retrieve and rag.rag_answer.build_cited_answer
- rag.retrieve uses rag.retriever.factory.get_retriever and rag.retriever.bm25_retriever.BM25Retriever
- rag.retriever.bm25_retriever depends on artifact from rag.index_bm25

## Retrieval tuning playbook (algorithm-heavy PDFs)

Core retrieval flags are shared by `rag.search`, `rag.retrieve`, and `rag.ask`.
Output flags (`--show-evidence`, `--json`) are `rag.ask`-only.

### Flag meanings

- `--mode {base|hybrid}`
	- `base`: uses one backend (`--backend`) and still supports query-fusion + RRF across variants.
	- `hybrid`: uses FAISS + BM25 together, then fuses with RRF.

- `--backend {faiss|bm25|...}`
	- Used when `--mode base`.
	- `faiss`: semantic retrieval, good for conceptual wording.
	- `bm25`: lexical retrieval, good for exact symbols/algorithm names.

- `--k N`
	- Final number of results returned after fusion/rerank.

- `--candidate-multiplier N`
	- Expands per-query candidate pool to `k * N` before fusion.
	- Higher values improve recall but increase latency.

- `--k0 N`
	- RRF smoothing constant in `1 / (k0 + rank)`.
	- Lower `k0` gives more weight to top ranks; higher `k0` flattens rank influence.

- `--no-query-fusion`
	- Disables deterministic query variants.
	- Use this when you want strict single-query behavior for debugging.

- `--no-rerank`
	- Disables the lightweight final rerank.
	- Default rerank boosts chunks with exact technical-token matches, then BM25 lexical score.

- `--rerank-pool N`
	- Number of fused candidates considered before truncating to final `k`.
	- Useful when the best exact match appears slightly lower in fused ranking.

- `--show-evidence` (`rag.ask` only)
	- Prints retrieved snippets before answer generation to help audit grounding.

- `--json` (`rag.ask` only)
	- Emits structured JSON with answer, citations, and model metadata.

### See all flags quickly

```powershell
python -m rag.search --help
python -m rag.retrieve --help
python -m rag.ask --help
```

## Environment variable reference

Use environment variables to set project-wide defaults for retrieval and QA CLIs.

### Retrieval

- `VECTOR_BACKEND` (default: `faiss`): base-mode backend.
- `TOP_K` (default: `8`): final number of hits.
- `RETRIEVAL_MODE` (default: `hybrid`): `base` or `hybrid`.
- `RETRIEVAL_QUERY_FUSION` (default: `true`): enables deterministic query rewrites.
- `RETRIEVAL_RRF_K0` (default: `60`): RRF constant in `1 / (k0 + rank)`.
- `RETRIEVAL_CANDIDATE_MULTIPLIER` (default: `4`): candidate expansion before fusion.
- `RETRIEVAL_ENABLE_RERANK` (default: `true`): enables lexical rerank stage.
- `RETRIEVAL_RERANK_POOL` (default: `40`): fused pool size considered before rerank truncation.

### Answering

- `ASK_MAX_CONTEXT_CHUNKS` (default: `6`): max chunks passed to generation.
- `ASK_MAX_CONTEXT_CHARS` (default: `12000`): max total context characters.
- `ASK_MIN_EVIDENCE_HITS` (default: `2`): minimum hits required to attempt answer.
- `ASK_REQUIRE_CITATIONS` (default: `true`): enforces citation-grounded output.
- `ASK_INCLUDE_NEIGHBOR_CHUNKS` (default: `true`): include nearby chunks for spillover.
- `ASK_NEIGHBOR_WINDOW` (default: `1`): neighbor distance in vector-id space.

### Ask CLI defaults

- `ASK_SHOW_EVIDENCE_DEFAULT` (default: `false`): default for `--show-evidence`.
- `ASK_JSON_DEFAULT` (default: `false`): default for `--json`.

### LLM determinism

- `LLM_TEMPERATURE` (default: `0.0`): generation temperature.

### PowerShell examples

```powershell
$env:RETRIEVAL_MODE = "hybrid"
$env:TOP_K = "10"
$env:RETRIEVAL_ENABLE_RERANK = "true"
$env:RETRIEVAL_RERANK_POOL = "50"
python -m rag.search "Algorithm 19 ML-KEM.KeyGen"

$env:ASK_SHOW_EVIDENCE_DEFAULT = "true"
python -m rag.ask "What does FIPS 203 specify for ML-KEM.Decaps?"
```

### Recommended presets

1) **Exact algorithm lookup (best for Algorithm N / KeyGen / Decaps)**

```powershell
python -m rag.search "Algorithm 19 ML-KEM.KeyGen" --mode hybrid --k 5 --candidate-multiplier 6 --k0 50
```

2) **Broader conceptual questions (overview/comparison)**

```powershell
python -m rag.search "How do ML-DSA and SLH-DSA differ in intended use?" --mode hybrid --k 8 --candidate-multiplier 4 --k0 60
```

3) **Debug baseline without fusion/rerank (ablation check)**

```powershell
python -m rag.search "ML-KEM key generation" --mode base --backend faiss --k 8 --no-query-fusion --no-rerank
```

4) **QA with explicit tuning**

```powershell
python -m rag.ask "What does FIPS 203 specify for ML-KEM.Decaps?" --mode hybrid --k 6 --candidate-multiplier 6 --k0 55 --show-evidence
```