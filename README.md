# nist-pqc-rag-agent
PQC Standards Navigator — Agentic RAG over NIST PDFs with citations + eval harness.

## Project overview
- Implements a retrieval-augmented generation (RAG) system over NIST's post-quantum cryptography standardization documents.
- Uses a LangGraph agent to perform tool-using question answering with retrieval and citation grounding.
- Provides a CLI for retrieval and QA, plus an evaluation suite for tuning retrieval performance.

## Key NIST document sources in the project
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

## Docker quickstart

1) Create runtime env file from template:

```powershell
Copy-Item .env.example .env
```

2) Build the image:

```powershell
docker compose build rag
```

3) Run deterministic ingestion/index pipeline in containers:

```powershell
docker compose run --rm rag python -m rag.ingest
docker compose run --rm rag python scripts/clean_pages.py
docker compose run --rm rag python scripts/make_chunks.py
docker compose run --rm rag python -m rag.embed
docker compose run --rm rag python -m rag.index_faiss
docker compose run --rm rag python -m rag.index_bm25
```

4) Run retrieval/eval from Docker:

```powershell
docker compose run --rm rag python -m rag.search "ML-KEM key generation"
docker compose run --rm rag python -m eval.run
```

5) Optional task runner (PowerShell):

```powershell
./scripts/docker.ps1 -Task build
./scripts/docker.ps1 -Task pipeline
./scripts/docker.ps1 -Task search -Query "Algorithm 19 ML-KEM.KeyGen"
./scripts/docker.ps1 -Task ask -Query "What does FIPS 203 specify for ML-KEM key generation?"
./scripts/docker.ps1 -Task test
```

Notes:
- `docker-compose.yml` mounts `data/`, `reports/`, and `runs/` so generated artifacts persist on host.
- Hugging Face model cache is persisted in named volume `hf_cache`.
- Docker image pins CPU-only PyTorch (`torch==2.7.0` from PyTorch CPU index) to avoid CUDA/NVIDIA wheel downloads.
- Keep secrets in `.env` only; never commit `.env`.

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
python -m rag.ask "What is Algorithm 19?" --json --no-evidence
python -m rag.ask "What is Algorithm 19?" --json --save-json reports/ask_algorithm19.json
```

### 5) Run retrieval sanity script

```powershell
python scripts/mini_retrieval_sanity.py
```

Outputs:
- reports/mini_retrieval_sanity.json
- reports/mini_retrieval_sanity.md

## metrics reference
### Retrieval metrics
- `Recall@k`: Proportion of questions for which at least one relevant document is in the top-k retrieved results.
- `MRR@k` (Mean Reciprocal Rank): Average of the reciprocal ranks of the first relevant document across all questions, considering only the top-k results.
- `nDCG@k` (normalized Discounted Cumulative Gain): Evaluates the quality of the ranked retrieval results by considering the relevance and position of retrieved documents, normalized to a scale of 0 to 1.

## Scripts reference

This section documents the runnable scripts/modules, what each one is for, and available flags.

### `scripts/clean_pages.py`

Purpose:
- Cleans `data/processed/pages.jsonl` into `data/processed/pages_clean.jsonl` using `rag.clean`.
- Applies header/footer removal, boilerplate filtering, and wrapped-line joining.

Run:

```powershell
python scripts/clean_pages.py
```

Flags:
- No CLI flags in this script.
- To change behavior, edit constants/config inside the file (`PAGES_IN`, `PAGES_OUT`, `CleanConfig`).

---

### `scripts/make_chunks.py`

Purpose:
- Builds chunked retrieval units from cleaned pages.
- Reads `data/processed/pages_clean.jsonl` and writes `data/processed/chunks.jsonl` via `rag.chunk.run_chunking_per_page`.

Run:

```powershell
python scripts/make_chunks.py
```

Flags:
- No CLI flags in this script.
- To change behavior, edit `ChunkConfig(...)` in the file.

---

### `scripts/mini_retrieval_sanity.py`

Purpose:
- Runs a small fixed query set to compare retrieval quality for:
	- base vector retrieval (`mode=base`, `backend=faiss`, no fusion), and
	- hybrid retrieval (`mode=hybrid`, fusion enabled).
- Produces quick qualitative reports for tuning sanity checks.

Run:

```powershell
python scripts/mini_retrieval_sanity.py
```

Outputs:
- `reports/mini_retrieval_sanity.json`
- `reports/mini_retrieval_sanity.md`

Flags:
- No CLI flags in this script.
- To change query set, edit `QUERIES` in the file.

---

### `rag/agent/ask.py` (LangGraph agent CLI)

Purpose:
- Runs the bounded tool-using LangGraph agent for one question.
- Uses routing + retrieval tools + citation-grounded answer generation.
- Optionally writes a trace JSON artifact for debugging and evaluation runs.

Run:

```powershell
python -m rag.agent.ask "What are the steps in Algorithm 2 SHAKE128?"
```

Flags:
- `question` (positional)
	- User question string sent to the agent.
- `--no-trace`
	- Disable writing trace JSON to disk.
- `--out-dir` (default: `runs/agent`)
	- Trace output directory when tracing is enabled.
- `--json`
	- Print full `AgentState` JSON to stdout instead of formatted answer/citations.

Outputs:
- Stdout answer + citations (default), or full JSON state with `--json`.
- Trace file in `<out-dir>/agent_<timestamp>_<slug>.json` (unless `--no-trace`).

Examples:

```powershell
python -m rag.agent.ask "What is ML-KEM?"
python -m rag.agent.ask "Compare ML-DSA and SLH-DSA" --json
python -m rag.agent.ask "ML-KEM key generation" --out-dir runs/agent/day3
python -m rag.agent.ask "What is Algorithm 19?" --no-trace
```

### Agent trace JSON fields

Trace files are produced by `rag.lc.trace.write_trace(...)` and serialize the final `AgentState`.

Typical top-level fields:
- `question`: original user query.
- `plan`: router decision object (action, reason, optional args/mode hint).
- `evidence`: retrieved evidence items used by the answer step.
	- item shape: `score`, `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`.
- `draft_answer`: final answer text (or refusal text).
- `citations`: citation list used to ground factual claims.
	- item shape: `doc_id`, `start_page`, `end_page`, `chunk_id`.
- `tool_calls`: number of tool-node executions.
- `steps`: total graph step count.
- `trace`: debug/provenance events recorded across nodes.
- `errors`: optional list of error strings.

Trace file naming:
- `<out-dir>/agent_<YYYYMMDD_HHMMSS>_<question_slug>.json`

Evidence truncation behavior:
- For readability, long evidence `text` values are truncated in trace output.
- Default limit is 800 characters per evidence item, followed by `…(truncated)`.

### `rag/lc/state_utils.py` (internal state helper module)

Purpose:
- Initializes and updates LangGraph `AgentState` in a consistent way.
- Centralizes trace event recording for plan, evidence, and answer steps.
- Reduces graph-node duplication by keeping state mutation helpers in one module.

Run:
- This is an internal library module; it is imported by graph code and not run directly.

Flags:
- No CLI flags available.

---

### `eval/day2/run.py`

Purpose:
- Runs retrieval evaluation over `eval/day2/questions.jsonl`.
- Computes `Recall@k`, `MRR@k`, and `nDCG@k`.
- Writes per-question and summary reports.

Run:

```powershell
python -m eval.day2.run
```

Outputs (default):
- `eval/day2/reports/summary.json`
- `eval/day2/reports/per_question.json`

Flags:
- No CLI flags currently.
- `main(...)` supports parameters programmatically:
	- `config_path` (retrieval config JSON)
	- `qpath` (questions JSONL)
	- `outdir` (report output directory)

---

### `eval/day2/ablate.py`

Purpose:
- Runs a batch of ablation variants from a baseline retrieval config.
- Executes eval per variant using `eval.day2.run.main(...)`.
- Produces per-variant folders plus combined CSV leaderboard.

Run:

```powershell
python -m eval.day2.ablate
```

Flags:
- `--baseline` (default: `eval/day2/baselines/day2_hybrid.json`)
	- Baseline config JSON used to generate variants.
- `--qpath` (default: `eval/day2/questions.jsonl`)
	- Evaluation question set.
- `--outroot` (default: `eval/day2/ablations`)
	- Root folder where run outputs are written.
- `--run-name` (default: timestamp)
	- Optional custom name for the run folder.
- `--stop-on-error` (flag)
	- Stop immediately if any variant fails.
- `--keep-old` (flag)
	- Keep existing run directory instead of deleting/recreating it.

Outputs:
- `<outroot>/<run-name>/results.csv`
- `<outroot>/<run-name>/errors.json` (only if failures occur)
- `<outroot>/<run-name>/<variant_name>/summary.json`
- `<outroot>/<run-name>/<variant_name>/per_question.json`

### Script/module dependency map

- rag.search uses rag.retrieve.retrieve
- rag.ask uses rag.retrieve.retrieve and rag.rag_answer.build_cited_answer
- rag.agent.ask uses rag.lc.graph.run_agent and rag.lc.trace.write_trace
- rag.retrieve uses rag.retriever.factory.get_retriever and rag.retriever.bm25_retriever.BM25Retriever
- rag.retriever.bm25_retriever depends on artifact from rag.index_bm25

## Retrieval tuning playbook (algorithm-heavy PDFs)

Core retrieval flags are shared by `rag.search`, `rag.retrieve`, and `rag.ask`.
Answer/output flags (`--show-evidence`, `--json`, `--no-evidence`, `--save-json`) are `rag.ask`-only.

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

- `--no-evidence` (`rag.ask` only; used with `--json`)
	- Omits the `evidence` array from JSON output.
	- Useful for smaller payloads when you only need answer + citations + retrieval settings.

- `--save-json PATH` (`rag.ask` only)
	- Writes the JSON payload to a file path (directories auto-created) in addition to printing JSON.
	- Best paired with `--json` for experiment logs or reproducible QA artifacts.

### See all flags quickly

```powershell
python -m rag.search --help
python -m rag.retrieve --help
python -m rag.ask --help
```

### `rag.ask` JSON output patterns

```powershell
# Include evidence in JSON (default behavior in --json mode)
python -m rag.ask "ML-KEM.Decaps summary" --json

# Exclude evidence for a lighter payload
python -m rag.ask "ML-KEM.Decaps summary" --json --no-evidence

# Save JSON artifact for reports / downstream checks
python -m rag.ask "ML-KEM.Decaps summary" --json --save-json reports/ask_mlkem_decaps.json
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
