# What this graph subsystem currently is

This repo does not implement a full knowledge-graph-backed RAG pipeline. What it actually ships today is a graph-lite sidecar built from `data/processed/chunks.jsonl`, plus a Neo4j export path. The graph encodes four entity types only, `Document`, `Section`, `Algorithm`, and `Term`, and three relationship types only, `IN_DOCUMENT`, `APPEARS_IN`, and `DEFINED_IN`. It is currently an offline organization/export feature, not a live retrieval or answer-generation dependency.

## Files examined most closely

- `reports/project_overview.md`
- `reports/week3_plan.md`
- `reports/week3_progress.md`
- `rag/graph/build_lite.py`
- `rag/graph/helpers.py`
- `rag/graph/types.py`
- `rag/graph/export_neo4j.py`
- `docker-compose.neo4j.yml`
- `tests/test_graph_build_lite.py`
- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`

## Main graph execution path

`data/processed/chunks.jsonl` -> `python -m rag.graph.build_lite` -> `data/processed/graph_lite_nodes.jsonl` + `data/processed/graph_lite_edges.jsonl` -> `python -m rag.graph.export_neo4j` -> `data/processed/neo4j_import/{nodes.csv,edges.csv,constraints.cypher,load.cypher}` -> `docker compose -f docker-compose.neo4j.yml up` -> manual Neo4j import.

I did not find any code path that consumes these graph artifacts during retrieval, reranking, answering, or agent execution. A repo-wide search only found the graph code, the graph test, and the Neo4j compose file.

---

# 1. Scope and intended role

The intended scope in the planning docs is explicitly modest:

- `reports/project_overview.md` describes this as a "scoped graph-lite knowledge organization layer", not a full graph platform.
- `reports/week3_plan.md` explicitly says not to attempt a full knowledge graph and instead add a lightweight layer over documents, sections, algorithms, and terms.

That is consistent with the implementation. The graph builder does not do open IE, dependency parsing, ontology management, graph embeddings, or graph-aware retrieval. It performs deterministic entity extraction from already-built chunk metadata and emits a small, stable graph that reflects document structure and a narrow set of seeded concepts.

# 2. End-to-end graph workflow

## 2.1 Input contract

The graph builder reads `data/processed/chunks.jsonl` from the existing ingestion/chunking pipeline. It depends on these chunk fields:

- `doc_id`
- `start_page`
- `end_page`
- `section_path`
- `text`

It does not read embeddings, FAISS, BM25, or answer traces.

## 2.2 Build flow

`rag/graph/build_lite.py` performs one pass over all chunks and accumulates:

- document page spans
- section page spans for every cumulative `section_path` prefix
- best anchor metadata for algorithms
- term mentions by section and by document
- a subset of term mentions marked as definitions based on section-title heuristics

After the accumulation pass, it materializes nodes first, then edges, then sorts both deterministically and writes JSONL.

## 2.3 Export flow

`rag/graph/export_neo4j.py` reads the graph-lite JSONL artifacts and rewrites them into:

- `nodes.csv`
- `edges.csv`
- `constraints.cypher`
- `load.cypher`

The export format is optimized for Neo4j CSV import plus APOC-based dynamic relationship creation.

## 2.4 Runtime position

The graph is not part of the main RAG serving loop:

- no retriever reads `graph_lite_nodes.jsonl` or `graph_lite_edges.jsonl`
- no LangGraph node calls Neo4j
- no API endpoint exposes graph queries

The current role is organizational and demonstrative, not operational.

## 2.5 Workflow diagram

```text
chunks.jsonl
  |
  v
rag.graph.build_lite
  |
  |-- scans chunk text, doc ids, pages, section paths
  |-- derives section prefixes
  |-- detects algorithm headers
  |-- matches seeded terms
  v
graph_lite_nodes.jsonl
graph_lite_edges.jsonl
  |
  v
rag.graph.export_neo4j
  |
  |-- nodes.csv
  |-- edges.csv
  |-- constraints.cypher
  |-- load.cypher
  v
docker-compose.neo4j.yml
  |
  v
Neo4j instance with :Entity nodes and dynamic relationship types
```

# 3. Core graph data model

The model is defined in `rag/graph/types.py`.

## 3.1 Node schema

`GraphNode` fields:

- `node_id`
- `label`
- `doc_id`
- `start_page`
- `end_page`
- `display_name`
- `properties`

## 3.2 Edge schema

`GraphEdge` fields:

- `edge_id`
- `type`
- `source_id`
- `target_id`
- `doc_id`
- `start_page`
- `end_page`
- `properties`

## 3.3 ID strategy

`rag/graph/helpers.py` defines deterministic ID constructors:

- document node: `doc::<doc_id>`
- section node: `section::<doc_id>::<full_section_path>`
- algorithm node: `alg::<doc_id>::<algorithm_key>`
- term node: `term::<normalized_term>`
- edge: `edge::<TYPE>::<source_id>::<target_id>`

This matters because the graph is rebuilt from artifacts rather than updated transactionally. Stable IDs are the main deduplication mechanism.

# 4. Node construction mechanisms

## 4.1 Document nodes

Construction logic:

- Every chunk with a valid `doc_id` contributes to `document_spans`.
- `_merge_page_span()` merges page ranges using min start and max end.
- After the pass, one `Document` node is created per distinct `doc_id`.

Properties:

- `display_name = doc_id`
- `properties = {"doc_id": doc_id}`

Observations:

- This is a very lightweight node type.
- The page span reflects the whole observed document range in the chunk set.

## 4.2 Section nodes

Section construction is the most structurally important part of the graph.

Mechanism:

- `get_section_path()` accepts either a list or a string.
- If it receives a string containing `>`, it splits conservatively on `>`.
- For each chunk, the builder creates nodes for every cumulative prefix of the section path, not only the deepest leaf.

Example:

If a chunk has:

```text
5. External Functions > 5.1 ML-DSA Key Generation > Algorithm 1 ML-DSA.KeyGen ()
```

the builder creates three section nodes:

- `5. External Functions`
- `5. External Functions > 5.1 ML-DSA Key Generation`
- `5. External Functions > 5.1 ML-DSA Key Generation > Algorithm 1 ML-DSA.KeyGen ()`

Properties:

- `full_section_path`
- `leaf_title`
- `depth`

Important design choice:

- The builder does not create parent-child section edges.
- Hierarchy is implicit in the `full_section_path` string and the existence of cumulative prefix nodes.
- Each section node is only connected explicitly to its document via `IN_DOCUMENT`.

Span behavior:

- `section_spans` are merged across all chunks that share the same prefix.
- Top-level sections therefore get broader page spans than leaf sections.

## 4.3 Algorithm nodes

Algorithm construction is intentionally conservative.

Detection logic from `rag/graph/helpers.py`:

- `Algorithm <number>`
- `ML-KEM.(KeyGen|Encaps|Decaps)`
- `ML-DSA.<identifier>`
- `SLH-DSA.<identifier>`

But `build_lite.py` calls `detect_algorithms(text, header_only=True)`, which changes the behavior materially:

- the search scope is restricted to lines matching `^\s*Algorithm\s+\d+\b[^\n]*`
- algorithm-like strings in prose are ignored
- algorithm names are only captured when they appear in an algorithm header line

Anchor selection behavior:

- front matter is excluded via `is_front_matter_section()`
- for each algorithm ID, the builder stores only one anchor section/page
- if multiple candidate chunks mention the same algorithm header, `_should_replace_algorithm_anchor()` keeps the best one

Anchor priority order:

1. section leaf titles starting with `Algorithm `
2. earlier page number
3. deeper section depth
4. lexical `section_id` tie-break

This is a good example of deterministic disambiguation. It avoids later references or table-of-contents noise becoming the canonical node location.

What becomes an algorithm node:

- numeric algorithm IDs such as `alg::NIST.FIPS.203::13`
- normalized named algorithms such as `alg::DOC::ml-dsa.keygen`

What does not become an algorithm node:

- prose phrases like `Compress algorithm 256 times`
- malformed glued identifiers like `ML-DSA.These`
- front matter mentions from `List of Tables`, `List of Algorithms`, etc.

This behavior is covered directly by `tests/test_graph_build_lite.py`.

## 4.4 Term nodes

Term extraction is seed-based, not model-based.

The only term nodes currently supported are the hardcoded `TERM_SEEDS`:

- `ML-KEM`
- `ML-DSA`
- `SLH-DSA`
- `encapsulation`
- `decapsulation`
- `key generation`
- `public key`
- `secret key`
- `ciphertext`
- `parameter set`

Mechanism:

- for each chunk, `term_occurs_in_text()` performs normalized substring matching
- matches update:
  - `term_doc_mentions`
  - `term_section_mentions`
  - `term_definitions` when the section path looks definition-like

Definition-like sections are recognized heuristically if any path element contains one of:

- `terms`
- `notation`
- `definitions`
- `acronyms`
- `symbols`

Important design choices:

- term nodes are global across documents because their IDs do not include `doc_id`
- term nodes have `doc_id = None`, `start_page = None`, `end_page = None`
- the document grounding of terms is expressed only through edges

This produces a cross-document concept hub, but only for ten seeded terms.

# 5. Edge construction mechanisms

There are only three edge types in the current graph.

## 5.1 `IN_DOCUMENT`

Used for:

- `Section -> Document`
- `Algorithm -> Document`
- `Term -> Document`

Purpose:

- attaches every non-document entity to one or more source documents
- preserves corpus membership even when the node itself is global, as with `Term`

Construction details:

- section `IN_DOCUMENT` edges are created during section-node materialization
- algorithm `IN_DOCUMENT` edges are created from the winning algorithm anchor metadata
- term `IN_DOCUMENT` edges are created from the set of documents in which each seed term occurred

## 5.2 `APPEARS_IN`

Used for:

- `Algorithm -> Section`
- `Term -> Section`

Purpose:

- links an entity to structural locations where it appears

Construction details:

- algorithm `APPEARS_IN` uses exactly one section, the chosen anchor section
- term `APPEARS_IN` is created for every mentioned section that is not marked definition-like

This means algorithm nodes behave like anchored definitions, while term nodes behave like multi-location mentions.

## 5.3 `DEFINED_IN`

Used only for:

- `Term -> Section`

Purpose:

- distinguishes mention from likely definitional context

Construction details:

- if the section path is "definition-like", the term-to-section edge becomes `DEFINED_IN`
- otherwise it becomes `APPEARS_IN`

This is the only semantic distinction beyond plain containment in the graph.

## 5.4 Missing edge types

What the graph does not encode is just as important:

- no `SECTION_CHILD_OF`
- no `DOCUMENT_REFERENCES_DOCUMENT`
- no `ALGORITHM_CALLS_ALGORITHM`
- no `SECTION_REFERENCES_SECTION`
- no `TERM_RELATED_TO_TERM`
- no `CHUNK_IN_SECTION`
- no `PAGE_IN_DOCUMENT`

So the graph is mainly a navigable index over structure, not a rich relational model.

# 6. Determinism and artifact stability

The graph builder is deliberately deterministic.

Determinism mechanisms visible in code:

- stable ID constructors
- set-based accumulation followed by sorted emission
- node sort key: `(label, doc_id or "", node_id)`
- edge sort key: `(type, source_id, target_id, doc_id or "")`
- JSONL writing with `sort_keys=True`

This matches the repo’s broader data-contract emphasis on stable artifacts.

# 7. Current graph artifacts in the repo

The graph artifacts already exist on disk:

- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`

I inspected those files directly.

## 7.1 Current node counts

- `Document`: 6
- `Section`: 369
- `Algorithm`: 102
- `Term`: 10

Total nodes: 487

## 7.2 Current edge counts

- `APPEARS_IN`: 655
- `DEFINED_IN`: 22
- `IN_DOCUMENT`: 516

Total edges: 1,193

## 7.3 Distribution by document

Section nodes by `doc_id`:

- `NIST.FIPS.203`: 71
- `NIST.FIPS.204`: 91
- `NIST.FIPS.205`: 74
- `NIST.IR.8545`: 25
- `NIST.IR.8547.ipd`: 55
- `NIST.SP.800-227`: 53

Algorithm nodes by `doc_id`:

- `NIST.FIPS.203`: 24
- `NIST.FIPS.204`: 53
- `NIST.FIPS.205`: 25

Observation:

- algorithm nodes currently appear only in the three FIPS standards
- the IR/SP documents contribute sections and term/document structure but no detected algorithms in the current artifacts

## 7.4 Concrete artifact shape

Example node shape observed in `graph_lite_nodes.jsonl`:

```json
{
  "display_name": "1",
  "doc_id": "NIST.FIPS.203",
  "end_page": 17,
  "label": "Algorithm",
  "node_id": "alg::NIST.FIPS.203::1",
  "properties": {"name": "1"},
  "start_page": 17
}
```

Example edge shape observed in `graph_lite_edges.jsonl`:

```json
{
  "doc_id": "NIST.FIPS.203",
  "edge_id": "edge::APPEARS_IN::alg::NIST.FIPS.203::1::section::NIST.FIPS.203::2. Terms, Acronyms, and Notation > 2.4 Interpreting the Pseudocode > 2.4.3 Arithmetic With Arrays of Integers > Algorithm 1 ForExample ()",
  "end_page": 17,
  "properties": {},
  "source_id": "alg::NIST.FIPS.203::1",
  "start_page": 17,
  "target_id": "section::NIST.FIPS.203::2. Terms, Acronyms, and Notation > 2.4 Interpreting the Pseudocode > 2.4.3 Arithmetic With Arrays of Integers > Algorithm 1 ForExample ()",
  "type": "APPEARS_IN"
}
```

# 8. Test coverage and what it proves

Graph-specific test coverage is currently narrow but useful.

`tests/test_graph_build_lite.py` proves:

- front matter algorithm mentions do not anchor algorithm nodes
- prose mentions like `algorithm 256` do not create false algorithm nodes
- malformed glued prose like `ML-DSA.These` does not become an algorithm
- a real algorithm block creates both numeric and named algorithm nodes
- the winning algorithm anchor is the real definition section, not a later reference

I also ran the existing graph test in the `pyt` environment on 2026-03-13:

```text
conda run -n pyt python -m pytest -q tests/test_graph_build_lite.py
1 passed in 0.04s
```

What is not covered by tests, at least from the files I examined:

- term extraction precision/recall
- section-prefix expansion behavior
- Neo4j export correctness
- end-to-end import into a running Neo4j container

# 9. Neo4j export and usage

## 9.1 Export format

`rag/graph/export_neo4j.py` does a straightforward transformation:

- reads graph-lite JSONL
- writes `nodes.csv`
- writes `edges.csv`
- writes `constraints.cypher`
- writes `load.cypher`

## 9.2 Neo4j node model

A key design choice: the exporter does not map graph node types to distinct Neo4j labels.

Instead, `load.cypher` creates all nodes as:

```cypher
(n:Entity {node_id: ...})
```

and stores the graph type in a property:

- `n.label = "Document" | "Section" | "Algorithm" | "Term"`

Implications:

- all queries must filter on `n.label`
- you do not get label-specific indexes or schema by default
- the graph is simpler to import, but less idiomatic as a Neo4j domain model

## 9.3 Neo4j relationship model

Relationships are created dynamically with APOC:

- `row.:TYPE` becomes the relationship type
- properties include `edge_id`, `doc_id`, `start_page`, `end_page`, `json_properties`

This is why `docker-compose.neo4j.yml` enables APOC and unrestricted procedures.

## 9.4 Exported properties

Node CSV columns:

- `node_id:ID`
- `label`
- `doc_id`
- `start_page:int`
- `end_page:int`
- `display_name`
- `json_properties`

Edge CSV columns:

- `edge_id`
- `:START_ID`
- `:END_ID`
- `:TYPE`
- `doc_id`
- `start_page:int`
- `end_page:int`
- `json_properties`

Important nuance:

- `json_properties` is not actual JSON serialization
- the exporter uses `str(props)`, which produces a Python dict string representation

That is fine for archival/debug use, but it is not directly queryable as a structured Neo4j map.

## 9.5 Constraint model

The only generated constraint is:

```cypher
CREATE CONSTRAINT node_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.node_id IS UNIQUE;
```

There are no constraints or indexes on:

- `label`
- `doc_id`
- `display_name`
- relationship properties

## 9.6 Expected usage path

The repo gives you the pieces for Neo4j import, but not a complete scripted load workflow.

What exists:

- a compose file for Neo4j
- generated CSVs and Cypher files

What I did not find:

- a Make target
- a shell script
- a Python loader that executes the import automatically
- any app code that queries Neo4j after import

So the intended usage appears to be:

1. build graph-lite JSONL
2. export Neo4j import files
3. start Neo4j via `docker-compose.neo4j.yml`
4. run `constraints.cypher` and `load.cypher` manually in Neo4j Browser or `cypher-shell`

That last step is an inference from the generated files and compose setup, because no import runbook is checked in.

# 10. Dockerized Neo4j path

`docker-compose.neo4j.yml` defines one service:

- image: `neo4j:5`
- container name: `pqc-neo4j`
- ports: `7474` and `7687`
- auth: `neo4j/password123`
- heap: `512m`
- plugin: `apoc`

Mounted volumes:

- `./data/processed/neo4j_import:/var/lib/neo4j/import`
- `./data/neo4j/data:/data`
- `./data/neo4j/logs:/logs`

Important operational detail from the current workspace:

- `data/processed/neo4j_import` is currently owned by uid `7474` with mode `700`
- `data/neo4j/data` and `data/neo4j/logs` are also owned by the Neo4j container user

That means the host user cannot currently inspect `data/processed/neo4j_import` contents directly. I hit `PermissionError: [Errno 13] Permission denied` when trying to read `nodes.csv` and `edges.csv`.

Practical implication:

- the export code clearly defines the intended import format
- but the checked-in host-side Neo4j import artifacts are not readable in this workspace state
- host-side reruns of the exporter will likely need permission/ownership cleanup first

# 11. How to use the graph subsystem as it currently exists

## 11.1 Build graph-lite artifacts

```bash
python -m rag.graph.build_lite
```

Inputs:

- `data/processed/chunks.jsonl`

Outputs:

- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`

## 11.2 Export for Neo4j

```bash
python -m rag.graph.export_neo4j
```

Outputs:

- `data/processed/neo4j_import/nodes.csv`
- `data/processed/neo4j_import/edges.csv`
- `data/processed/neo4j_import/constraints.cypher`
- `data/processed/neo4j_import/load.cypher`

## 11.3 Start Neo4j

```bash
docker compose -f docker-compose.neo4j.yml up -d
```

## 11.4 Load the graph

The repo does not provide an automated import wrapper. The practical path is to run the generated `constraints.cypher` and `load.cypher` manually in Neo4j after the container starts.

## 11.5 Example Cypher queries after import

Count entity types:

```cypher
MATCH (n:Entity)
RETURN n.label, count(*) AS count
ORDER BY n.label;
```

Inspect algorithm anchors:

```cypher
MATCH (a:Entity {label: "Algorithm"})-[r:APPEARS_IN]->(s:Entity {label: "Section"})
RETURN a.display_name, a.doc_id, s.display_name, r.start_page
ORDER BY a.doc_id, r.start_page
LIMIT 25;
```

Inspect term definitions:

```cypher
MATCH (t:Entity {label: "Term"})-[r:DEFINED_IN]->(s:Entity {label: "Section"})
RETURN t.display_name, s.doc_id, s.display_name, r.start_page
ORDER BY t.display_name, s.doc_id;
```

Document-local neighborhood for a seeded term:

```cypher
MATCH (t:Entity {label: "Term", display_name: "ML-KEM"})-[:IN_DOCUMENT]->(d:Entity {label: "Document"})
MATCH (t)-[r:APPEARS_IN|DEFINED_IN]->(s:Entity {label: "Section"})-[:IN_DOCUMENT]->(d)
RETURN d.display_name, type(r), s.display_name, r.start_page
ORDER BY d.display_name, r.start_page;
```

# 12. What is implemented versus not implemented

## Implemented

- deterministic graph-lite builder over chunk artifacts
- four node types and three edge types
- algorithm-header anchoring logic with false-positive guards
- seeded term extraction with definitional section heuristics
- deterministic JSONL output
- Neo4j CSV/Cypher export
- Neo4j Docker compose setup with APOC

## Partial or limited

- terms are only a hardcoded seed list of ten concepts
- definition detection is heuristic and section-title-based
- section hierarchy exists only implicitly in strings, not as explicit edges
- algorithm modeling captures anchored occurrence, not internal call structure

## Not implemented

- graph-backed retrieval
- graph-aware reranking
- graph-aware answer synthesis
- live Neo4j queries from the API or agent
- automatic Neo4j import orchestration
- richer entity types such as table, figure, page, chunk, citation, or requirement
- richer relation extraction such as references, invokes, depends-on, contrasts-with, or synonymy

# 13. Concise engineering assessment

## Strongest aspects

- The graph builder is deterministic and easy to reason about.
- Algorithm anchoring is thoughtfully constrained and has a targeted regression test.
- The graph is grounded in page spans and document IDs, which keeps it aligned with the repo’s citation discipline.
- The Neo4j export path is simple enough that another engineer can inspect and extend it quickly.

## Most important limitations

- This is not yet a knowledge-graph-powered application; it is an offline sidecar.
- Term extraction is intentionally narrow and may miss many useful PQC concepts.
- Section hierarchy is represented textually rather than relationally.
- Neo4j import is only semi-packaged; the database starts in Docker, but import remains manual.
- The exported `json_properties` field is a Python string repr, not real JSON.
- Current `neo4j_import` directory permissions are a real usability problem for host-side inspection and regeneration.

## Highest-value next improvements

1. Add explicit hierarchy edges such as `CHILD_OF` for sections so navigation does not depend on parsing `full_section_path`.
2. Replace seed-only term extraction with a controlled extractor over definitions, tables, and identifier patterns already present in the chunk metadata.
3. Make Neo4j import fully scripted and fix ownership/permission handling for `data/processed/neo4j_import`.
4. Promote node types to real Neo4j labels instead of storing the type only in `n.label`.
5. Integrate at least one graph-assisted retrieval or agent path so the graph affects user-facing behavior rather than remaining a side artifact.

---

# Quick start for a new engineer

If you only want to understand the graph path, read these files in this order:

1. `rag/graph/build_lite.py`
2. `rag/graph/helpers.py`
3. `tests/test_graph_build_lite.py`
4. `rag/graph/export_neo4j.py`
5. `docker-compose.neo4j.yml`

Then inspect the shipped artifacts:

- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`

Then run:

```bash
python -m rag.graph.build_lite
python -m rag.graph.export_neo4j
docker compose -f docker-compose.neo4j.yml up -d
```

At that point, the only missing step is manual execution of the generated Cypher import files in Neo4j.

# Gaps / risks / next steps

- The graph feature is honest but narrow: it demonstrates knowledge organization, not graph-native question answering.
- The current on-disk graph artifacts are useful and already populated, but they are disconnected from the live RAG path.
- The biggest engineering risk is not correctness of the builder; it is product incompleteness around Neo4j import and downstream usage.
- If this subsystem is meant to matter architecturally, the next step should be to pick one concrete graph-assisted workflow and connect it to retrieval or agent evidence selection.

---

# Addendum — Post-Upgrade Repo Review (2026-03-14)

This addendum reflects the graph-lite upgrade that shipped after the original review above. The earlier sections are still useful for understanding the pre-upgrade design, but several of the original limitations are no longer accurate:

- term extraction is no longer seed-only
- section hierarchy is no longer only implicit in strings
- the LangGraph path now has one narrow graph-assisted lookup flow
- Neo4j is still not part of the live LangGraph runtime path

The current graph subsystem is still scoped and still honest about its role: it remains a deterministic sidecar plus a small runtime support layer for definition-style query analysis. It is not a graph-native retriever, not a graph-backed answer engine, and not a live Neo4j application.

## Files examined for this addendum

- `rag/graph/build_lite.py`
- `rag/graph/helpers.py`
- `rag/graph/query.py`
- `rag/graph/export_neo4j.py`
- `rag/lc/graph.py`
- `rag/lc/state.py`
- `rag/lc/state_utils.py`
- `rag/lc/trace.py`
- `tests/test_graph_build_lite.py`
- `tests/test_graph_export_neo4j.py`
- `tests/test_graph_query.py`
- `tests/test_lc_graph.py`
- `eval/graph_definition_sanity.jsonl`
- `eval/graph_definition_sanity.py`
- `reports/eval/graph_definition_sanity.md`
- `data/processed/graph_lite_nodes.jsonl`
- `data/processed/graph_lite_edges.jsonl`

## 1. What changed in the graph build

### 1.1 Build flow is now an explicit 3-pass deterministic pipeline

`rag/graph/build_lite.py` no longer does one flat accumulation pass. It now splits work into:

1. structure pass
   - collects document spans
   - collects cumulative section prefixes
   - records explicit section parent-child pairs
   - extracts canonical numbered algorithm anchors from actual algorithm headers
2. term aggregation pass
   - scans chunks for deterministic term candidates
   - aggregates evidence across chunks, sections, and documents
   - tracks candidate source type, surface forms, chunk support, and anchor attachment
3. materialization pass
   - emits nodes only after candidate validation is complete
   - emits edges only after the final node set is known
   - sorts both node and edge rows deterministically before JSONL write

This is a meaningful architectural upgrade because terms are no longer emitted opportunistically from a single local hit. They are now admitted only after cross-chunk validation.

### 1.2 Section hierarchy is now explicit

The old review correctly noted that section hierarchy was represented only through cumulative path strings. That is no longer true.

Current mechanism:

- for every cumulative `section_path` prefix, the builder still creates a `Section` node
- while doing that, it also records `(child_section_id, parent_section_id)` pairs
- during materialization it emits `CHILD_OF` edges from the deeper prefix to its immediate parent

This means section navigation no longer depends entirely on reparsing `full_section_path`.

Observed current artifact count after rebuild:

- `CHILD_OF`: 284 edges

Example current edge shape:

```json
{
  "type": "CHILD_OF",
  "source_id": "section::NIST.FIPS.203::1. Introduction > 1.1 Purpose and Scope",
  "target_id": "section::NIST.FIPS.203::1. Introduction"
}
```

### 1.3 Algorithm nodes are now canonical numbered nodes with richer properties

The old graph created numeric algorithm nodes and, in some cases, separate normalized named algorithm nodes. That duplicate modeling has been removed.

Current mechanism:

- `rag/graph/helpers.py` now exposes `extract_algorithm_header_info(text)`
- it parses only true algorithm header lines matched by `ALGORITHM_HEADER_PATTERN`
- for each header it extracts:
  - `algorithm_number`
  - `algorithm_label`
  - `algorithm_name`
  - `raw_header`
- `build_lite.py` uses the numeric algorithm number as the canonical ID key:
  - `alg::<doc_id>::<algorithm_number>`
- if multiple chunks could anchor the same numbered algorithm, the existing deterministic anchor priority still applies

Current `Algorithm` node properties now include:

- `algorithm_number`
- `algorithm_label`
- `algorithm_name`
- `raw_header`
- `section_id`

This directly improves lookup and traceability because runtime code can now resolve `Algorithm 19` style anchors against one canonical node form.

### 1.4 Term extraction is no longer a fixed ten-term seed list

This is the biggest functional change in the graph builder.

The old version only matched a hardcoded `TERM_SEEDS` list. The current version uses deterministic multi-source extraction in `rag/graph/helpers.py` via `extract_term_candidates(...)`.

Current candidate sources:

- definition-like sections
  - detected from section path hints such as `terms`, `definitions`, `notation`, `acronyms`, `symbols`
  - definition rows are extracted with a pipe-delimited heuristic such as `term | definition`
- identifier/acronym regexes
  - examples include `ML-KEM`, `ML-DSA`, `SLH-DSA`, dotted identifiers, and bounded acronym-like spans such as `SHAKE128`
- algorithm headers
  - `algorithm_name` values from parsed headers
  - operation phrases visible in headers
- section headings
  - only domain-looking identifiers or operation phrases are retained

The builder aggregates candidate evidence across chunks and then validates each normalized term with this rule:

- keep the term if it appears in a definition-like section, or
- keep the term if it matched the identifier regex path, or
- keep the term if it appears in at least 2 chunks, or
- keep the term if it is attached to a section/algorithm anchor

This is still heuristic, but it is materially broader and more defensible than the old fixed seed list because every term now has traceable support.

### 1.5 Term nodes carry richer metadata

Current `Term` node properties include:

- `normalized_term`
- `surface_forms`
- `term_type`
- `definition_strength`

Current `term_type` values come from deterministic classification in `classify_term_type(...)` and can be:

- `identifier`
- `acronym`
- `operation`
- `concept`
- `symbol`

Current `definition_strength` in this shipped slice is:

- `heuristic_definition_section` if the term had definition-like section evidence
- `seed` otherwise

The label `seed` is now being used as a fallback strength bucket, not as a claim that the term came from the old fixed term seed list.

### 1.6 Current artifact shape and counts

After the rebuilt artifacts were written from the upgraded builder, the current graph-lite JSONL counts are:

- nodes: 660 total
  - `Document`: 6
  - `Section`: 369
  - `Algorithm`: 87
  - `Term`: 198
- edges: 1,840 total
  - `IN_DOCUMENT`: 727
  - `APPEARS_IN`: 703
  - `CHILD_OF`: 284
  - `DEFINED_IN`: 126

That is a real shift in graph density and usefulness relative to the older sidecar described above.

## 2. What changed in the runtime path

### 2.1 There is now a real graph-assisted LangGraph hook

The earlier report said the graph artifacts were not consumed during agent execution. That is no longer fully true.

There is now one narrow runtime integration point:

- definition-mode query analysis in `rag/lc/graph.py`

The integration is intentionally scoped:

- no graph-assisted retrieval ranking
- no graph traversal during answer generation
- no graph-aware reranking
- no Neo4j dependency

### 2.2 The runtime lookup path is file-backed, not database-backed

This is the most important implementation fact to state clearly.

Current live runtime path:

```text
/ask-agent
  -> rag.lc.graph.node_analyze_query
  -> rag.graph.query.lookup_term
  -> data/processed/graph_lite_nodes.jsonl
  -> data/processed/graph_lite_edges.jsonl
```

Current LangGraph code imports:

```python
from rag.graph.query import lookup_term
```

There is no Neo4j client import in `rag/lc/graph.py`, no Bolt connection setup, no Cypher execution, and no runtime call into `export_neo4j.py`.

So the answer to “is Neo4j actually used in the LangGraph path?” is:

- no, Neo4j is not used in the LangGraph path
- yes, the LangGraph path now uses graph data
- but that graph data is read from the JSONL sidecar through `rag/graph/query.py`, not from Neo4j

### 2.3 How `rag/graph/query.py` works

`rag/graph/query.py` is a new file-backed lookup layer over the prebuilt JSONL artifacts.

Mechanism:

- uses `NODES_PATH = data/processed/graph_lite_nodes.jsonl`
- uses `EDGES_PATH = data/processed/graph_lite_edges.jsonl`
- lazily builds an in-memory cached index with `@lru_cache(maxsize=1)`
- stores:
  - `nodes`
  - outgoing edges by source node
  - term lookup maps by:
    - normalized term
    - surface form
    - display name

Match priority:

1. exact normalized term
2. surface form
3. display name

Return payload from `lookup_term(...)`:

- `matched_entities`
- `candidate_doc_ids`
- `candidate_section_ids`
- `required_anchors`
- `match_reason`

This is deliberately simple and bounded. It gives the agent just enough structure to improve query analysis without creating a graph-database dependency.

### 2.4 How `rag/lc/graph.py` applies the lookup

The runtime integration lives in `_apply_definition_graph_lookup(...)` and is called from `node_analyze_query(...)`.

Behavior:

- only runs when:
  - graph lookup is enabled, and
  - `analysis.mode_hint == "definition"`
- uses `analysis.canonical_query` or `analysis.original_query` as the lookup value
- calls `lookup_term(...)`
- if there is a match:
  - merges graph-provided `candidate_doc_ids` into `analysis.doc_ids`
  - merges graph-provided anchors into:
    - `required_anchors`
    - `protected_spans`
- records graph lookup debug state in:
  - `state["graph_lookup"]`
  - trace events of type `graph_lookup_applied`

Important boundary:

- `candidate_section_ids` are recorded for debug/trace only
- they do not currently drive retrieval filtering
- retrieval remains doc/chunk-centric, consistent with the repo’s current retrieval interface

That design matches the stated scope discipline: the graph assists analysis and narrowing, but does not become the retrieval engine.

### 2.5 Graph lookup is switchable for tests and controlled runs

`run_agent(...)` now accepts:

- `use_graph_lookup: bool = True`

`init_state(...)` stores:

- `graph_lookup_enabled`
- `graph_lookup`

This matters because it gives the repo a clean way to compare before/after graph-assisted query analysis behavior without branching the main agent flow.

## 3. Neo4j status after the update

### 3.1 Neo4j export improved, but remains offline-only

`rag/graph/export_neo4j.py` was improved in one practical way:

- `json_properties` is now serialized with `json.dumps(..., sort_keys=True)`

This fixes one of the explicit issues in the original review: properties are now emitted as real JSON strings instead of Python dict repr strings.

### 3.2 What Neo4j still is in this repo

Neo4j is still:

- an export target
- a demo / inspection path
- a manual import workflow

Neo4j is still not:

- queried by `rag/lc/graph.py`
- queried by `rag/service.py`
- queried by any retriever
- a prerequisite for `/ask-agent`
- a dependency of the shipped runtime lookup path

### 3.3 Neo4j import packaging is still incomplete

The original review’s operational point remains true:

- the export path exists
- `docker-compose.neo4j.yml` exists
- APOC-based load scripts exist
- but import is still an offline path rather than a first-class runtime service

The `neo4j_import` directory permissions issue also remains a practical usability caveat in this workspace state.

## 4. Tests and validation added with the upgrade

The graph upgrade shipped with new targeted coverage.

### 4.1 Graph build tests

`tests/test_graph_build_lite.py` now verifies:

- canonical numbered algorithm nodes
- rich algorithm properties
- `CHILD_OF` edges
- deterministic term extraction and rejection of generic noise
- byte-identical output across repeated builds

### 4.2 Neo4j export test

`tests/test_graph_export_neo4j.py` verifies:

- `json_properties` in exported CSV rows is valid JSON

### 4.3 Query lookup test

`tests/test_graph_query.py` verifies:

- `lookup_term(...)` returns expected:
  - candidate doc IDs
  - candidate section IDs
  - required anchors

### 4.4 LangGraph integration test

`tests/test_lc_graph.py` now includes graph-specific checks that verify:

- definition-mode analyze-query applies graph lookup
- graph-derived `doc_ids` and anchors are merged into state
- graph lookup can be disabled cleanly

## 5. Eval artifact added with the upgrade

The repo now includes a tiny graph-specific evaluation artifact:

- dataset: `eval/graph_definition_sanity.jsonl`
- runner: `eval/graph_definition_sanity.py`
- output report: `reports/eval/graph_definition_sanity.md`

This is not a full retrieval ablation. It is intentionally narrower.

What it measures:

- whether graph lookup improves doc narrowing
- whether graph lookup improves required-anchor enrichment
- whether the analyzed request becomes more constrained before retrieval

Observed current result from the generated report:

- 4/4 sanity queries improved anchor quality
- 1/4 sanity queries improved doc narrowing from no doc scope to the expected doc scope

That is a good fit for the actual shipped feature, because the feature is an analysis-time graph assist, not a ranking rewrite.

## 6. Revised engineering assessment after the upgrade

### Stronger than before

- The graph builder is still deterministic, but now materially more useful.
- The section hierarchy is now explicit enough to support navigation-oriented use cases.
- The graph now affects real runtime behavior in a narrow and inspectable way.
- The runtime graph hook respects the repo’s retriever abstraction by operating before retrieval rather than inside backend-specific ranking code.
- The Neo4j export path is slightly more correct because property serialization is now real JSON.

### Still true limitations

- This is still not a graph-native RAG system.
- Neo4j is still not part of the live LangGraph path.
- `candidate_section_ids` are still observational/debug signals, not retrieval constraints.
- The graph does not yet support section-reference edges, near-algorithm edges, or richer cross-entity semantics.
- The runtime graph usage is intentionally limited to definition-mode analysis, not general question answering.

### Bottom-line status

The graph subsystem has moved from:

- offline sidecar only

to:

- deterministic graph sidecar
- plus a narrow JSONL-backed lookup layer in the LangGraph analyze-query step

It has not moved to:

- Neo4j-backed runtime querying
- graph-aware retrieval ranking
- graph-driven answer synthesis

That is the honest, repo-grounded way to describe the current implementation.
