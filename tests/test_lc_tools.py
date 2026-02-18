import json
import pytest

from rag.lc import tools as t


# ----------------------------
# Utilities for stubs
# ----------------------------

def _stub_retriever(query: str, k: int = 8, mode_hint=None, filters=None):
    # Mimic your retriever returning ChunkHit-like dicts
    out = []
    for i in range(k):
        out.append({
            "score": 1.0 - i * 0.01,
            "chunk_id": f"c{i}",
            "doc_id": filters.get("doc_id", "NIST.FIPS.203") if filters else "NIST.FIPS.203",
            "start_page": 10 + i,
            "end_page": 10 + i,
            "text": f"Chunk {i} for {query} (mode={mode_hint})"
        })
    return out


def _e(chunk_id: str, doc="D", sp=1, ep=1, score=0.5, text="x"):
    return t.EvidenceItem(score=score, chunk_id=chunk_id, doc_id=doc, start_page=sp, end_page=ep, text=text)


# ----------------------------
# Unit tests (deterministic)
# ----------------------------

def test_mode_hint_from_query():
    assert t._mode_hint_from_query("Algorithm 2 SHAKE128") == "algorithm"
    assert t._mode_hint_from_query("Define ML-KEM") == "definition"
    # a mild “symbolic-ish” trigger
    assert t._mode_hint_from_query("What is η in FIPS?") in ("symbolic", "general")


def test_retrieve_tool_smoke(monkeypatch):
    # Patch retrieval entrypoint discovery to return our stub
    monkeypatch.setattr(t, "_find_retrieve_entrypoint", lambda: _stub_retriever)

    res = t.retrieve.invoke({"query": "What is ML-KEM?", "k": 3})
    assert res["tool"] == "retrieve"
    assert res["query"] == "What is ML-KEM?"
    assert res["k"] == 3
    assert isinstance(res["evidence"], list)
    assert len(res["evidence"]) == 3

    e0 = res["evidence"][0]
    for key in ("score", "chunk_id", "doc_id", "start_page", "end_page", "text"):
        assert key in e0


def test_retrieve_tool_with_doc_filter(monkeypatch):
    monkeypatch.setattr(t, "_find_retrieve_entrypoint", lambda: _stub_retriever)

    res = t.retrieve.invoke({"query": "Algorithm 2 SHAKE128", "k": 2, "doc_id": "NIST.FIPS.205"})
    assert res["filters"]["doc_id"] == "NIST.FIPS.205"
    assert all(e["doc_id"] == "NIST.FIPS.205" for e in res["evidence"])


def test_resolve_definition_query(monkeypatch):
    monkeypatch.setattr(t, "_find_retrieve_entrypoint", lambda: _stub_retriever)

    res = t.resolve_definition.invoke({"term": "SHAKE128", "k": 2})
    assert res["tool"] == "resolve_definition"
    assert "definition of SHAKE128" in res["query"]
    assert len(res["evidence"]) == 2


def test_compare_merges_and_dedupes(monkeypatch):
    # Patch _run_retrieve directly to avoid dealing with mode_hint heuristics
    def fake_run(query: str, k: int = 6, mode_hint=None, filters=None):
        # topic_a returns c0,c1 ; topic_b returns c1,c2 -> dedupe should remove one c1
        if "topicA" in query:
            return [_e("c0"), _e("c1")]
        return [_e("c1"), _e("c2")]

    monkeypatch.setattr(t, "_run_retrieve", fake_run)

    res = t.compare.invoke({"topic_a": "topicA", "topic_b": "topicB", "k": 2})
    chunk_ids = [e["chunk_id"] for e in res["evidence"]]
    assert set(chunk_ids) == {"c0", "c1", "c2"}
    assert res["stats"]["n_merged"] == 3


def test_summarize_reads_chunks_jsonl(tmp_path, monkeypatch):
    # Create a temporary chunks.jsonl file
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "a", "doc_id": "DOC1", "start_page": 1, "end_page": 1, "text": "p1"},
        {"chunk_id": "b", "doc_id": "DOC1", "start_page": 2, "end_page": 2, "text": "p2"},
        {"chunk_id": "c", "doc_id": "DOC2", "start_page": 1, "end_page": 1, "text": "other doc"},
    ]
    with chunks_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Patch module-level constants/cache
    monkeypatch.setattr(t, "CHUNKS_JSONL", chunks_path)
    monkeypatch.setattr(t, "_CHUNKS_META_CACHE", None)

    res = t.summarize.invoke({"doc_id": "DOC1", "start_page": 1, "end_page": 2, "k": 10})
    assert res["tool"] == "summarize"
    assert res["doc_id"] == "DOC1"
    assert len(res["evidence"]) == 2
    assert [e["chunk_id"] for e in res["evidence"]] == ["a", "b"]


# ----------------------------
# Optional: integration test (real retriever)
# ----------------------------
@pytest.mark.skipif(
    not bool(int(__import__("os").environ.get("RUN_REAL_RETRIEVE", "0"))),
    reason="Set RUN_REAL_RETRIEVE=1 to run against real index/pipeline"
)
def test_retrieve_real_pipeline_smoke():
    res = t.retrieve.invoke({"query": "Algorithm 2 SHAKE128", "k": 3})
    assert isinstance(res["evidence"], list)
    assert len(res["evidence"]) > 0
