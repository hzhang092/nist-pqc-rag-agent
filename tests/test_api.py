from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from rag.retriever.base import ChunkHit
import rag.service as service_module


client = TestClient(app)


class _FakeBackend:
    backend_name = "ollama"
    model_name = "qwen3:8B"

    def __init__(self, *, response: str = "ML-KEM is a key-encapsulation mechanism [c1].") -> None:
        self._response = response

    def generate(self, prompt: str, *, system=None, temperature=None, **kwargs) -> str:
        _ = prompt, system, temperature, kwargs
        return self._response

    def ping(self) -> bool:
        return True


def _hits(n: int = 2) -> list[ChunkHit]:
    return [
        ChunkHit(
            score=1.0 / (idx + 1),
            chunk_id=f"TEST::chunk-{idx + 1}",
            doc_id="NIST.FIPS.203",
            start_page=idx + 1,
            end_page=idx + 1,
            text=f"Evidence text {idx + 1} about ML-KEM and key generation.",
        )
        for idx in range(n)
    ]


def test_service_maps_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(service_module, "get_backend", lambda name=None: _FakeBackend())
    monkeypatch.setattr(
        service_module,
        "retrieve_with_stages_and_timing",
        lambda **kwargs: (
            {
                "final_hits": _hits(1),
                "pre_rerank_fused_hits": _hits(1),
                "post_rerank_hits": _hits(1),
                "rerank_pool": 1,
            },
            {"retrieve_ms": 1.0, "rerank_ms": 0.1},
        ),
    )

    payload = service_module.ask_question("What is ML-KEM?")

    assert payload["refusal_reason"] == "insufficient_evidence"
    assert payload["answer"] == "not found in provided docs"


def test_service_maps_empty_answer(monkeypatch):
    monkeypatch.setattr(service_module, "get_backend", lambda name=None: _FakeBackend(response=""))
    monkeypatch.setattr(
        service_module,
        "retrieve_with_stages_and_timing",
        lambda **kwargs: (
            {
                "final_hits": _hits(2),
                "pre_rerank_fused_hits": _hits(2),
                "post_rerank_hits": _hits(2),
                "rerank_pool": 2,
            },
            {"retrieve_ms": 1.0, "rerank_ms": 0.1},
        ),
    )

    payload = service_module.ask_question("What is ML-KEM?")

    assert payload["refusal_reason"] == "empty_answer"


def test_service_maps_backend_error(monkeypatch):
    class _ErrorBackend(_FakeBackend):
        def generate(self, prompt: str, *, system=None, temperature=None, **kwargs) -> str:
            _ = prompt, system, temperature, kwargs
            raise RuntimeError("backend down")

    monkeypatch.setattr(service_module, "get_backend", lambda name=None: _ErrorBackend())
    monkeypatch.setattr(
        service_module,
        "retrieve_with_stages_and_timing",
        lambda **kwargs: (
            {
                "final_hits": _hits(2),
                "pre_rerank_fused_hits": _hits(2),
                "post_rerank_hits": _hits(2),
                "rerank_pool": 2,
            },
            {"retrieve_ms": 1.0, "rerank_ms": 0.1},
        ),
    )

    payload = service_module.ask_question("What is ML-KEM?")

    assert payload["refusal_reason"] == "backend_error"


def test_service_ask_agent_maps_state(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "run_agent",
        lambda question, k=None: {
            "question": question,
            "original_query": question,
            "canonical_query": "ML-KEM",
            "mode_hint": "definition",
            "required_anchors": [],
            "compare_topics": None,
            "doc_ids": ["NIST.FIPS.203"],
            "doc_family": "FIPS 203",
            "analysis_notes": "test",
            "answer_prompt_question": question,
            "final_answer": "ML-KEM is a key-encapsulation mechanism [c1].",
            "draft_answer": "ML-KEM is a key-encapsulation mechanism [c1].",
            "citations": [
                {
                    "key": "c1",
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 3,
                    "end_page": 3,
                    "chunk_id": "NIST.FIPS.203::p0003::c000",
                }
            ],
            "refusal_reason": "",
            "trace": [{"type": "step", "node": "analyze_query"}],
            "plan": {"action": "resolve_definition"},
            "steps": 5,
            "retrieval_round": 1,
            "tool_calls": 1,
            "stop_reason": "sufficient_evidence",
            "timing_ms": {"analyze": 1.0, "retrieve": 2.0, "generate": 3.0, "total": 7.0},
            "evidence": [{"chunk_id": "c1"}],
        },
    )

    payload = service_module.ask_agent_question("What is ML-KEM?", k=3)

    assert payload["analysis"]["canonical_query"] == "ML-KEM"
    assert payload["analysis"]["doc_ids"] == ["NIST.FIPS.203"]
    assert payload["trace_summary"]["entry_node"] == "analyze_query"
    assert payload["timing_ms"]["analyze"] == 1.0
    assert payload["timing_ms"]["total"] == 7.0


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.main.health_status",
        lambda: {
            "status": "ok",
            "llm_backend": "ollama",
            "llm_model": "qwen3:8B",
            "backend_reachable": True,
            "retrieval_ready": True,
        },
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_backend": "ollama",
        "llm_model": "qwen3:8B",
        "backend_reachable": True,
        "retrieval_ready": True,
    }


def test_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.main.search_query",
        lambda q, k: {
            "query": q,
            "hits": [
                {
                    "chunk_id": "c1",
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 44,
                    "end_page": 44,
                    "score": 0.9,
                    "preview_text": "ML-KEM key generation",
                    "section_path": "7.1 ML-KEM Key Generation",
                    "block_type": "text",
                }
            ],
            "retrieval": {"mode": "hybrid", "backend": "faiss", "k": k},
            "timing_ms": {"retrieve": 1.0, "rerank": 0.2},
        },
    )

    response = client.get("/search", params={"q": "ML-KEM key generation", "k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "ML-KEM key generation"
    assert payload["hits"][0]["section_path"] == "7.1 ML-KEM Key Generation"
    assert payload["hits"][0]["block_type"] == "text"


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.main.ask_question",
        lambda question, k=None: {
            "answer": "ML-KEM is a key-encapsulation mechanism [c1].",
            "citations": [
                {
                    "key": "c1",
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 3,
                    "end_page": 3,
                    "chunk_id": "NIST.FIPS.203::p0003::c000",
                }
            ],
            "refusal_reason": None,
            "trace_summary": {
                "llm_backend": "ollama",
                "llm_model": "qwen3:8B",
                "retrieval_mode": "hybrid",
                "vector_backend": "faiss",
                "query_fusion": True,
                "rerank_enabled": True,
                "retrieved_hit_count": 2,
                "evidence_chunk_count": 2,
                "top_chunk_ids": ["c1"],
            },
            "timing_ms": {"retrieve": 1.0, "rerank": 0.2, "generate": 12.0, "total": 15.0},
        },
    )

    response = client.post("/ask", json={"question": "What is ML-KEM?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("ML-KEM")
    assert payload["citations"][0]["doc_id"] == "NIST.FIPS.203"
    assert payload["trace_summary"]["llm_backend"] == "ollama"
    assert payload["timing_ms"]["generate"] == 12.0


def test_ask_agent_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.main.ask_agent_question",
        lambda question, k=None: {
            "answer": "ML-KEM is a key-encapsulation mechanism [c1].",
            "citations": [
                {
                    "key": "c1",
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 3,
                    "end_page": 3,
                    "chunk_id": "NIST.FIPS.203::p0003::c000",
                }
            ],
            "refusal_reason": None,
            "trace_summary": {
                "entry_node": "analyze_query",
                "mode_hint": "definition",
                "canonical_query": "ML-KEM",
            },
            "timing_ms": {"analyze": 1.0, "retrieve": 2.0, "generate": 12.0, "total": 15.0},
            "analysis": {
                "original_query": question,
                "canonical_query": "ML-KEM",
                "mode_hint": "definition",
                "required_anchors": [],
                "compare_topics": None,
                "doc_ids": ["NIST.FIPS.203"],
                "doc_family": "FIPS 203",
                "analysis_notes": "test",
                "answer_prompt_question": question,
            },
        },
    )

    response = client.post("/ask-agent", json={"question": "What is ML-KEM?", "k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("ML-KEM")
    assert payload["analysis"]["canonical_query"] == "ML-KEM"
    assert payload["trace_summary"]["entry_node"] == "analyze_query"
    assert payload["timing_ms"]["analyze"] == 1.0


def test_ask_agent_endpoint_refusal(monkeypatch):
    monkeypatch.setattr(
        "api.main.ask_agent_question",
        lambda question, k=None: {
            "answer": "I do not have enough citable evidence in the indexed NIST documents to answer that reliably.",
            "citations": [],
            "refusal_reason": "insufficient_evidence",
            "trace_summary": {
                "entry_node": "analyze_query",
                "mode_hint": "general",
                "canonical_query": question,
            },
            "timing_ms": {"analyze": 1.0, "retrieve": 2.0, "generate": 0.0, "total": 3.5},
            "analysis": {
                "original_query": question,
                "canonical_query": question,
                "mode_hint": "general",
                "required_anchors": [],
                "compare_topics": None,
                "doc_ids": [],
                "doc_family": "",
                "analysis_notes": "test",
                "answer_prompt_question": question,
            },
        },
    )

    response = client.post("/ask-agent", json={"question": "What does NIST say about PQC for WiFi 9?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["refusal_reason"] == "insufficient_evidence"
    assert payload["analysis"]["mode_hint"] == "general"
    assert payload["citations"] == []
