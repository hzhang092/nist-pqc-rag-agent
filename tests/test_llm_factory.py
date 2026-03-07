from __future__ import annotations

from dataclasses import replace

import pytest

from rag.llm.factory import get_backend
from rag.llm.ollama import OllamaBackend
import rag.llm.factory as factory_module
import rag.llm.ollama as ollama_module


def test_factory_selects_backend_from_settings(monkeypatch):
    monkeypatch.setattr(
        factory_module,
        "SETTINGS",
        replace(factory_module.SETTINGS, LLM_BACKEND="ollama", LLM_MODEL="qwen3:8B"),
    )

    backend = get_backend()

    assert backend.backend_name == "ollama"
    assert backend.model_name == "qwen3:8B"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_backend("bogus")


def test_ollama_backend_generate_parses_response(monkeypatch):
    calls: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, host: str, timeout: int) -> None:
            calls["host"] = host
            calls["timeout"] = timeout

        def generate(self, **kwargs) -> dict:
            calls["kwargs"] = kwargs
            return {"response": "Grounded answer [c1].", "done": True}

    monkeypatch.setattr(ollama_module, "Client", _FakeClient)

    backend = OllamaBackend(
        base_url="http://localhost:11434",
        model_name="qwen3:8B",
        timeout_s=12,
    )
    text = backend.generate("What is ML-KEM?", system="Be concise.", temperature=0.0)

    assert text == "Grounded answer [c1]."
    assert calls["host"] == "http://localhost:11434"
    assert calls["timeout"] == 12
    assert calls["kwargs"] == {
        "model": "qwen3:8B",
        "prompt": "What is ML-KEM?",
        "stream": False,
        "options": {"temperature": 0.0},
        "system": "Be concise.",
    }


def test_ollama_backend_ping_uses_tags_endpoint(monkeypatch):
    called: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, host: str, timeout: int) -> None:
            called["host"] = host
            called["timeout"] = timeout

        def list(self) -> dict:
            called["list_called"] = True
            return {"models": []}

    monkeypatch.setattr(ollama_module, "Client", _FakeClient)

    backend = OllamaBackend(base_url="http://localhost:11434", model_name="qwen3:8B", timeout_s=20)

    assert backend.ping() is True
    assert called["host"] == "http://localhost:11434"
    assert called["timeout"] == 20
    assert called["list_called"] is True
