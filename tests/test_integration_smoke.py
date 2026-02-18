import os
from pathlib import Path
import pytest

# RUN_INTEGRATION=1 pytest -q
# powershell: $env:RUN_INTEGRATION=1; pytest tests/test_integration_smoke.py

@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION", "0") != "1",
    reason="Set RUN_INTEGRATION=1 to run smoke test with real index.",
)
def test_agent_smoke_real_retrieval_no_external_llm(monkeypatch):
    # Skip if index artifacts missing
    if not Path("data/processed/faiss.index").exists():
        pytest.skip("Missing FAISS index at data/processed/faiss.index")

    # Deterministic “LLM” that always returns valid cited bullets using [c1].
    def fake_generate_fn(_prompt: str) -> str:
        return "- Grounded statement [c1].\n- Grounded statement [c1].\n- Grounded statement [c1]."

    # Patch the generator maker used by your graph adapter
    import rag.llm.gemini as gemini
    monkeypatch.setattr(gemini, "make_generate_fn", lambda: fake_generate_fn)

    from rag.lc.graph import run_agent

    state = run_agent("What is ML-KEM?")

    assert state.get("final_answer"), "final_answer should be set"
    assert state.get("citations"), "citations should be non-empty"
    assert state["tool_calls"] >= 1
