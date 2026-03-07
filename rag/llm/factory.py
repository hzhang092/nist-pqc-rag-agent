from __future__ import annotations

from rag.config import SETTINGS

from .base import LLMBackend
from .gemini import GeminiBackend
from .ollama import OllamaBackend


def get_backend(name: str | None = None) -> LLMBackend:
    backend_name = (name or SETTINGS.LLM_BACKEND).strip().lower()
    if backend_name == "gemini":
        return GeminiBackend()
    if backend_name == "ollama":
        return OllamaBackend(
            base_url=SETTINGS.LLM_BASE_URL,
            model_name=SETTINGS.LLM_MODEL,
            timeout_s=SETTINGS.LLM_TIMEOUT_S,
        )
    raise ValueError(f"Unknown LLM backend: {backend_name}")
