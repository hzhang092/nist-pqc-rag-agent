# rag/llm/gemini.py
from __future__ import annotations

import os
import time
from typing import Any, Callable

from rag.config import SETTINGS

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

MODEL = "gemini-3-flash-preview"


def get_model_name() -> str:
    """Returns the effective Gemini model name from environment/config."""
    configured_model = SETTINGS.LLM_MODEL.strip()
    if configured_model:
        return configured_model
    return os.getenv("GEMINI_MODEL", MODEL)


def _resolve_client_kwargs() -> dict[str, object]:
    """
    Resolves explicit `google.genai.Client(...)` kwargs from env vars.

    Uses Google AI Studio API key mode only.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return {"api_key": api_key}

    raise RuntimeError(
        "Missing Google API key. Set GEMINI_API_KEY (for example in a root .env file)."
    )


class GeminiBackend:
    backend_name = "gemini"

    def __init__(self) -> None:
        self.model_name = get_model_name()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: google-genai. Install with: pip install -U google-genai"
            ) from e

        client = genai.Client(**_resolve_client_kwargs())
        resolved_temperature = SETTINGS.LLM_TEMPERATURE if temperature is None else temperature
        contents = prompt if not system else f"{system}\n\n{prompt}"
        last_err = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=resolved_temperature,
                    ),
                )
                return resp.text or ""
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Gemini generate_content failed after retries: {last_err}") from last_err

    def ping(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())


def make_generate_fn() -> Callable[[str], str]:
    """
    Gemini Developer API generator: generate_fn(prompt) -> text

    Uses google-genai SDK with explicit client credential arguments resolved
    from environment variables for deterministic behavior across SDK versions.
    """
    backend = GeminiBackend()

    def _gen(prompt: str) -> str:
        return backend.generate(prompt, temperature=SETTINGS.LLM_TEMPERATURE)

    return _gen
