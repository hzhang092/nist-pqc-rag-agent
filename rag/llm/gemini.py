# rag/llm/gemini.py
from __future__ import annotations

import os
import time
from typing import Callable

from rag.config import SETTINGS

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def get_model_name() -> str:
    """Returns the effective Gemini model name from environment/config."""
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


def make_generate_fn() -> Callable[[str], str]:
    """
    Gemini Developer API generator: generate_fn(prompt) -> text

    Uses google-genai SDK with explicit client credential arguments resolved
    from environment variables for deterministic behavior across SDK versions.
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: google-genai. Install with: pip install -U google-genai"
        ) from e

    model = get_model_name()

    client = genai.Client(**_resolve_client_kwargs())

    def _gen(prompt: str) -> str:
        # Simple retry for free-tier rate limits / transient errors
        last_err = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=SETTINGS.LLM_TEMPERATURE,
                    ),
                )
                return resp.text or ""
            except Exception as e:
                last_err = e
                # backoff: 0.5s, 1s, 2s
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Gemini generate_content failed after retries: {last_err}") from last_err

    return _gen
