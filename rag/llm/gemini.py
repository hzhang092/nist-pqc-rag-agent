# rag/llm/gemini.py
from __future__ import annotations

import os
import time
from typing import Callable

from rag.config import SETTINGS


def make_generate_fn() -> Callable[[str], str]:
    """
    Gemini Developer API generator: generate_fn(prompt) -> text

    Uses google-genai SDK. The client reads GEMINI_API_KEY from the environment
    by default. (Docs assume GEMINI_API_KEY is set.)
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: google-genai. Install with: pip install -U google-genai"
        ) from e

    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    # Client gets API key from GEMINI_API_KEY automatically; alternatively pass api_key=...
    client = genai.Client()

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
