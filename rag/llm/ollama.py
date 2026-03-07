from __future__ import annotations

from typing import Any

import httpx

from rag.config import SETTINGS


class OllamaBackend:
    backend_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self._base_url = (base_url or SETTINGS.LLM_BASE_URL).rstrip("/")
        self.model_name = (model_name or SETTINGS.LLM_MODEL).strip()
        self._timeout_s = int(timeout_s or SETTINGS.LLM_TIMEOUT_S)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        model_name = self.model_name.strip()
        if not model_name:
            raise RuntimeError("LLM_MODEL must be set when LLM_BACKEND=ollama.")

        options = dict(kwargs.pop("options", {}) or {})
        resolved_temperature = SETTINGS.LLM_TEMPERATURE if temperature is None else temperature
        options["temperature"] = resolved_temperature

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        payload.update(kwargs)

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                response = client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        text = str(data.get("response", "") or "")
        if not text and data.get("done") is not True:
            raise RuntimeError("Ollama returned an empty response payload.")
        return text

    def ping(self) -> bool:
        try:
            with httpx.Client(timeout=min(self._timeout_s, 5)) as client:
                response = client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False
