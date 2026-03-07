from __future__ import annotations

from typing import Any

from ollama import Client

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
        self._client = Client(host=self._base_url, timeout=self._timeout_s)

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
            response = self._client.generate(**payload)
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        text = str(response.get("response", "") or "")
        if not text and response.get("done") is not True:
            raise RuntimeError("Ollama returned an empty response payload.")
        return text

    def ping(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False
