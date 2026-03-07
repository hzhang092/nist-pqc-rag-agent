from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        ...

    def ping(self) -> bool:
        ...
