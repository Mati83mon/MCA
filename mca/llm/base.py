"""LLM backend protocol used by the orchestrator, the uncertainty estimator and the decomposer."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal async text-generation interface."""

    name: str

    async def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256, seed: int | None = None) -> str:
        """Return a full completion."""
        ...

    def stream(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> AsyncIterator[str]:
        """Yield the completion incrementally (token or word pieces)."""
        ...


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first parseable JSON object from an LLM completion."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    match = _JSON_BLOCK.search(text)
    if match:
        candidates.append(match.group(0))
    for cand in candidates:
        # progressively trim trailing garbage after the last closing brace
        for end in range(len(cand), 0, -1):
            if cand[end - 1] != "}":
                continue
            try:
                value = json.loads(cand[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None
