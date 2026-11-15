"""Deterministic scripted backend for tests and reproducible demos."""

from __future__ import annotations

import asyncio
import itertools
from typing import AsyncIterator, Callable, Sequence


class ScriptedLLM:
    """Returns pre-defined completions.

    ``responses`` may be a fixed sequence (cycled) or a callable ``prompt -> str``.
    ``token_delay_s`` simulates decoding latency when streaming;
    ``first_token_delay_s`` simulates prefill latency.
    """

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str], str],
        *,
        token_delay_s: float = 0.0,
        first_token_delay_s: float = 0.0,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._fn = responses if callable(responses) else None
        self._cycle = itertools.cycle(list(responses)) if not callable(responses) else None
        self.token_delay_s = token_delay_s
        self.first_token_delay_s = first_token_delay_s
        self.calls: list[str] = []

    def _next(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        assert self._cycle is not None
        return next(self._cycle)

    async def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256, seed: int | None = None) -> str:
        await asyncio.sleep(self.first_token_delay_s)
        return self._next(prompt)

    async def stream(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> AsyncIterator[str]:
        text = self._next(prompt)
        await asyncio.sleep(self.first_token_delay_s)
        for i, word in enumerate(text.split(" ")):
            if i:
                await asyncio.sleep(self.token_delay_s)
            yield word if i == 0 else " " + word
