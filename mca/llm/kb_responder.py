"""Offline, retrieval-grounded responder used when no neural LLM is attached.

It answers from symbolic oracles first and otherwise from the best-matching
knowledge-base passage. It is deterministic, so semantic entropy over its
samples is zero – the uncertainty layer then relies on the other layers.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from mca.core.evidence_verifier import EvidenceStore
from mca.core.oracles import DEFAULT_ORACLES


class KnowledgeBaseResponder:
    name = "kb-responder"

    def __init__(self, store: EvidenceStore, *, min_score: float = 0.35, token_delay_s: float = 0.0) -> None:
        self.store = store
        self.min_score = min_score
        self.token_delay_s = token_delay_s

    def answer(self, prompt: str) -> str:
        for oracle in DEFAULT_ORACLES:
            if oracle.applies(prompt):
                verdict = oracle.evaluate(prompt)
                if verdict is not None:
                    return verdict.explanation
        hits = self.store.search(prompt, k=1)
        if hits and hits[0].score >= self.min_score:
            return hits[0].text
        return "Nie mam w bazie wiedzy informacji pozwalających odpowiedzieć na to pytanie."

    async def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256, seed: int | None = None) -> str:
        return self.answer(prompt)

    async def generate_many(self, prompt: str, n: int, *, temperature: float = 0.8, max_new_tokens: int = 96) -> list[str]:
        return [self.answer(prompt)] * n

    async def stream(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> AsyncIterator[str]:
        for i, word in enumerate(self.answer(prompt).split(" ")):
            if i and self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield word if i == 0 else " " + word
