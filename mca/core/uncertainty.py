"""Layer 2 – Uncertainty Estimator (semantic entropy, Kuhn et al., ICLR 2023).

1. Draw up to ``N`` sampled answers from the LLM.
2. Cluster them into semantic-equivalence classes with *bidirectional* NLI entailment.
3. Compute Shannon entropy over the cluster distribution.

Speculative early stopping: ``early_stop_samples`` (default 2) samples are drawn
first; if they are already semantically equivalent the remaining ``N-2``
samples are skipped and the report is flagged ``early_stopped``.

Cluster probabilities use the discrete (count-based) estimator, which is the
variant recommended when sequence log-likelihoods are unavailable (e.g. for
API models); it equals the likelihood-weighted estimator under uniform weights.
"""

from __future__ import annotations

import asyncio
import math
from typing import Sequence

from mca.config import MCAConfig
from mca.llm.base import LLMBackend
from mca.telemetry.schemas import SemanticCluster, UncertaintyReport
from mca.utils.nli_client import NLIClient

ANSWER_PROMPT = "Odpowiedz krótko (jedno lub dwa zdania) na pytanie: {query}"


def cluster_semantically(samples: Sequence[str], nli: NLIClient, threshold: float = 0.5) -> list[list[int]]:
    """Greedy equivalence clustering: a sample joins the first cluster whose representative it bi-entails."""
    clusters: list[list[int]] = []
    for i, text in enumerate(samples):
        for cluster in clusters:
            if nli.bidirectional_entails(samples[cluster[0]], text, threshold):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def semantic_entropy(cluster_sizes: Sequence[int]) -> tuple[float, float]:
    """Return (entropy in nats, entropy normalised to [0, 1] by log(n_samples))."""
    n = sum(cluster_sizes)
    if n <= 1:
        return 0.0, 0.0
    h = -sum((c / n) * math.log(c / n) for c in cluster_sizes if c)
    return h, h / math.log(n)


class UncertaintyEstimator:
    def __init__(self, config: MCAConfig, nli: NLIClient) -> None:
        self.config = config
        self.nli = nli

    async def _sample(self, llm: LLMBackend, prompt: str, n: int) -> list[str]:
        cfg = self.config
        many = getattr(llm, "generate_many", None)
        if many is not None:
            return list(await many(prompt, n, temperature=cfg.sample_temperature, max_new_tokens=cfg.sample_max_new_tokens))
        return list(await asyncio.gather(*[
            llm.generate(prompt, temperature=cfg.sample_temperature, max_new_tokens=cfg.sample_max_new_tokens, seed=1000 + i)
            for i in range(n)
        ]))

    def report(self, samples: list[str], early_stopped: bool) -> UncertaintyReport:
        groups = cluster_semantically(samples, self.nli, self.config.nli_threshold)
        sizes = [len(g) for g in groups]
        h, h_norm = semantic_entropy(sizes)
        n = len(samples)
        clusters = [
            SemanticCluster(cluster_id=k, members=g, representative=samples[g[0]], probability=len(g) / n)
            for k, g in enumerate(sorted(groups, key=len, reverse=True))
        ]
        return UncertaintyReport(
            n_samples=n,
            samples=samples,
            clusters=clusters,
            semantic_entropy=round(h, 6),
            normalized_entropy=round(h_norm, 6),
            disagreement=round(1.0 - max(sizes) / n, 6) if n else 0.0,
            early_stopped=early_stopped,
        )

    async def estimate(self, llm: LLMBackend, query: str) -> UncertaintyReport:
        cfg = self.config
        prompt = ANSWER_PROMPT.format(query=query)
        first = await self._sample(llm, prompt, min(cfg.early_stop_samples, cfg.n_samples))
        if cfg.n_samples <= len(first):
            return await asyncio.to_thread(self.report, first, False)
        agree = await asyncio.to_thread(lambda: len(cluster_semantically(first, self.nli, cfg.nli_threshold)) == 1)
        if agree:
            return await asyncio.to_thread(self.report, first, True)
        rest = await self._sample(llm, prompt, cfg.n_samples - len(first))
        return await asyncio.to_thread(self.report, first + rest, False)
