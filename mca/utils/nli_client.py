"""NLI runtime helpers: DeBERTa cross-encoder backend plus an offline lexical backend.

The :class:`NLIClient` is what every layer talks to. It caches predictions,
batches requests and exposes the bidirectional-entailment primitive used for
semantic clustering (Kuhn et al., 2023).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Protocol, Sequence

from mca.config import MCAConfig
from mca.telemetry.schemas import NLILabel
from mca.utils.text import EXCLUSIVE_GROUPS, content_profile, split_clauses

logger = logging.getLogger(__name__)

NLIScores = dict[NLILabel, float]


class NLIBackend(Protocol):
    """Anything that maps (premise, hypothesis) pairs to a label distribution."""

    name: str

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        """Return one probability distribution over :class:`NLILabel` per pair."""
        ...


class TransformersNLIBackend:
    """Cross-encoder NLI (default ``cross-encoder/nli-deberta-v3-large``).

    Label order is read from ``model.config.id2label`` so any MNLI/XNLI
    cross-encoder (e.g. a multilingual mDeBERTa for Polish input) works.
    """

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"transformers:{model_name}"
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.label_index = self._resolve_labels(self.model.config.id2label)

    @staticmethod
    def _resolve_labels(id2label: dict[int, str]) -> dict[NLILabel, int]:
        mapping: dict[NLILabel, int] = {}
        for idx, raw in id2label.items():
            low = raw.lower()
            for label in NLILabel:
                if low.startswith(label.value[:5]):
                    mapping[label] = int(idx)
        if len(mapping) != 3:
            raise ValueError(f"cannot map NLI labels from id2label={id2label}")
        return mapping

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        torch = self._torch
        out: list[NLIScores] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in chunk], [h for _, h in chunk],
                padding=True, truncation=True, max_length=384, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                out.append({label: float(row[i]) for label, i in self.label_index.items()})
        return out


class LexicalNLIBackend:
    """Deterministic clause-level lexical NLI used offline and in CI.

    Heuristics (applied per premise clause, strongest clause wins):

    * *coverage* – share of hypothesis content stems found in the clause;
    * *polarity conflict* – differing negation parity (incl. Polish ``nie-`` prefixes);
    * *numeric/ordinal conflict* – hypothesis numbers absent from a clause that has numbers;
    * *exclusive-slot conflict* – the two texts pick different members of an exclusive group.

    It is a calibrated baseline, not a replacement for a trained cross-encoder.
    """

    name = "lexical"

    def __init__(self, entail_coverage: float = 0.75, conflict_coverage: float = 0.5) -> None:
        self.entail_coverage = entail_coverage
        self.conflict_coverage = conflict_coverage

    @staticmethod
    def _exclusive_conflict(h: frozenset[str], p: frozenset[str]) -> bool:
        for group in EXCLUSIVE_GROUPS:
            h_members = {s for s in h if any(s.startswith(g) or g.startswith(s) for g in group)}
            p_members = {s for s in p if any(s.startswith(g) or g.startswith(s) for g in group)}
            if h_members and p_members and not (h_members & p_members):
                return True
        return False

    def _score_clause(self, clause: str, hypothesis: str) -> NLIScores:
        p_stems, p_neg, p_nums = content_profile(clause)
        h_stems, h_neg, h_nums = content_profile(hypothesis)
        if not h_stems:
            return {NLILabel.ENTAILMENT: 0.0, NLILabel.NEUTRAL: 1.0, NLILabel.CONTRADICTION: 0.0}
        shared = h_stems & p_stems
        coverage = len(shared) / len(h_stems)
        conflict = False
        if coverage >= self.conflict_coverage and len(shared) >= 2:
            if p_neg != h_neg:
                conflict = True
            elif p_nums and h_nums and not (h_nums <= p_nums):
                conflict = True
            elif self._exclusive_conflict(h_stems - shared, p_stems - shared):
                conflict = True
        if conflict:
            c = min(0.97, 0.45 + 0.5 * coverage)
            return {NLILabel.CONTRADICTION: c, NLILabel.NEUTRAL: 1.0 - c, NLILabel.ENTAILMENT: 0.0}
        numbers_ok = not h_nums or h_nums <= p_nums
        if coverage >= self.entail_coverage and numbers_ok:
            e = min(0.97, 0.4 + 0.55 * coverage)
            return {NLILabel.ENTAILMENT: e, NLILabel.NEUTRAL: 1.0 - e, NLILabel.CONTRADICTION: 0.0}
        return {NLILabel.ENTAILMENT: 0.25 * coverage, NLILabel.NEUTRAL: 1.0 - 0.25 * coverage, NLILabel.CONTRADICTION: 0.0}

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        results: list[NLIScores] = []
        for premise, hypothesis in pairs:
            clauses = split_clauses(premise) or [premise]
            best = max(
                (self._score_clause(c, hypothesis) for c in clauses),
                key=lambda s: max(s[NLILabel.CONTRADICTION], s[NLILabel.ENTAILMENT]),
            )
            results.append(best)
        return results


class NLIClient:
    """Thread-safe, cached facade over an :class:`NLIBackend`."""

    def __init__(self, backend: NLIBackend, cache_size: int = 8192) -> None:
        self.backend = backend
        self._cache: OrderedDict[tuple[str, str], NLIScores] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.backend.name

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        with self._lock:
            missing = [p for p in dict.fromkeys(pairs) if p not in self._cache]
        if missing:
            scores = self.backend.predict(missing)
            with self._lock:
                for pair, score in zip(missing, scores):
                    self._cache[pair] = score
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        with self._lock:
            return [self._cache[p] for p in pairs]

    def classify(self, premise: str, hypothesis: str) -> tuple[NLILabel, NLIScores]:
        scores = self.predict([(premise, hypothesis)])[0]
        return max(scores, key=scores.__getitem__), scores

    def entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        return self.predict([(premise, hypothesis)])[0][NLILabel.ENTAILMENT] >= threshold

    def bidirectional_entails(self, a: str, b: str, threshold: float = 0.5) -> bool:
        """Semantic equivalence test of Kuhn et al.: a ⊨ b and b ⊨ a."""
        if a.strip().lower() == b.strip().lower():
            return True
        fwd, bwd = self.predict([(a, b), (b, a)])
        return fwd[NLILabel.ENTAILMENT] >= threshold and bwd[NLILabel.ENTAILMENT] >= threshold


def load_nli_client(config: MCAConfig) -> NLIClient:
    """Instantiate the configured backend; ``auto`` falls back to lexical if models are unavailable."""
    if config.nli_backend in {"auto", "transformers"}:
        try:
            return NLIClient(TransformersNLIBackend(config.nli_model, device=config.device))
        except Exception as exc:  # noqa: BLE001 - any load failure (offline, no torch) triggers fallback
            if config.nli_backend == "transformers":
                raise
            logger.warning("NLI model %s unavailable (%s); using lexical NLI backend", config.nli_model, exc)
    return NLIClient(LexicalNLIBackend())
