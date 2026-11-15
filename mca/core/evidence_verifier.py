"""Layer 4 – Evidence & Grounding Verifier.

Holds the local :class:`EvidenceStore` (dense retrieval over a JSONL knowledge
base, FAISS ``IndexFlatIP`` when installed) and turns retrieval + premise
verdicts + logic results into an :class:`EvidenceReport`. It separates

* **epistemic void** – the answer depends on information that is simply not
  available (unresolved referents, under-determined quantities), from
* **logical inconsistency** – the question's own assumptions contradict each
  other or established facts.
"""

from __future__ import annotations

import json
import threading
from importlib import resources
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from mca.config import MCAConfig
from mca.telemetry.schemas import (
    EvidenceReport,
    LogicReport,
    LogicStatus,
    Premise,
    PremiseKind,
    PremiseStatus,
    PremiseVerdict,
    RetrievedEvidence,
)
from mca.utils.embeddings import Embedder, HashingEmbedder, VectorIndex
from mca.utils.text import STOPWORDS, strip_accents, tokenize


class KnowledgeDocument(BaseModel):
    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    name: str | None = None
    resolution: str | None = None

    @property
    def is_paradox(self) -> bool:
        return "paradox" in self.tags


def load_seed_documents() -> list[KnowledgeDocument]:
    """Load the knowledge base bundled with the package."""
    text = resources.files("mca.knowledge").joinpath("seed_kb.jsonl").read_text(encoding="utf-8")
    return [KnowledgeDocument.model_validate_json(line) for line in text.splitlines() if line.strip()]


class EvidenceStore:
    """Dense vector store over :class:`KnowledgeDocument` objects."""

    def __init__(self, embedder: Embedder | None = None, use_faiss: bool = True) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.index = VectorIndex(self.embedder.dim, use_faiss=use_faiss)
        self.documents: list[KnowledgeDocument] = []
        self._vocab: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_seed(cls, embedder: Embedder | None = None, use_faiss: bool = True) -> "EvidenceStore":
        store = cls(embedder, use_faiss)
        store.add(load_seed_documents())
        return store

    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: Embedder | None = None, use_faiss: bool = True) -> "EvidenceStore":
        store = cls(embedder, use_faiss)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        store.add(KnowledgeDocument.model_validate(json.loads(line)) for line in lines if line.strip())
        return store

    def add(self, docs: Iterable[KnowledgeDocument]) -> None:
        docs = list(docs)
        if not docs:
            return
        vectors = self.embedder.encode([self._index_text(d) for d in docs])
        with self._lock:
            self.index.add(vectors)
            self.documents.extend(docs)
            for d in docs:
                self._vocab.update(self._entity_keys(d.text + " " + (d.name or "")))

    @staticmethod
    def _index_text(doc: KnowledgeDocument) -> str:
        return f"{doc.name}. {doc.text}" if doc.name else doc.text

    @staticmethod
    def _entity_keys(text: str) -> set[str]:
        return {strip_accents(t)[:5] for t in tokenize(text) if len(t) >= 3 and t not in STOPWORDS}

    def mentions(self, surface: str) -> bool:
        """Does any document mention an entity with this surface form (inflection-tolerant)?"""
        keys = self._entity_keys(surface)
        return bool(keys) and keys <= self._vocab

    def search(self, text: str, k: int = 4, tag: str | None = None) -> list[RetrievedEvidence]:
        with self._lock:
            if not self.documents:
                return []
            pool = len(self.documents) if tag else k
            scores, idx = self.index.search(self.embedder.encode([text]), pool)
            hits = []
            for score, i in zip(scores[0], idx[0]):
                doc = self.documents[int(i)]
                if tag and tag not in doc.tags:
                    continue
                hits.append(RetrievedEvidence(doc_id=doc.id, text=doc.text, score=float(score), tags=doc.tags))
                if len(hits) == k:
                    break
            return hits

    def get(self, doc_id: str) -> KnowledgeDocument | None:
        return next((d for d in self.documents if d.id == doc_id), None)


class EvidenceVerifier:
    """Aggregates grounding signals into an :class:`EvidenceReport`."""

    def __init__(self, store: EvidenceStore, config: MCAConfig) -> None:
        self.store = store
        self.config = config

    def paradox_match(self, query: str) -> RetrievedEvidence | None:
        hits = self.store.search(query, k=1, tag="paradox")
        if hits and hits[0].score >= self.config.paradox_similarity:
            return hits[0]
        return None

    def verify(self, query: str, premises: list[Premise], verdicts: list[PremiseVerdict], logic: LogicReport) -> EvidenceReport:
        retrieved = [h for h in self.store.search(query, k=self.config.retrieval_top_k) if h.score >= self.config.retrieval_min_similarity]
        checkable = [v for v in verdicts if v.premise.kind is not PremiseKind.REFERENT]
        supported = sum(v.status is PremiseStatus.SUPPORTED for v in checkable)
        support_ratio = supported / len(checkable) if checkable else 0.0

        missing: list[str] = []
        unresolved = [v for v in verdicts if v.premise.kind is PremiseKind.REFERENT and v.status is PremiseStatus.UNVERIFIED]
        missing.extend(v.premise.missing_variable or v.premise.text for v in unresolved)
        if logic.status is LogicStatus.UNDERDETERMINED:
            missing.extend(logic.free_variables)

        return EvidenceReport(
            support_ratio=round(support_ratio, 4),
            retrieved=retrieved,
            epistemic_void=bool(unresolved) or logic.status is LogicStatus.UNDERDETERMINED,
            logical_inconsistency=logic.status in {LogicStatus.UNSAT, LogicStatus.PARADOX_SELF_REF},
            missing_variables=list(dict.fromkeys(missing)),
            paradox_match=self.paradox_match(query),
        )

