"""MCA – Metacognitive Control Architecture.

Neuro-symbolic middleware that gates LLM generation:
premise validation → semantic entropy → Z3 logic → evidence grounding → policy gate.
"""

from mca.config import MCAConfig
from mca.core.orchestrator import MCAOrchestrator, StreamEvent
from mca.telemetry.schemas import DecisionEnum, MCAResult, MetacognitiveTelemetry, QuestionItem


def build_offline_orchestrator(config: MCAConfig | None = None, **kwargs: object) -> MCAOrchestrator:
    """Orchestrator with zero-download backends: lexical NLI, hashing embeddings, KB responder."""
    from mca.core.evidence_verifier import EvidenceStore
    from mca.llm.kb_responder import KnowledgeBaseResponder
    from mca.utils.embeddings import HashingEmbedder
    from mca.utils.nli_client import LexicalNLIBackend, NLIClient

    cfg = config or MCAConfig(nli_backend="lexical", embedding_backend="hashing")
    store = EvidenceStore.from_seed(HashingEmbedder(), use_faiss=cfg.use_faiss)
    llm = KnowledgeBaseResponder(store)
    return MCAOrchestrator(llm, cfg, store=store, nli=NLIClient(LexicalNLIBackend()), **kwargs)  # type: ignore[arg-type]


__all__ = [
    "DecisionEnum",
    "MCAConfig",
    "MCAOrchestrator",
    "MCAResult",
    "MetacognitiveTelemetry",
    "QuestionItem",
    "StreamEvent",
    "build_offline_orchestrator",
]
__version__ = "0.1.0"
