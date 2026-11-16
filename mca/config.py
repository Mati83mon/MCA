"""Central configuration for the Metacognitive Control Architecture.

All thresholds that influence routing live here so the Decision Controller stays
deterministic and auditable: the same telemetry plus the same config always
yields the same decision.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MCAConfig(BaseModel):
    """Runtime configuration of the MCA pipeline.

    Every field can be overridden from the environment with the ``MCA_`` prefix,
    e.g. ``MCA_ENTROPY_TAU=0.5`` (see :meth:`from_env`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- Backends -----------------------------------------------------------
    nli_backend: Literal["auto", "transformers", "lexical"] = "auto"
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    embedding_backend: Literal["auto", "sentence-transformers", "hashing"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_faiss: bool = True
    device: str = "auto"

    # --- Layer 1: premise validation -----------------------------------------
    nli_threshold: float = Field(0.5, ge=0.0, le=1.0)
    use_llm_decomposition: bool = True

    # --- Layer 2: semantic entropy -------------------------------------------
    n_samples: int = Field(5, ge=1, le=32)
    early_stop_samples: int = Field(2, ge=2)
    sample_temperature: float = Field(0.8, gt=0.0)
    sample_max_new_tokens: int = Field(96, ge=8)
    entropy_tau: float = Field(0.6, ge=0.0, le=1.0, description="Normalised semantic-entropy threshold for ABSTAIN.")

    # --- Layer 3: logic ------------------------------------------------------
    z3_timeout_ms: int = Field(2000, ge=50)

    # --- Layer 4: evidence ---------------------------------------------------
    retrieval_top_k: int = Field(4, ge=1)
    retrieval_min_similarity: float = Field(0.35, ge=-1.0, le=1.0)
    paradox_similarity: float = Field(0.30, ge=-1.0, le=1.0)
    evidence_support_threshold: float = Field(0.5, ge=0.0, le=1.0)
    evidence_answer: bool = Field(True, description="R8b: when entropy is high but a KB passage entails a premise, answer from that passage.")
    evidence_answer_min_similarity: float = Field(0.6, ge=-1.0, le=1.0)
    evidence_answer_min_entailment: float = Field(0.9, ge=0.0, le=1.0)

    # --- Orchestrator --------------------------------------------------------
    flush_window_ms: int = Field(250, ge=0)
    ttft_budget_ms: int = Field(400, ge=1)
    max_new_tokens: int = Field(256, ge=1)

    @classmethod
    def from_env(cls, **overrides: object) -> "MCAConfig":
        """Build a config from ``MCA_*`` environment variables plus explicit overrides."""
        values: dict[str, object] = {}
        for name, field in cls.model_fields.items():
            raw = os.environ.get(f"MCA_{name.upper()}")
            if raw is None:
                continue
            annotation = field.annotation
            if annotation is bool:
                values[name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif annotation is int:
                values[name] = int(raw)
            elif annotation is float:
                values[name] = float(raw)
            else:
                values[name] = raw
        values.update(overrides)
        return cls(**values)


DEFAULT_CONFIG = MCAConfig()
