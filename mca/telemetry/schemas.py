"""Strictly typed telemetry contracts shared by every MCA layer (Pydantic v2)."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DecisionEnum(str, Enum):
    """Actions the Policy Gate can take."""

    PASS = "PASS"
    CHALLENGE = "CHALLENGE"
    ABSTAIN = "ABSTAIN"
    EXPLAIN_PARADOX = "EXPLAIN_PARADOX"


class Category(str, Enum):
    """Benchmark categories (Polish identifiers as used in the dataset)."""

    PARADOKS = "paradoks"
    FALSZYWE_ZALOZENIE = "falszywe_zalozenie"
    BRAK_DANYCH = "brak_danych"
    NORMALNY_FAKT = "normalny_fakt"

    @property
    def expected_decision(self) -> DecisionEnum:
        """Gold policy decision for an item of this category."""
        return CATEGORY_TO_DECISION[self]


CATEGORY_TO_DECISION: dict[Category, DecisionEnum] = {
    Category.PARADOKS: DecisionEnum.EXPLAIN_PARADOX,
    Category.FALSZYWE_ZALOZENIE: DecisionEnum.CHALLENGE,
    Category.BRAK_DANYCH: DecisionEnum.ABSTAIN,
    Category.NORMALNY_FAKT: DecisionEnum.PASS,
}


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class PremiseKind(str, Enum):
    PRESUPPOSITION = "presupposition"
    ASSERTION = "assertion"
    REFERENT = "referent"


class PremiseStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class LogicStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    PARADOX_SELF_REF = "PARADOX_SELF_REF"
    UNDERDETERMINED = "UNDERDETERMINED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class QuestionItem(BaseModel):
    """One benchmark question as stored in ``pytania_podchwytliwe_*.json``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^Q\d{3,}$")
    kategoria: Category
    temat: str = Field(min_length=1)
    pytanie: str = Field(min_length=3)
    odpowiedz_wzorcowa: str = Field(min_length=1)
    wyjasnienie: str = Field(min_length=1)
    brakujace_dane: list[str] = Field(default_factory=list)
    zrodla: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _missing_data_consistency(self) -> "QuestionItem":
        if self.kategoria is Category.BRAK_DANYCH and not self.brakujace_dane:
            raise ValueError(f"{self.id}: category 'brak_danych' requires non-empty 'brakujace_dane'")
        if self.kategoria is not Category.BRAK_DANYCH and self.brakujace_dane:
            raise ValueError(f"{self.id}: 'brakujace_dane' allowed only for category 'brak_danych'")
        for url in self.zrodla:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"{self.id}: source is not an http(s) URL: {url!r}")
        return self


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    tytul: str
    jezyk: str = "pl"
    wersja: str
    liczba_pytan: int = Field(ge=0)
    liczby_w_kategoriach: dict[Category, int]


class BenchmarkDataset(BaseModel):
    """Whole dataset file with cross-item consistency checks."""

    model_config = ConfigDict(extra="forbid")

    metadane: DatasetMetadata
    pytania: list[QuestionItem]

    @model_validator(mode="after")
    def _check_counts(self) -> "BenchmarkDataset":
        ids = [q.id for q in self.pytania]
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        if dupes:
            raise ValueError(f"duplicate question ids: {dupes}")
        if self.metadane.liczba_pytan != len(self.pytania):
            raise ValueError(f"metadata declares {self.metadane.liczba_pytan} questions, found {len(self.pytania)}")
        counts = Counter(q.kategoria for q in self.pytania)
        for cat, declared in self.metadane.liczby_w_kategoriach.items():
            if counts.get(cat, 0) != declared:
                raise ValueError(f"category {cat.value}: declared {declared}, found {counts.get(cat, 0)}")
        return self


# ---------------------------------------------------------------------------
# Layer reports
# ---------------------------------------------------------------------------
class Premise(BaseModel):
    """An atomic presupposition/assertion extracted from the user query."""

    id: str
    text: str
    kind: PremiseKind = PremiseKind.PRESUPPOSITION
    source: str = Field("rules", description="'rules', 'llm' or 'formalizer'.")
    existential: bool = Field(False, description="Open wh-presupposition ('there is an x such that ...').")
    missing_variable: str | None = Field(None, description="For REFERENT premises: what the user must supply.")


class RetrievedEvidence(BaseModel):
    doc_id: str
    text: str
    score: float
    tags: list[str] = Field(default_factory=list)


class PremiseVerdict(BaseModel):
    premise: Premise
    status: PremiseStatus
    nli_label: NLILabel = NLILabel.NEUTRAL
    nli_scores: dict[NLILabel, float] = Field(default_factory=dict)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    oracle: str | None = Field(None, description="Name of a symbolic oracle that decided the verdict, if any.")
    explanation: str = ""


class SemanticCluster(BaseModel):
    cluster_id: int
    members: list[int]
    representative: str
    probability: float


class UncertaintyReport(BaseModel):
    n_samples: int
    samples: list[str] = Field(default_factory=list)
    clusters: list[SemanticCluster] = Field(default_factory=list)
    semantic_entropy: float = 0.0
    normalized_entropy: float = 0.0
    disagreement: float = 0.0
    early_stopped: bool = False


class LogicReport(BaseModel):
    status: LogicStatus = LogicStatus.NOT_APPLICABLE
    formulas: list[str] = Field(default_factory=list)
    unsat_core: list[str] = Field(default_factory=list)
    model: dict[str, str] = Field(default_factory=dict)
    kleene_assignment: dict[str, str] = Field(default_factory=dict)
    free_variables: list[str] = Field(default_factory=list)
    explanation: str = ""


class EvidenceReport(BaseModel):
    support_ratio: float = 0.0
    retrieved: list[RetrievedEvidence] = Field(default_factory=list)
    epistemic_void: bool = False
    logical_inconsistency: bool = False
    missing_variables: list[str] = Field(default_factory=list)
    paradox_match: RetrievedEvidence | None = None


class MetacognitiveTelemetry(BaseModel):
    """Everything the Policy Gate is allowed to look at."""

    model_config = ConfigDict(extra="forbid")

    query: str
    premises: list[Premise] = Field(default_factory=list)
    premise_verdicts: list[PremiseVerdict] = Field(default_factory=list)
    uncertainty: UncertaintyReport | None = None
    logic: LogicReport = Field(default_factory=LogicReport)
    evidence: EvidenceReport = Field(default_factory=EvidenceReport)
    latency_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def contradicted(self) -> list[PremiseVerdict]:
        return [v for v in self.premise_verdicts if v.status is PremiseStatus.CONTRADICTED]


class PolicyDecision(BaseModel):
    decision: DecisionEnum
    confidence: float = Field(ge=0.0, le=1.0)
    triggered_rules: list[str] = Field(default_factory=list)
    rationale: str
    response_text: str = ""


class MCAResult(BaseModel):
    """Final output of a (non-streaming) MCA run."""

    query: str
    decision: PolicyDecision
    telemetry: MetacognitiveTelemetry
    base_response: str = ""
    final_response: str
    ttft_ms: float | None = None
    aborted_generation: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
