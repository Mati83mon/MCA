"""R8b: high semantic entropy must not override a strong, NLI-entailed KB passage.

Regression from the ZeroGPU Space (Qwen2.5-1.5B): for "Która planeta jest druga od Słońca?" the raw
model hallucinated, 5 samples fell into 5 clusters (normalised H = 1.0) and MCA abstained under R8,
although the KB passage "Wenus jest drugą planetą od Słońca." (similarity 0.88, entailment 0.99)
supported the question's presupposition.
"""

from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.telemetry.schemas import (
    DecisionEnum,
    MetacognitiveTelemetry,
    NLILabel,
    Premise,
    PremiseKind,
    PremiseStatus,
    PremiseVerdict,
    RetrievedEvidence,
    UncertaintyReport,
)

QUERY = "Która planeta jest druga od Słońca?"
VENUS = "Wenus jest drugą planetą od Słońca."


def _telemetry(score: float = 0.88, entail: float = 0.99, kind: PremiseKind = PremiseKind.PRESUPPOSITION,
               status: PremiseStatus = PremiseStatus.SUPPORTED, entropy: float = 1.0) -> MetacognitiveTelemetry:
    premise = Premise(id="p1", text="Planeta jest druga od Słońca", kind=kind, existential=True)
    verdict = PremiseVerdict(
        premise=premise, status=status, nli_label=NLILabel.ENTAILMENT,
        nli_scores={NLILabel.ENTAILMENT: entail, NLILabel.NEUTRAL: 1 - entail, NLILabel.CONTRADICTION: 0.0},
        evidence=[RetrievedEvidence(doc_id="kb-venus", text=VENUS, score=score)],
    )
    return MetacognitiveTelemetry(
        query=QUERY, premises=[premise], premise_verdicts=[verdict],
        uncertainty=UncertaintyReport(n_samples=5, normalized_entropy=entropy),
    )


def _decide(t: MetacognitiveTelemetry, **cfg: object):
    return DecisionController(MCAConfig(nli_backend="lexical", embedding_backend="hashing", **cfg)).decide(t)


def test_space_regression_answers_from_kb_instead_of_abstaining():
    d = _decide(_telemetry())
    assert d.decision is DecisionEnum.PASS
    assert d.triggered_rules == ["R8b:evidence_answer"]
    assert VENUS in d.response_text


def test_weak_retrieval_still_abstains():
    assert _decide(_telemetry(score=0.45)).triggered_rules == ["R8:semantic_entropy"]


def test_weak_entailment_still_abstains():
    assert _decide(_telemetry(entail=0.6)).triggered_rules == ["R8:semantic_entropy"]


def test_referent_evidence_is_not_an_answer():
    assert _decide(_telemetry(kind=PremiseKind.REFERENT)).decision is not DecisionEnum.PASS


def test_rule_can_be_disabled():
    assert _decide(_telemetry(), evidence_answer=False).triggered_rules == ["R8:semantic_entropy"]


def test_low_entropy_pass_keeps_model_answer():
    d = _decide(_telemetry(entropy=0.1))
    assert d.triggered_rules == ["R9:pass"] and d.response_text == ""
