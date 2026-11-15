"""Regressions found in the Kaggle 2×T4 run (Qwen2.5-7B / DeepSeek-R1-Distill-Qwen-7B)."""

from mca.config import MCAConfig
from mca.core.premise_validator import PremiseValidator
from mca.core.evidence_verifier import EvidenceStore
from mca.llm.transformers_llm import strip_reasoning
from mca.telemetry.schemas import Premise, PremiseKind, PremiseStatus
from mca.utils.embeddings import HashingEmbedder
from mca.utils.nli_client import NLIClient, LexicalNLIBackend
from mca.utils.z3_helpers import FormalSpec, check_consistency, normalize_expr


def test_infix_llm_formulas_are_normalised():
    assert normalize_expr("x > 0 And x < 0") == "x > 0 and x < 0"
    assert normalize_expr("y == 2*x Or y == x/2") == "y == 2*x or y == x/2"
    assert normalize_expr("And(x > 0, Or(y, z))") == "And(x > 0, Or(y, z))"
    assert normalize_expr("(p) -> (q)") == "(p) >> (q)"
    spec = FormalSpec.model_validate({"vars": {"x": "Int"}, "assertions": [{"id": "a", "expr": "x > 0 And x < 0"}]})
    assert check_consistency(spec).status == "unsat"


def test_strip_reasoning_keeps_final_answer():
    assert strip_reasoning("<think>hmm, Pluto...</think>\n\nOsiem.") == "Osiem."
    assert strip_reasoning("Osiem.") == "Osiem."


def _validator():
    cfg = MCAConfig(nli_backend="lexical", embedding_backend="hashing")
    store = EvidenceStore.from_seed(HashingEmbedder(), use_faiss=False)
    return PremiseValidator(cfg, store=store, nli=NLIClient(LexicalNLIBackend()))


def test_llm_referent_grounded_in_kb_is_resolved():
    v = _validator()
    p = Premise(id="lr1", text="standardowy wykaz planet Układu Słonecznego", kind=PremiseKind.REFERENT,
                source="llm", missing_variable="jakiego wykazu")
    assert v.validate([p], "Ile planet obejmuje standardowy wykaz planet Układu Słonecznego?")[0].status is PremiseStatus.SUPPORTED


def test_rule_referent_stays_unresolved():
    v = _validator()
    p = Premise(id="r1", text="moja siostra", kind=PremiseKind.REFERENT, missing_variable="kim jest siostra")
    assert v.validate([p], "Ile lat ma moja siostra?")[0].status is PremiseStatus.UNVERIFIED
