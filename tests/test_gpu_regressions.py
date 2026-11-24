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


def test_is_specific_referent():
    from mca.core.premise_validator import is_specific_referent

    q = "Jan Kowalski kupił samochód. Ile lat ma mój brat Piotr?"
    assert is_specific_referent("mój brat", q)
    assert is_specific_referent("Piotr", q)
    assert is_specific_referent("ten samochód", q)
    assert is_specific_referent("wczorajszy mecz", q)
    assert not is_specific_referent("samochód", q)
    assert not is_specific_referent("kolor piłki", "W pudełku jest piłka. Jakiego jest koloru?")


def test_r7_llm_referent_modes():
    from mca.core.decision_controller import DecisionController
    from mca.telemetry.schemas import MetacognitiveTelemetry, PremiseVerdict

    q = "Bąk obraca się 10 razy na sekundę. Ile obrotów wykona w ciągu 3 sekund?"
    hint = Premise(id="lr1", text="bąk", kind=PremiseKind.REFERENT, source="llm", missing_variable="bąk")
    t = MetacognitiveTelemetry(query=q, premise_verdicts=[PremiseVerdict(premise=hint, status=PremiseStatus.UNVERIFIED)])

    def first(mode):
        return DecisionController(MCAConfig(nli_backend="lexical", embedding_backend="hashing",
                                            r7_llm_referents=mode)).decide(t).triggered_rules[0]

    assert first("all") == "R7:unresolved_referent"
    assert first("specific") != "R7:unresolved_referent"
    assert first("off") != "R7:unresolved_referent"


def test_parse_answerability():
    from mca.core.premise_validator import parse_answerability

    assert parse_answerability("TAK") is True
    assert parse_answerability("Nie.") is False
    assert parse_answerability("NIE, brakuje ceny") is False
    assert parse_answerability("Tak, bo ...") is True
    assert parse_answerability("Trudno powiedzieć") is None


def test_r7_gated_mode_follows_probe():
    from mca.core.decision_controller import DecisionController
    from mca.telemetry.schemas import AnswerabilityReport, MetacognitiveTelemetry, PremiseVerdict

    hint = Premise(id="lr1", text="kolor piłki", kind=PremiseKind.REFERENT, source="llm", missing_variable="kolor")
    base = dict(query="W pudełku jest piłka. Jakiego jest koloru?",
                premise_verdicts=[PremiseVerdict(premise=hint, status=PremiseStatus.UNVERIFIED)])
    ctl = DecisionController(MCAConfig(nli_backend="lexical", embedding_backend="hashing", r7_llm_referents="gated"))

    def first(**kw):
        return ctl.decide(MetacognitiveTelemetry(**base, **kw)).triggered_rules[0]

    assert first(answerability=AnswerabilityReport(answerable=False)) == "R7:unresolved_referent"
    assert first(answerability=AnswerabilityReport(answerable=True)) != "R7:unresolved_referent"
    assert first() == "R7:unresolved_referent"  # no probe -> same as "all"


def test_answerability_probe_in_fast_path():
    import asyncio

    from mca.core.orchestrator import MCAOrchestrator
    from mca.llm.scripted import ScriptedLLM

    def reply(prompt: str) -> str:
        return "NIE" if "TAK albo NIE" in prompt else "{}"

    cfg = MCAConfig(nli_backend="lexical", embedding_backend="hashing", answerability_check=True)
    orch = MCAOrchestrator(ScriptedLLM(reply), cfg, enable_uncertainty=False)
    t = asyncio.run(orch.fast_path("Kupiono trzy książki. Ile zapłacono?"))
    orch.close()
    assert t.answerability is not None and t.answerability.answerable is False


def test_probe_variant_parsers():
    from mca.core.premise_validator import PROBE_VARIANTS

    assert PROBE_VARIANTS["missing"][1]("KOMPLETNE") is True
    assert PROBE_VARIANTS["missing"][1]("cena biletu") is False
    assert PROBE_VARIANTS["missing"][1]("  ") is None
    assert PROBE_VARIANTS["inverse"][1]("TAK") is False   # "yes, data is missing"
    assert PROBE_VARIANTS["inverse"][1]("NIE") is True


def test_fewshot_examples_are_not_test_items():
    import json
    from pathlib import Path

    from mca.core.premise_validator import ANSWERABILITY_FEWSHOT

    shots = {line.split(":", 1)[1].strip() for line in ANSWERABILITY_FEWSHOT.splitlines() if line.startswith("Pytanie: ") and "{query}" not in line}
    ds = Path(__file__).resolve().parents[1] / "benchmarks" / "datasets"
    for f in ds.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data["pytania"] if isinstance(data, dict) else data
        assert not shots & {i["pytanie"] for i in items}, f.name


def test_select_r7_far_budget_prefers_safe_mode():
    from benchmarks import rule_table as rt

    curve_fars = {"all": (0.10, 0.70), "gated": (0.13, 0.40), "specific": (0.25, 0.20), "off": (0.28, 0.18)}

    def fake_eval(rows, cfg):
        far, ore = curve_fars[cfg.r7_llm_referents]
        return {"accuracy": 0.5, "FAR": far, "over_refusal": ore}

    orig = rt._evaluate
    rt._evaluate = fake_eval
    try:
        assert rt.select_r7_mode([], far_budget=0.05)["chosen"]["mode"] == "gated"
        assert rt.select_r7_mode([])["chosen"]["mode"] == "specific"
    finally:
        rt._evaluate = orig
