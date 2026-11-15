from __future__ import annotations

import math

from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.core.uncertainty import UncertaintyEstimator, cluster_semantically, semantic_entropy
from mca.llm.scripted import ScriptedLLM
from mca.telemetry.schemas import DecisionEnum, MetacognitiveTelemetry


def test_entropy_math():
    assert semantic_entropy([5]) == (0.0, 0.0)
    h, hn = semantic_entropy([1, 1, 1, 1])
    assert math.isclose(h, math.log(4)) and math.isclose(hn, 1.0)


def test_clustering(nli):
    samples = ["Wenus jest drugą planetą od Słońca", "Drugą planetą od Słońca jest Wenus", "Mars jest drugą planetą od Słońca"]
    assert cluster_semantically(samples, nli) == [[0, 1], [2]]


async def test_early_stop_skips_samples(nli):
    llm = ScriptedLLM(["Wenus jest drugą planetą od Słońca."])
    est = UncertaintyEstimator(MCAConfig(n_samples=5), nli)
    report = await est.estimate(llm, "Która planeta jest druga od Słońca?")
    assert report.early_stopped and report.n_samples == 2 and len(llm.calls) == 2
    assert report.semantic_entropy == 0.0


async def test_high_entropy_triggers_abstain(nli):
    answers = ["Merkury jest pierwszy", "Wenus jest najgorętsza", "Jowisz jest gazowy", "Saturn ma pierścienie", "Neptun jest daleko"]
    llm = ScriptedLLM(answers)
    cfg = MCAConfig(n_samples=5, entropy_tau=0.6)
    report = await UncertaintyEstimator(cfg, nli).estimate(llm, "Pytanie bez jednoznacznej odpowiedzi?")
    assert not report.early_stopped and report.n_samples == 5 and report.normalized_entropy > 0.9
    decision = DecisionController(cfg).decide(MetacognitiveTelemetry(query="q", uncertainty=report))
    assert decision.decision is DecisionEnum.ABSTAIN and "R8:semantic_entropy" in decision.triggered_rules
