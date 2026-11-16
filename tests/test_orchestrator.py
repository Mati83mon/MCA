"""Speculative fast/slow streaming: buffering, abort-on-violation, flush window, TTFT budget."""

from __future__ import annotations

import asyncio

from mca import MCAConfig, MCAOrchestrator
from mca.core.evidence_verifier import EvidenceStore
from mca.llm.scripted import ScriptedLLM
from mca.telemetry.schemas import DecisionEnum


def make(llm, store, nli, **cfg):
    config = MCAConfig(nli_backend="lexical", embedding_backend="hashing", n_samples=2, **cfg)
    return MCAOrchestrator(llm, config, store=store, nli=nli)


async def collect(orch, query):
    return [ev async for ev in orch.stream(query)]


async def test_violation_aborts_before_any_token(store, nli):
    llm = ScriptedLLM(["Mars jest drugą planetą, bo tak wynika z orbit."], token_delay_s=0.01, first_token_delay_s=0.05)
    orch = make(llm, store, nli)
    events = await collect(orch, "Dlaczego Mars jest drugą planetą od Słońca?")
    kinds = [e.kind for e in events]
    assert "token" not in kinds and kinds[0] == "decision"
    assert events[0].payload["decision"]["decision"] == DecisionEnum.CHALLENGE.value
    orch.close()


async def test_pass_streams_within_ttft_budget(store, nli):
    llm = ScriptedLLM(["Wenus jest drugą planetą od Słońca."], token_delay_s=0.005, first_token_delay_s=0.02)
    orch = make(llm, store, nli)
    events = await collect(orch, "Która planeta jest druga od Słońca?")
    tokens = [e for e in events if e.kind == "token"]
    assert "".join(t.text for t in tokens) == "Wenus jest drugą planetą od Słońca."
    assert tokens[0].t_ms < orch.config.ttft_budget_ms
    final = next(e for e in events if e.kind == "decision")
    assert final.payload["decision"]["decision"] == DecisionEnum.PASS.value
    orch.close()


async def test_window_flush_then_late_violation_correction(store, nli, monkeypatch):
    llm = ScriptedLLM(["To jest bardzo długa odpowiedź " * 20], token_delay_s=0.01)
    orch = make(llm, store, nli, flush_window_ms=50)
    real_fast_path = orch.fast_path

    async def slow_fast_path(query):
        await asyncio.sleep(0.3)
        return await real_fast_path(query)

    monkeypatch.setattr(orch, "fast_path", slow_fast_path)
    events = await collect(orch, "Dlaczego kilogram jest podstawową jednostką długości w SI?")
    kinds = [e.kind for e in events]
    first_token = next(e for e in events if e.kind == "token")
    assert first_token.t_ms < 300, "buffer must flush after the 50 ms window, before the fast path ends"
    assert "correction" in kinds and kinds.index("correction") > kinds.index("token")
    correction = next(e for e in events if e.kind == "correction")
    assert correction.payload["decision"]["decision"] == DecisionEnum.CHALLENGE.value
    orch.close()


async def test_run_returns_typed_result(orchestrator):
    result = await orchestrator.run("Ile lat ma Anna?")
    assert result.decision.decision is DecisionEnum.ABSTAIN
    assert result.aborted_generation and result.base_response == ""
    assert "Anna" in result.final_response
    assert result.telemetry.latency_ms["fast_path_total"] > 0


async def test_audit_mode_records_entropy_even_when_fast_path_decides(store, nli):
    query = "Dlaczego Mars jest drugą planetą od Słońca?"
    for audit, expect_entropy in ((False, False), (True, True)):
        orch = make(ScriptedLLM(["Bo tak.", "Nie wiem."]), store, nli, audit_all_layers=audit)
        result = await orch.run(query)
        assert result.decision.decision is DecisionEnum.CHALLENGE, "audit mode must not change the decision"
        assert (result.telemetry.uncertainty is not None) is expect_entropy
        orch.close()
