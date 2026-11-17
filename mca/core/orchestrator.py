"""Speculative Fast/Slow Orchestrator.

Streaming path (:meth:`MCAOrchestrator.stream`):

1. base-LLM generation starts immediately and fills an internal FIFO buffer;
2. the *fast path* (premise decomposition + validation, Z3 logic, evidence
   grounding) runs concurrently in worker threads;
3. if the fast path returns a violation inside ``flush_window_ms`` (250 ms by
   default) the generation task is cancelled and the policy response is emitted
   instead – the user never sees the unsafe draft;
4. if the window elapses without a violation, the buffer is flushed and
   streaming continues (TTFT ≈ window + first token). The fast path keeps
   running; a late violation aborts the stream and appends a correction;
5. the *slow path* (semantic entropy, N samples) runs in the background and is
   attached to the final telemetry; high entropy after streaming appends a
   calibrated uncertainty note.

:meth:`MCAOrchestrator.run` executes all layers to completion before answering
(used by benchmarks).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.core.evidence_verifier import EvidenceStore, EvidenceVerifier
from mca.core.logic_engine import LogicEngine
from mca.core.premise_validator import PremiseValidator
from mca.core.uncertainty import UncertaintyEstimator
from mca.llm.base import LLMBackend
from mca.telemetry.schemas import (
    DecisionEnum,
    MCAResult,
    MetacognitiveTelemetry,
    PolicyDecision,
)
from mca.utils.embeddings import load_embedder
from mca.utils.nli_client import NLIClient, load_nli_client

EventKind = Literal["token", "decision", "correction", "telemetry", "done"]


@dataclass
class StreamEvent:
    kind: EventKind
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    t_ms: float = 0.0


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


class MCAOrchestrator:
    """Wires the five layers together around a base :class:`LLMBackend`."""

    def __init__(
        self,
        llm: LLMBackend,
        config: MCAConfig | None = None,
        *,
        store: EvidenceStore | None = None,
        nli: NLIClient | None = None,
        sampler: LLMBackend | None = None,
        enable_uncertainty: bool = True,
    ) -> None:
        self.config = config or MCAConfig()
        self.llm = llm
        self.sampler = sampler or llm
        self.nli = nli or load_nli_client(self.config)
        self.store = store or EvidenceStore.from_seed(load_embedder(self.config), use_faiss=self.config.use_faiss)
        self.premises = PremiseValidator(self.config, self.store, self.nli, llm=llm)
        self.logic = LogicEngine(self.config)
        self.evidence = EvidenceVerifier(self.store, self.config)
        self.uncertainty = UncertaintyEstimator(self.config, self.nli)
        self.controller = DecisionController(self.config, self.store)
        self.enable_uncertainty = enable_uncertainty

    def close(self) -> None:
        self.logic.close()

    # -- layers ----------------------------------------------------------------
    async def fast_path(self, query: str) -> MetacognitiveTelemetry:
        """Layers 1, 3, 4 (no sampling). Typically a few ms with the offline backends."""
        lat: dict[str, float] = {}
        t0 = time.perf_counter()
        probe = asyncio.create_task(self.premises.answerability(query))
        decomposition = await self.premises.decompose(query)
        lat["decompose"] = _ms(t0)
        t1 = time.perf_counter()
        verdicts_task = asyncio.create_task(self.premises.avalidate(decomposition.premises, query))
        logic_task = asyncio.create_task(self.logic.acheck(decomposition.specs))
        verdicts, logic = await asyncio.gather(verdicts_task, logic_task)
        lat["premises+logic"] = _ms(t1)
        t2 = time.perf_counter()
        evidence = await asyncio.to_thread(self.evidence.verify, query, decomposition.premises, verdicts, logic)
        lat["evidence"] = _ms(t2)
        answerability = await probe
        lat["fast_path_total"] = _ms(t0)
        return MetacognitiveTelemetry(query=query, premises=decomposition.premises, premise_verdicts=verdicts,
                                      logic=logic, evidence=evidence, answerability=answerability, latency_ms=lat)

    async def _slow_path(self, query: str) -> Any:
        t0 = time.perf_counter()
        report = await self.uncertainty.estimate(self.sampler, query)
        return report, _ms(t0)

    # -- non-streaming -------------------------------------------------------------
    async def run(self, query: str, *, generate_base: bool = True) -> MCAResult:
        """Run every layer to completion, then decide. Generation of the base answer is concurrent."""
        t0 = time.perf_counter()
        base_task = asyncio.create_task(self.llm.generate(query, max_new_tokens=self.config.max_new_tokens)) if generate_base else None
        slow_task = asyncio.create_task(self._slow_path(query)) if self.enable_uncertainty else None
        telemetry = await self.fast_path(query)
        pre = self.controller.decide(telemetry)
        if slow_task is not None:
            if pre.decision is DecisionEnum.PASS or self.config.audit_all_layers:
                report, ms = await slow_task
                telemetry.uncertainty, telemetry.latency_ms["uncertainty"] = report, ms
            else:
                slow_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await slow_task
        decision = self.controller.decide(telemetry)
        base = ""
        if base_task is not None:
            if decision.decision is DecisionEnum.PASS:
                base = await base_task
            else:
                base_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await base_task
        telemetry.latency_ms["total"] = _ms(t0)
        final = decision.response_text if decision.response_text else base
        return MCAResult(query=query, decision=decision, telemetry=telemetry, base_response=base,
                         final_response=final, aborted_generation=decision.decision is not DecisionEnum.PASS)

    # -- streaming -------------------------------------------------------------------
    async def stream(self, query: str) -> AsyncIterator[StreamEvent]:
        """Speculative streaming with a ``flush_window_ms`` guard (see module docstring)."""
        cfg = self.config
        t0 = time.perf_counter()
        buffer: asyncio.Queue[str | None] = asyncio.Queue()

        async def _produce() -> None:
            try:
                async for piece in self.llm.stream(query, max_new_tokens=cfg.max_new_tokens):
                    await buffer.put(piece)
            finally:
                await buffer.put(None)

        gen_task = asyncio.create_task(_produce())
        fast_task = asyncio.create_task(self.fast_path(query))
        slow_task = asyncio.create_task(self._slow_path(query)) if self.enable_uncertainty else None

        async def _cancel(*tasks: asyncio.Task[Any] | None) -> None:
            for task in tasks:
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

        done, _ = await asyncio.wait({fast_task}, timeout=cfg.flush_window_ms / 1000.0)
        telemetry: MetacognitiveTelemetry | None = fast_task.result() if done else None
        if telemetry is not None:
            decision = self.controller.decide(telemetry)
            if decision.decision is not DecisionEnum.PASS:
                await _cancel(gen_task, slow_task)
                telemetry.latency_ms["total"] = _ms(t0)
                yield StreamEvent("decision", decision.response_text, {"decision": decision.model_dump(mode="json")}, _ms(t0))
                yield StreamEvent("telemetry", payload=telemetry.model_dump(mode="json"), t_ms=_ms(t0))
                yield StreamEvent("done", t_ms=_ms(t0))
                return

        # Window elapsed (or fast path PASSed): flush buffer, keep monitoring.
        ttft: float | None = None
        emitted: list[str] = []
        aborted = False
        while True:
            get_task = asyncio.create_task(buffer.get())
            waiters: set[asyncio.Task[Any]] = {get_task}
            if telemetry is None:
                waiters.add(fast_task)
            finished, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if fast_task in finished and telemetry is None:
                telemetry = fast_task.result()
                late = self.controller.decide(telemetry)
                if late.decision is not DecisionEnum.PASS:
                    await _cancel(get_task, gen_task, slow_task)
                    aborted = True
                    yield StreamEvent("correction", "\n\n[MCA: przerwano generowanie] " + late.response_text,
                                      {"decision": late.model_dump(mode="json")}, _ms(t0))
                    break
                if get_task not in finished:
                    piece = await get_task
                else:
                    piece = get_task.result()
            else:
                piece = get_task.result()
            if piece is None:
                break
            if ttft is None:
                ttft = _ms(t0)
            emitted.append(piece)
            yield StreamEvent("token", piece, t_ms=_ms(t0))

        if telemetry is None:
            telemetry = await fast_task
        decision: PolicyDecision = self.controller.decide(telemetry)
        if not aborted and slow_task is not None:
            report, ms = await slow_task
            telemetry.uncertainty, telemetry.latency_ms["uncertainty"] = report, ms
            decision = self.controller.decide(telemetry)
            if decision.decision is DecisionEnum.ABSTAIN:
                yield StreamEvent("correction", "\n\n[MCA: niska pewność] " + decision.response_text,
                                  {"decision": decision.model_dump(mode="json")}, _ms(t0))
            elif decision.decision is DecisionEnum.PASS and decision.response_text:
                yield StreamEvent("correction", "\n\n[MCA: odpowiedź z bazy wiedzy] " + decision.response_text,
                                  {"decision": decision.model_dump(mode="json")}, _ms(t0))
        if ttft is not None:
            telemetry.latency_ms["ttft"] = ttft
        telemetry.latency_ms["total"] = _ms(t0)
        yield StreamEvent("decision", "", {"decision": decision.model_dump(mode="json"), "streamed_text": "".join(emitted)}, _ms(t0))
        yield StreamEvent("telemetry", payload=telemetry.model_dump(mode="json"), t_ms=_ms(t0))
        yield StreamEvent("done", t_ms=_ms(t0))
