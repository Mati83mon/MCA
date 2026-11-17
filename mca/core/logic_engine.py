"""Layer 3 – Logical Consistency & SMT Engine (Microsoft Z3).

For every :class:`~mca.utils.z3_helpers.FormalSpec` the engine runs the
appropriate check and aggregates the results into one :class:`LogicReport`:

* ``self_reference`` specs  -> classical Z3 check + strong-Kleene / Łukasiewicz
  fixed points; no classical but a gap fixed point => ``PARADOX_SELF_REF``;
* ``entailment_goal`` specs -> validity of the claimed inference (counter-model on failure);
* ``target`` specs         -> determinacy (is the asked quantity uniquely fixed?);
* plain specs              -> satisfiability with a minimised ``unsat_core()``.

Status priority: PARADOX_SELF_REF > UNSAT > UNDERDETERMINED > UNKNOWN > SAT.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import z3

from mca.config import MCAConfig
from mca.telemetry.schemas import LogicReport, LogicStatus
from mca.utils.z3_helpers import (
    DSLError,
    FormalSpec,
    check_consistency,
    check_determinacy,
    check_entailment,
    kleene_fixed_points,
)

logger = logging.getLogger(__name__)

_PRIORITY = {
    LogicStatus.PARADOX_SELF_REF: 5,
    LogicStatus.UNSAT: 4,
    LogicStatus.UNDERDETERMINED: 3,
    LogicStatus.UNKNOWN: 2,
    LogicStatus.SAT: 1,
    LogicStatus.NOT_APPLICABLE: 0,
}


class LogicEngine:
    """Runs Z3 checks on formal specs. CPU work goes to a dedicated worker thread."""

    def __init__(self, config: MCAConfig, max_workers: int = 2) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mca-z3")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- single spec ---------------------------------------------------------
    def check_spec(self, spec: FormalSpec) -> LogicReport:
        timeout = self.config.z3_timeout_ms
        formulas = [f"[{a.id}] {a.expr}" for a in spec.assertions]
        try:
            if spec.self_reference:
                classical = check_consistency(spec, timeout)
                k3 = kleene_fixed_points(spec.self_reference, "kleene")
                l3 = kleene_fixed_points(spec.self_reference, "lukasiewicz")
                if k3.is_paradox or (classical.status == "unsat" and not k3.classical):
                    gap = k3.three_valued[0] if k3.three_valued else {}
                    return LogicReport(
                        status=LogicStatus.PARADOX_SELF_REF,
                        formulas=formulas,
                        unsat_core=classical.core,
                        kleene_assignment=gap,
                        explanation=(
                            "Samoodnosząca się równoważność typu P ⇔ ¬P: brak klasycznego przypisania wartości (Z3: UNSAT); "
                            f"jedyny punkt stały w logice Kleenego K3: {gap}; w Ł3: {l3.three_valued[0] if l3.three_valued else {}}."
                        ),
                    )
                return LogicReport(status=LogicStatus.SAT, formulas=formulas, model=classical.model,
                                   explanation="Definicje samoodnoszące się mają klasyczny punkt stały.")
            if spec.entailment_goal:
                entailed, counter = check_entailment(spec, timeout)
                if entailed:
                    return LogicReport(status=LogicStatus.SAT, formulas=formulas + [f"⊨ {spec.entailment_goal}"],
                                       explanation="Wniosek wynika logicznie z przesłanek.")
                names = spec.target_description
                return LogicReport(
                    status=LogicStatus.UNSAT,
                    formulas=formulas + [f"claim: premises ⊨ {spec.entailment_goal}"],
                    unsat_core=["entailment_claim"],
                    model=counter,
                    explanation=(
                        "Przesłanki nie implikują wniosku: Z3 znalazł kontrmodel spełniający przesłanki i negację wniosku"
                        + (f" (np. coś, co jest „{names.get('B')}”, a nie jest „{names.get('A')}”)." if names else ".")
                    ),
                )
            if spec.target:
                det = check_determinacy(spec, timeout)
                if det.status == "underdetermined":
                    return LogicReport(
                        status=LogicStatus.UNDERDETERMINED,
                        formulas=formulas + [f"target: {spec.target}"],
                        free_variables=list(spec.target_description.values()),
                        model={"value_1": det.values[0], "value_2": det.values[1]},
                        explanation=f"Dane nie wyznaczają jednoznacznie wartości {spec.target}: Z3 znalazł co najmniej dwa modele ({det.values[0]} i {det.values[1]}).",
                    )
                if det.status == "inconsistent":
                    out = check_consistency(spec, timeout)
                    return LogicReport(status=LogicStatus.UNSAT, formulas=formulas, unsat_core=out.core,
                                       explanation="Warunki zadania są wzajemnie sprzeczne.")
                if det.status == "determined":
                    return LogicReport(status=LogicStatus.SAT, formulas=formulas, model={"target": det.values[0]},
                                       explanation=f"Wartość {spec.target} jest wyznaczona jednoznacznie: {det.values[0]}.")
                return LogicReport(status=LogicStatus.UNKNOWN, formulas=formulas, explanation="Z3 nie rozstrzygnął w limicie czasu.")
            out = check_consistency(spec, timeout)
            if out.status == "unsat":
                texts = {a.id: a.text or a.expr for a in spec.assertions}
                return LogicReport(
                    status=LogicStatus.UNSAT,
                    formulas=formulas,
                    unsat_core=out.core,
                    explanation="Założenia są sprzeczne; minimalny rdzeń sprzeczności: " + "; ".join(texts[c] for c in out.core),
                )
            if out.status == "sat":
                return LogicReport(status=LogicStatus.SAT, formulas=formulas, model=out.model, explanation="Założenia są niesprzeczne.")
            return LogicReport(status=LogicStatus.UNKNOWN, formulas=formulas, explanation="Z3 nie rozstrzygnął w limicie czasu.")
        except (DSLError, z3.Z3Exception) as exc:
            logger.warning("invalid formal spec %s: %s", spec.label, exc)
            return LogicReport(status=LogicStatus.UNKNOWN, formulas=formulas, explanation=f"Niepoprawna formalizacja: {exc}")

    # -- aggregation -----------------------------------------------------------
    def check(self, specs: list[FormalSpec]) -> LogicReport:
        if not specs:
            return LogicReport(status=LogicStatus.NOT_APPLICABLE, explanation="Brak twierdzeń poddających się formalizacji.")
        reports = [(spec, self.check_spec(spec)) for spec in specs]
        spec, worst = max(reports, key=lambda r: _PRIORITY[r[1].status])
        merged = worst.model_copy(deep=True)
        merged.formulas = [f"{s.label}: {f}" for s, r in reports for f in r.formulas]
        merged.free_variables = sorted({v for _, r in reports for v in r.free_variables}) if worst.status is LogicStatus.UNDERDETERMINED else worst.free_variables
        merged.explanation = f"[{spec.label}] {worst.explanation}"
        return merged

    async def acheck(self, specs: list[FormalSpec]) -> LogicReport:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.check, specs)
