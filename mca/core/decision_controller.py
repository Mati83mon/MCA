"""Layer 5 – Decision Controller / Policy Gate.

A pure, deterministic function of :class:`MetacognitiveTelemetry` and
:class:`MCAConfig`. Rules are evaluated in priority order; the first that fires
wins and is recorded in ``triggered_rules`` for auditability.

=====  ==================================================  =================
Prio   Rule                                                Decision
=====  ==================================================  =================
R1     Z3 ``PARADOX_SELF_REF`` (P ⇔ ¬P, diagonal)          EXPLAIN_PARADOX
R2     Formal contradiction (UNSAT core) in presuppositions CHALLENGE
R3     Oracle refutes a presupposition                      CHALLENGE
R4     Query matches a catalogued antinomy/paradox          EXPLAIN_PARADOX
R5     NLI contradiction of a presupposition vs evidence    CHALLENGE
R6     Epistemic void: under-determined target              ABSTAIN
R7     Epistemic void: unresolved referent                  ABSTAIN
R8b    High entropy, but a KB passage entails a premise   PASS (answer = passage)
R8     Semantic entropy above ``entropy_tau``               ABSTAIN
R9     otherwise                                            PASS
=====  ==================================================  =================
"""

from __future__ import annotations

from mca.config import MCAConfig
from mca.core.evidence_verifier import EvidenceStore
from mca.telemetry.schemas import (
    DecisionEnum,
    LogicStatus,
    MetacognitiveTelemetry,
    NLILabel,
    PolicyDecision,
    PremiseKind,
    PremiseStatus,
    PremiseVerdict,
)


def _challengeable(v: PremiseVerdict) -> bool:
    return v.status is PremiseStatus.CONTRADICTED and v.premise.kind is PremiseKind.PRESUPPOSITION


class DecisionController:
    def __init__(self, config: MCAConfig, store: EvidenceStore | None = None) -> None:
        self.config = config
        self.store = store

    def decide(self, t: MetacognitiveTelemetry) -> PolicyDecision:
        logic, ev = t.logic, t.evidence
        oracle_refuted = [v for v in t.premise_verdicts if _challengeable(v) and v.oracle]
        nli_refuted = [v for v in t.premise_verdicts if _challengeable(v) and not v.oracle]
        unresolved = [v for v in t.premise_verdicts if v.premise.kind is PremiseKind.REFERENT and v.status is PremiseStatus.UNVERIFIED]
        if any(v.status is PremiseStatus.SUPPORTED and v.premise.kind is not PremiseKind.REFERENT for v in t.premise_verdicts):
            # A KB-grounded question: LLM referent hints alone must not trigger an epistemic-void abstention.
            unresolved = [v for v in unresolved if v.premise.source != "llm"]

        if logic.status is LogicStatus.PARADOX_SELF_REF:
            return self._paradox(t, ["R1:z3_paradox_self_ref"], 0.97)
        if logic.status is LogicStatus.UNSAT:
            return PolicyDecision(
                decision=DecisionEnum.CHALLENGE, confidence=0.95, triggered_rules=["R2:z3_unsat_core"],
                rationale=logic.explanation, response_text=self._challenge_text(t, logic.explanation),
            )
        if oracle_refuted:
            v = oracle_refuted[0]
            return PolicyDecision(
                decision=DecisionEnum.CHALLENGE, confidence=0.95, triggered_rules=[f"R3:oracle:{v.oracle}"],
                rationale=v.explanation, response_text=self._challenge_text(t, v.explanation, v.premise.text),
            )
        if ev.paradox_match is not None:
            return self._paradox(t, [f"R4:paradox_catalog:{ev.paradox_match.doc_id}"], min(0.95, 0.5 + ev.paradox_match.score))
        if nli_refuted:
            v = max(nli_refuted, key=lambda x: x.nli_scores.get(x.nli_label, 0.0))
            conf = float(v.nli_scores.get(v.nli_label, self.config.nli_threshold))
            return PolicyDecision(
                decision=DecisionEnum.CHALLENGE, confidence=round(conf, 4), triggered_rules=["R5:nli_contradiction"],
                rationale=v.explanation, response_text=self._challenge_text(t, v.explanation, v.premise.text),
            )
        if logic.status is LogicStatus.UNDERDETERMINED:
            return self._abstain(t, ["R6:underdetermined_target"], 0.9, logic.explanation)
        if unresolved:
            return self._abstain(t, ["R7:unresolved_referent"], 0.85, "; ".join(v.explanation for v in unresolved))
        u = t.uncertainty
        if u is not None and u.n_samples > 1 and u.normalized_entropy > self.config.entropy_tau:
            grounded = self._evidence_answer(t)
            if grounded is not None:
                score, ent, text = grounded
                return PolicyDecision(
                    decision=DecisionEnum.PASS, confidence=round(min(0.9, score * ent), 4), triggered_rules=["R8b:evidence_answer"],
                    rationale=(f"Model niespójny (entropia {u.normalized_entropy:.2f} > {self.config.entropy_tau}), "
                               f"ale fragment bazy wiedzy potwierdza założenie pytania (podobieństwo {score:.2f}, entailment {ent:.2f})."),
                    response_text=f"Według bazy wiedzy: {text}",
                )
            return self._abstain(t, ["R8:semantic_entropy"], round(min(0.95, u.normalized_entropy), 4),
                                 f"Wysoka entropia semantyczna ({u.normalized_entropy:.2f} > {self.config.entropy_tau}).")
        conf = 0.6 + 0.3 * ev.support_ratio
        if u is not None and u.n_samples > 1:
            conf *= 1.0 - 0.5 * u.normalized_entropy
        return PolicyDecision(decision=DecisionEnum.PASS, confidence=round(min(conf, 0.95), 4), triggered_rules=["R9:pass"],
                              rationale="Brak naruszeń: założenia niesprzeczne, niska niepewność.")

    def _evidence_answer(self, t: MetacognitiveTelemetry) -> tuple[float, float, str] | None:
        """Best KB passage that NLI-entails a non-referent premise of the query, if strong enough."""
        if not self.config.evidence_answer:
            return None
        best: tuple[float, float, str] | None = None
        for v in t.premise_verdicts:
            if v.status is not PremiseStatus.SUPPORTED or v.premise.kind is PremiseKind.REFERENT or not v.evidence:
                continue
            top = v.evidence[0]
            ent = float(v.nli_scores.get(NLILabel.ENTAILMENT, 0.0))
            if top.score < self.config.evidence_answer_min_similarity or ent < self.config.evidence_answer_min_entailment:
                continue
            if best is None or top.score > best[0]:
                best = (float(top.score), ent, top.text)
        return best

    # -- response templates ----------------------------------------------------------
    def _paradox(self, t: MetacognitiveTelemetry, rules: list[str], conf: float) -> PolicyDecision:
        name, resolution = "paradoks samoodniesienia", ""
        match = t.evidence.paradox_match
        if match is not None and self.store is not None:
            doc = self.store.get(match.doc_id)
            if doc is not None:
                name, resolution = doc.name or name, doc.resolution or ""
        parts = [f"To jest {name} – pytanie nie ma prostej odpowiedzi typu tak/nie w przyjętych założeniach."]
        if t.logic.status is LogicStatus.PARADOX_SELF_REF:
            parts.append(t.logic.explanation)
        if resolution:
            parts.append(f"Rozstrzygnięcie: {resolution}")
        return PolicyDecision(decision=DecisionEnum.EXPLAIN_PARADOX, confidence=round(conf, 4), triggered_rules=rules,
                              rationale=t.logic.explanation or name, response_text=" ".join(parts))

    @staticmethod
    def _challenge_text(t: MetacognitiveTelemetry, reason: str, premise: str | None = None) -> str:
        if premise is None:
            pres = [p.text for p in t.premises if p.kind is PremiseKind.PRESUPPOSITION]
            premise = pres[0] if pres else t.query
        return f"Pytanie opiera się na fałszywym założeniu: „{premise}”. {reason}"

    @staticmethod
    def _abstain(t: MetacognitiveTelemetry, rules: list[str], conf: float, rationale: str) -> PolicyDecision:
        missing = t.evidence.missing_variables
        text = "Nie da się udzielić wiarygodnej odpowiedzi na podstawie podanych informacji."
        if missing:
            text += " Brakuje: " + "; ".join(missing) + "."
        return PolicyDecision(decision=DecisionEnum.ABSTAIN, confidence=conf, triggered_rules=rules, rationale=rationale, response_text=text)
