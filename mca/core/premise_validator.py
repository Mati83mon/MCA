"""Layer 1 – Premise Validator.

1. **Decomposition** of the query into atomic premises:
   * rule-based presupposition triggers (``Dlaczego X?`` presupposes *X*,
     ``Która N jest <Name>?`` presupposes *<Name> jest N*, other wh-questions
     carry an existential presupposition, ``Czy X?`` merely *asks* X);
   * unresolved referents (first-person possessives, deixis, unknown proper names);
   * optional LLM decomposition returning JSON (premises + a formal spec in the Z3 DSL);
   * the rule-based :class:`~mca.core.formalizer.RuleBasedFormalizer`.
2. **Validation** of every premise: symbolic oracles first, then NLI
   (``cross-encoder/nli-deberta-v3-large`` or lexical fallback) against evidence
   retrieved from the :class:`~mca.core.evidence_verifier.EvidenceStore`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from mca.config import MCAConfig
from mca.core.evidence_verifier import EvidenceStore
from mca.core.formalizer import RuleBasedFormalizer
from mca.core.oracles import DEFAULT_ORACLES, NumberTheoryOracle, PythonSemanticsOracle
from mca.llm.base import LLMBackend, extract_json
from mca.telemetry.schemas import NLILabel, Premise, PremiseKind, PremiseStatus, PremiseVerdict
from mca.utils.nli_client import NLIClient
from mca.utils.z3_helpers import FormalSpec

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """Rozłóż pytanie użytkownika na atomowe założenia. Zwróć WYŁĄCZNIE obiekt JSON:
{{
  "presuppositions": [{{"text": "zdanie oznajmujące, które pytanie przyjmuje za prawdziwe", "existential": false}}],
  "referents": [{{"text": "odniesienie, którego nie da się rozwiązać bez kontekstu", "missing": "czego brakuje"}}],
  "formal": null
}}
Pole "formal" wypełnij tylko dla twierdzeń matematycznych/logicznych, w formacie:
{{"vars": {{"x": "Real"}}, "assertions": [{{"id": "a1", "expr": "x * x == -9", "text": "opis"}}], "target": null}}
Dozwolone: + - * / %, porównania, And, Or, Not, Implies, Iff, ForAll(x, ...), Exists(x, ...). "target" ustaw na
szukaną wielkość, jeśli pytanie prosi o wartość liczbową.

Pytanie: {query}
JSON:"""

_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_POSSESSIVE = re.compile(r"\b(?:m[óo]j|moja|moje|mojego|mojej|moim|mojemu|mych|moich|nasz\w*)\s+(\w+(?:\.\w+)?)", re.IGNORECASE)
_DEIXIS = re.compile(r"\b(stąd|dotąd|tutaj|wczoraj\w*|dzisiejsz\w*|jutrzejsz\w*|przedwczoraj\w*|ubieg\w+\s+tygodni\w*)\b", re.IGNORECASE)
_DEIXIS_MISSING = {
    "stąd": "lokalizacja punktu startowego (i sposób pomiaru odległości)",
    "tutaj": "lokalizacja, do której odnosi się pytanie",
    "dotąd": "lokalizacja, do której odnosi się pytanie",
}


@dataclass
class Decomposition:
    premises: list[Premise] = field(default_factory=list)
    specs: list[FormalSpec] = field(default_factory=list)


def _question_sentence(query: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", query.strip()) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    return questions[-1] if questions else (sentences[-1] if sentences else query)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().rstrip("?").strip()
    return text[:1].upper() + text[1:] if text else text


class RulePremiseExtractor:
    """Presupposition triggers and referent detection without an LLM."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store = store

    def presuppositions(self, query: str) -> list[Premise]:
        q = _question_sentence(query)
        m = re.match(r"\s*(?:Dlaczego|Czemu|Why|Z jakiego powodu)\s+(.+)", q, re.IGNORECASE)
        if m:
            return [Premise(id="p1", text=_clean(m.group(1)), kind=PremiseKind.PRESUPPOSITION)]
        m = re.match(r"\s*(?:Czy|Is|Are|Does|Do|Can)\s+(.+)", q)
        if m:
            return [Premise(id="p1", text=_clean(m.group(1)), kind=PremiseKind.ASSERTION)]
        m = re.match(rf"\s*Któr\w+\s+(.+?)\s+jest\s+([{_UPPER}]\w+)\s*\??$", q)
        if m:
            return [Premise(id="p1", text=f"{m.group(2)} jest {m.group(1)}", kind=PremiseKind.PRESUPPOSITION)]
        m = re.match(r"\s*(?:Któr\w+|Jak\w*|Ile|Kto|Co|Gdzie|Kiedy|Which|What|How many|Who)\s+(.+)", q, re.IGNORECASE)
        if m:
            return [Premise(id="p1", text=_clean(m.group(1)), kind=PremiseKind.PRESUPPOSITION, existential=True)]
        return []

    def referents(self, query: str) -> list[Premise]:
        found: list[Premise] = []
        for m in _POSSESSIVE.finditer(query):
            obj = m.group(1)
            found.append(Premise(
                id=f"r{len(found) + 1}", text=m.group(0), kind=PremiseKind.REFERENT,
                missing_variable=f"dane/dostęp do obiektu „{m.group(0)}” (kontekst użytkownika)",
            ))
            if re.fullmatch(r"\w+\.\w+", obj):
                found[-1].missing_variable = f"zawartość pliku {obj} lub wynik jego analizy"
        for m in _DEIXIS.finditer(query):
            word = m.group(1).lower()
            missing = _DEIXIS_MISSING.get(word, "data odniesienia oraz identyfikacja zdarzenia")
            found.append(Premise(id=f"r{len(found) + 1}", text=m.group(0), kind=PremiseKind.REFERENT, missing_variable=missing))
        for sentence in re.split(r"(?<=[.!?])\s+", query):
            unquoted = re.sub(r"[„\"“][^”\"“]*[”\"]", " ", sentence)
            tokens = re.findall(r"[\w.]+", unquoted)
            for tok in tokens[1:]:
                tok = tok.strip(".")
                if (
                    len(tok) >= 3 and tok[0].isupper() and not tok.isupper() and tok.isalpha()
                    and not self.store.mentions(tok)
                ):
                    found.append(Premise(
                        id=f"r{len(found) + 1}", text=tok, kind=PremiseKind.REFERENT,
                        missing_variable=f"tożsamość „{tok}” i potrzebne dane (np. data urodzenia, data odniesienia)",
                    ))
        return found


class PremiseValidator:
    """Decomposes queries and validates premises against oracles and evidence."""

    def __init__(
        self,
        config: MCAConfig,
        store: EvidenceStore,
        nli: NLIClient,
        llm: LLMBackend | None = None,
        formalizer: RuleBasedFormalizer | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.nli = nli
        self.llm = llm
        self.formalizer = formalizer or RuleBasedFormalizer()
        self.extractor = RulePremiseExtractor(store)
        self.oracles: tuple[PythonSemanticsOracle | NumberTheoryOracle, ...] = DEFAULT_ORACLES

    # -- decomposition -----------------------------------------------------------
    async def _llm_decompose(self, query: str) -> Decomposition:
        if self.llm is None or not self.config.use_llm_decomposition:
            return Decomposition()
        try:
            raw = await self.llm.generate(DECOMPOSITION_PROMPT.format(query=query), temperature=0.0, max_new_tokens=384)
        except Exception as exc:  # noqa: BLE001 - LLM failures must not break validation
            logger.warning("LLM decomposition failed: %s", exc)
            return Decomposition()
        data = extract_json(raw) or {}
        out = Decomposition()
        for i, item in enumerate(data.get("presuppositions") or []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text:
                out.premises.append(Premise(id=f"l{i + 1}", text=_clean(text), source="llm",
                                            existential=bool(item.get("existential")) if isinstance(item, dict) else False))
        for i, item in enumerate(data.get("referents") or []):
            if isinstance(item, dict) and item.get("text"):
                out.premises.append(Premise(id=f"lr{i + 1}", text=item["text"], kind=PremiseKind.REFERENT, source="llm",
                                            missing_variable=item.get("missing") or item["text"]))
        formal = data.get("formal")
        if isinstance(formal, dict) and formal.get("assertions"):
            try:
                out.specs.append(FormalSpec.model_validate({**formal, "label": "llm.formal"}))
            except ValidationError as exc:
                logger.info("discarding invalid LLM formal spec: %s", exc)
        return out

    async def decompose(self, query: str) -> Decomposition:
        llm_part = await self._llm_decompose(query)
        premises = self.extractor.presuppositions(query) + self.extractor.referents(query) + llm_part.premises
        seen: set[str] = set()
        unique: list[Premise] = []
        for p in premises:
            key = p.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        specs = self.formalizer.formalize(query) + llm_part.specs
        return Decomposition(premises=unique, specs=specs)

    # -- validation ----------------------------------------------------------------
    def _validate_one(self, premise: Premise, query: str) -> PremiseVerdict:
        if premise.kind is PremiseKind.REFERENT:
            return PremiseVerdict(premise=premise, status=PremiseStatus.UNVERIFIED,
                                  explanation=f"Nierozwiązane odniesienie: {premise.missing_variable}")
        for oracle in self.oracles:
            for text in (premise.text, query):
                if oracle.applies(text):
                    verdict = oracle.evaluate(text)
                    if verdict is not None:
                        return PremiseVerdict(premise=premise, status=verdict.status, oracle=verdict.oracle,
                                              nli_label=NLILabel.CONTRADICTION if verdict.status is PremiseStatus.CONTRADICTED else NLILabel.ENTAILMENT,
                                              explanation=verdict.explanation)
        evidence = [e for e in self.store.search(premise.text, k=self.config.retrieval_top_k)
                    if e.score >= self.config.retrieval_min_similarity]
        if not evidence:
            return PremiseVerdict(premise=premise, status=PremiseStatus.UNVERIFIED, explanation="Brak dowodów w bazie wiedzy.")
        scores = self.nli.predict([(e.text, premise.text) for e in evidence])
        best_ent = max(range(len(scores)), key=lambda i: scores[i][NLILabel.ENTAILMENT])
        best_con = max(range(len(scores)), key=lambda i: scores[i][NLILabel.CONTRADICTION])
        ent, con = scores[best_ent][NLILabel.ENTAILMENT], scores[best_con][NLILabel.CONTRADICTION]
        thr = self.config.nli_threshold
        if con >= thr and con > ent and not premise.existential:
            return PremiseVerdict(premise=premise, status=PremiseStatus.CONTRADICTED, nli_label=NLILabel.CONTRADICTION,
                                  nli_scores=scores[best_con], evidence=[evidence[best_con]],
                                  explanation=f"Sprzeczne z: „{evidence[best_con].text}”")
        if ent >= thr:
            return PremiseVerdict(premise=premise, status=PremiseStatus.SUPPORTED, nli_label=NLILabel.ENTAILMENT,
                                  nli_scores=scores[best_ent], evidence=[evidence[best_ent]],
                                  explanation=f"Potwierdzone przez: „{evidence[best_ent].text}”")
        return PremiseVerdict(premise=premise, status=PremiseStatus.UNVERIFIED, nli_label=NLILabel.NEUTRAL,
                              nli_scores=scores[best_ent], evidence=evidence[:2], explanation="Dowody neutralne wobec założenia.")

    def validate(self, premises: list[Premise], query: str) -> list[PremiseVerdict]:
        return [self._validate_one(p, query) for p in premises]

    async def avalidate(self, premises: list[Premise], query: str) -> list[PremiseVerdict]:
        return await asyncio.to_thread(self.validate, premises, query)
