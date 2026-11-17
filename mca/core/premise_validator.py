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
from mca.telemetry.schemas import AnswerabilityReport, NLILabel, Premise, PremiseKind, PremiseStatus, PremiseVerdict
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

ANSWERABILITY_PROMPT = """Czy na poniższe pytanie da się udzielić jednoznacznej odpowiedzi wyłącznie na podstawie jego treści
i powszechnej wiedzy (faktów, definicji, rachunków)? Odpowiedz NIE, jeśli brakuje danych, których nie ma w pytaniu
(konkretnych wartości, kontekstu, treści pliku, zdjęcia lub dokumentu, tożsamości osoby, miejsca albo czasu).
W przeciwnym razie odpowiedz TAK. Odpowiedz jednym słowem: TAK albo NIE.

Pytanie: {query}
Odpowiedź:"""

ANSWERABILITY_FEWSHOT = """Oceń, czy na pytanie da się jednoznacznie odpowiedzieć wyłącznie na podstawie jego treści i wiedzy ogólnej.
Jeśli brakuje liczby, wartości, kontekstu, dokumentu, osoby, miejsca albo czasu, odpowiedz NIE.
Jeśli wystarczy wiedza ogólna, definicja albo rachunek, odpowiedz TAK. Odpowiadaj jednym słowem.

Pytanie: Ile wynosi 6 razy 7?
Odpowiedź: TAK
Pytanie: Prostopadłościenne akwarium ma długość 50 cm. Ile litrów wody mieści?
Odpowiedź: NIE
Pytanie: Jak nazywa się stolica Hiszpanii?
Odpowiedź: TAK
Pytanie: Pan Nowak wysłał paczkę. Ile ważyła?
Odpowiedź: NIE
Pytanie: Ile dni ma tydzień?
Odpowiedź: TAK
Pytanie: Zdjęcie zrobiono w parku. O której godzinie je zrobiono?
Odpowiedź: NIE

Pytanie: {query}
Odpowiedź:"""

ANSWERABILITY_MISSING = """Sprawdź, czy pytanie zawiera wszystkie dane potrzebne do jednoznacznej odpowiedzi (wiedza ogólna,
definicje i rachunki są dostępne). Nie odpowiadaj na samo pytanie. Jeśli niczego nie brakuje, napisz tylko: KOMPLETNE.
Jeśli czegoś brakuje, wypisz krótko, czego (np. „cena biletu”, „treść pliku”).

Pytanie: {query}
Brakujące dane:"""

ANSWERABILITY_INVERSE = """Czy w poniższym pytaniu brakuje danych potrzebnych do jednoznacznej odpowiedzi (konkretnych wartości,
kontekstu, treści pliku, zdjęcia lub dokumentu, tożsamości osoby, miejsca albo czasu)? Wiedza ogólna, definicje i rachunki
są dostępne. Odpowiedz jednym słowem: TAK (brakuje) albo NIE (niczego nie brakuje).

Pytanie: {query}
Odpowiedź:"""

_YES = re.compile(r"\b(tak|yes)\b", re.IGNORECASE)
_NO = re.compile(r"\b(nie|no)\b", re.IGNORECASE)


def parse_answerability(text: str) -> bool | None:
    """TAK -> True, NIE -> False; whichever comes first wins, ``None`` when neither appears."""
    yes, no = _YES.search(text), _NO.search(text)
    if yes and (not no or yes.start() < no.start()):
        return True
    if no:
        return False
    return None


def parse_missing(text: str) -> bool | None:
    """'KOMPLETNE' -> answerable; any other non-empty reply names missing data -> not answerable."""
    t = text.strip()
    if not t:
        return None
    return True if re.search(r"komplet", t, re.IGNORECASE) else False


def _parse_inverse(text: str) -> bool | None:
    v = parse_answerability(text)
    return None if v is None else not v


PROBE_VARIANTS = {  # variant -> (template, parser, minimum token budget)
    "direct": (ANSWERABILITY_PROMPT, parse_answerability, 1),
    "fewshot": (ANSWERABILITY_FEWSHOT, parse_answerability, 1),
    "missing": (ANSWERABILITY_MISSING, parse_missing, 24),
    "inverse": (ANSWERABILITY_INVERSE, _parse_inverse, 1),
}


async def probe_answerability(llm: LLMBackend, query: str, variant: str = "direct", max_new_tokens: int = 8) -> AnswerabilityReport:
    """Ask ``llm`` once (greedy, no reasoning phase) whether ``query`` is answerable as posed."""
    template, parse, min_tok = PROBE_VARIANTS[variant]
    prompt, n_tok = template.format(query=query), max(max_new_tokens, min_tok)
    many = getattr(llm, "generate_many", None)
    raw = (await many(prompt, 1, temperature=0.0, max_new_tokens=n_tok))[0] if many else \
        await llm.generate(prompt, temperature=0.0, max_new_tokens=n_tok)
    return AnswerabilityReport(answerable=parse(raw), raw=raw.strip()[:200], variant=variant)


_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_POSSESSIVE = re.compile(r"\b(?:m[óo]j|moja|moje|mojego|mojej|moim|mojemu|mych|moich|nasz\w*)\s+(\w+(?:\.\w+)?)", re.IGNORECASE)
_DEIXIS = re.compile(r"\b(stąd|dotąd|tutaj|wczoraj\w*|dzisiejsz\w*|jutrzejsz\w*|przedwczoraj\w*|ubieg\w+\s+tygodni\w*)\b", re.IGNORECASE)
_DEMONSTRATIVE = re.compile(r"^\s*(?:ten|ta|to|tego|tej|tym|tę|tą|tamt\w+|ów|owa|owe|this|that|these|those)\b", re.IGNORECASE)


def is_specific_referent(text: str, query: str) -> bool:
    """True when ``text`` names something only the user's context can resolve.

    That covers possessives, deixis, a leading demonstrative, or a word that ``query`` capitalises
    mid-sentence (a proper name). Generic noun phrases ("samochód", "kolor piłki") are not specific.
    """
    if _POSSESSIVE.search(text) or _DEIXIS.search(text) or _DEMONSTRATIVE.search(text):
        return True
    words = set(re.findall(r"\w+", text))
    for sentence in re.split(r"(?<=[.!?])\s+", query):
        tokens = re.findall(r"\w+", sentence)[1:]
        if any(tok in words and len(tok) >= 3 and tok[0].isupper() for tok in tokens):
            return True
    return False


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

    async def answerability(self, query: str) -> AnswerabilityReport | None:
        """One greedy probe (wording from ``config.answerability_prompt``); reasoning models skip thinking."""
        if self.llm is None or not self.config.answerability_check:
            return None
        try:
            return await probe_answerability(self.llm, query, self.config.answerability_prompt,
                                             self.config.answerability_max_new_tokens)
        except Exception as exc:  # noqa: BLE001 - the probe must not break validation
            logger.warning("answerability probe failed: %s", exc)
            return None

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
            # Rule-detected referents (possessives, deixis, unknown names) are unresolvable by construction.
            # LLM-proposed referents are only hints: small models flag almost every noun phrase, so they
            # count as resolved when the knowledge base grounds them.
            if premise.source == "llm":
                grounded = [e for e in self.store.search(premise.text, k=1) if e.score >= self.config.retrieval_min_similarity]
                if grounded:
                    return PremiseVerdict(premise=premise, status=PremiseStatus.SUPPORTED, evidence=grounded,
                                          explanation=f"Odniesienie ugruntowane w bazie wiedzy: „{grounded[0].text}”")
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
