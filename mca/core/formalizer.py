"""Rule-based semantic parser: natural-language (Polish) query -> :class:`FormalSpec`.

This is the symbolic fallback of the neuro-symbolic translation step. When an
LLM is attached, :class:`~mca.core.premise_validator.PremiseValidator` also asks
it for formal specs in the same DSL; both sources are merged and every spec is
checked by Z3 in :mod:`mca.core.logic_engine`.

Rules cover recurring *templates*, not individual questions:

* self-reference (named sentences, diagonal definitions),
* existential presuppositions over numeric domains, superlatives over unbounded domains,
* arithmetic claims (division, fractions, percentage chains, probability bounds, parity),
* cardinality/ordinal presuppositions (``n``-th element of an ``m``-element collection, empty containers),
* invalid inference (converse of a universal implication),
* quantity schemas for under-determined questions (speed, area, mean, energy, conversions, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from mca.utils.text import CARDINALS, cardinal_value, ordinal_value, stem, tokenize
from mca.utils.z3_helpers import Assertion, FormalSpec

RuleFn = Callable[[str], list[FormalSpec]]

POLYGON_SIDES: dict[str, int] = {
    "trójką": 3, "czworo": 4, "kwadra": 4, "prosto": 4, "romb": 4, "pięcio": 5, "sześci": 6, "ośmiok": 8,
}


def _num(text: str) -> float:
    return float(text.replace(" ", "").replace(" ", "").replace(",", "."))


def _word_number(token: str) -> int | None:
    token = token.lower()
    value = cardinal_value(token)
    if value is not None:
        return value
    return ordinal_value(token)


def _last_question(query: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", query.strip())
    questions = [s for s in sentences if s.endswith("?")]
    return questions[-1] if questions else sentences[-1]


def _spec(label: str, vars: dict[str, str], assertions: list[tuple[str, str, str]], **kw: object) -> FormalSpec:
    return FormalSpec(
        label=label,
        vars=vars,
        assertions=[Assertion(id=i, expr=e, text=t) for i, e, t in assertions],
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Self-reference
# ---------------------------------------------------------------------------
_NAMED_SENTENCE = re.compile(r"[Zz]dani\w*\s+([A-Z]\w*)\s*(?:brzmi|głosi|mówi)?:?\s*[„\"]([^”\"]+)[”\"]")
_SENTENCE_REF = re.compile(r"[Zz]danie\s+([A-Z]\w*)\s+jest\s+(prawdziwe|fałszywe)")


def rule_named_self_reference(query: str) -> list[FormalSpec]:
    definitions: dict[str, str] = {}
    for name, content in _NAMED_SENTENCE.findall(query):
        ref = _SENTENCE_REF.search(content)
        if ref:
            target, value = ref.groups()
            definitions[name] = target if value == "prawdziwe" else f"Not({target})"
    if not definitions or any(re.findall(r"[A-Za-z_]\w*", d)[-1] not in definitions for d in definitions.values()):
        return []
    assertions = [(f"def_{n}", f"{n} == {d}", f"{n} ⇔ {d}") for n, d in definitions.items()]
    return [_spec("self_reference.named", {n: "Bool" for n in definitions}, assertions, self_reference=definitions)]


_DIAGONAL = re.compile(
    r"\bnie\s+(\w+)(?:\s+się)?(?:\s+do)?\s+(?:sam(?:i|ych|ego|a)?\s+siebie|sam(?:i|ych|ego|a)?\b)", re.IGNORECASE
)


def rule_diagonal(query: str) -> list[FormalSpec]:
    m = _DIAGONAL.search(query)
    if not m:
        return []
    question = _last_question(query)
    content = [stem(t) for t in tokenize(question) if len(t) > 3 or t.isupper() or len(t) == 1 and t.isalpha()]
    raw = [t for t in re.findall(r"\w+", question) if t[:1].isupper() or len(t) > 3]
    repeated = len(content) != len(set(content)) or len(raw) != len(set(raw))
    if not (repeated or re.search(r"\bsiebie\b|\bsam\w*\b", question)):
        return []
    verb = m.group(1)
    return [
        FormalSpec(
            label="self_reference.diagonal",
            sorts=["Obj"],
            vars={"d": "Obj", "x": "Obj"},
            functions={"R": ["Obj", "Obj", "Bool"]},
            assertions=[Assertion(id="diagonal_definition", expr="ForAll(x, R(d, x) == Not(R(x, x)))",
                                  text=f"d jest w relacji „{verb}” dokładnie z tymi x, które nie są w niej z samymi sobą")],
            self_reference={"Rdd": "Not(Rdd)"},
        )
    ]


# ---------------------------------------------------------------------------
# Existential presuppositions over numbers
# ---------------------------------------------------------------------------
def rule_numeric_existence(query: str) -> list[FormalSpec]:
    q = _last_question(query)
    if not re.match(r"\s*(Która|Jaka|Jakie|Jaki|Ile|Podaj|Which|What)", q):
        return []
    dom = re.search(r"liczb\w*\s+(całkowit|rzeczywist|naturaln)\w*|(?:dodatni|ujemn)\w*\s+liczb\w*\s+(całkowit|rzeczywist|naturaln)", q)
    if not dom:
        return []
    kind = dom.group(1) or dom.group(2)
    sort = "Real" if kind.startswith("rzecz") else "Int"
    asserts: list[tuple[str, str, str]] = []
    if kind.startswith("natur"):
        asserts.append(("domain", "x >= 0", "x jest liczbą naturalną"))
    if re.search(r"\bdodatni", q):
        asserts.append(("positive", "x > 0", "x jest dodatnia"))
    if re.search(r"\bujemn", q):
        asserts.append(("negative", "x < 0", "x jest ujemna"))
    for op, word in (("<", "mniejsz"), (">", "większ")):
        cmp = re.search(word + r"\w*\s+od\s+(zera|-?\d+(?:[.,]\d+)?)", q)
        if cmp:
            bound = "0" if cmp.group(1) == "zera" else str(_num(cmp.group(1)))
            asserts.append((f"cmp_{word}", f"x {op} {bound}", f"x {op} {bound}"))
    sq = re.search(r"(?:do kwadratu|kwadrat\w*)\s+(?:daje|wynosi|równa się|jest równ\w*)\s+(-?\d+(?:[.,]\d+)?)", q)
    if sq:
        asserts.append(("square", f"x * x == {_num(sq.group(1))}", f"x² = {sq.group(1)}"))
    if len(asserts) < 2 and not sq:
        return []
    return [_spec("existence.numeric", {"x": sort}, asserts)]


def rule_unbounded_superlative(query: str) -> list[FormalSpec]:
    q = _last_question(query)
    if re.search(r"największ\w*\s+liczb\w*\s+(naturaln|całkowit)", q):
        return [_spec("existence.superlative", {"n": "Int", "m": "Int"}, [
            ("domain", "n >= 0", "n jest liczbą naturalną"),
            ("maximum", "ForAll(m, Implies(m >= 0, m <= n))", "n jest nie mniejsza od każdej liczby naturalnej"),
        ])]
    if re.search(r"najmniejsz\w*\s+dodatni\w*\s+liczb\w*\s+rzeczywist", q):
        return [_spec("existence.superlative", {"x": "Real", "y": "Real"}, [
            ("positive", "x > 0", "x > 0"),
            ("minimum", "ForAll(y, Implies(y > 0, x <= y))", "x jest nie większa od każdej dodatniej liczby rzeczywistej"),
        ])]
    return []


# ---------------------------------------------------------------------------
# Arithmetic claims
# ---------------------------------------------------------------------------
def rule_division(query: str) -> list[FormalSpec]:
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", query)
    if not m or re.search(r"(większ|mniejsz)", query) or not re.search(r"wynik|ile wynosi|jaka liczba|równa", query, re.I):
        return []
    a, b = int(m.group(1)), int(m.group(2))
    return [_spec("arithmetic.division", {"x": "Real"}, [("quotient", f"x * {b} == {a}", f"x = {a}/{b}, czyli x·{b} = {a}")])]


def rule_fraction_comparison(query: str) -> list[FormalSpec]:
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+jest\s+(większe|mniejsze)\s+od\s+(\d+)\s*/\s*(\d+)", query)
    if not m:
        return []
    a, b, rel, c, d = m.groups()
    op = ">" if rel == "większe" else "<"
    return [_spec("arithmetic.fraction_order", {}, [("claim", f"{a}/{b} {op} {c}/{d}", f"{a}/{b} {op} {c}/{d}")])]


def rule_percent_chain(query: str) -> list[FormalSpec]:
    m = re.search(r"(obniżk|podwyżk)\w*.*?o\s*(\d+(?:[.,]\d+)?)\s*%.*?(obniżk|podwyżk)\w*.*?o\s*(\d+(?:[.,]\d+)?)\s*%", query)
    if not m or not re.search(r"przywraca|wraca|początkow|taka sama|ta sama", query):
        return []
    f1 = (1 - _num(m.group(2)) / 100) if m.group(1) == "obniżk" else (1 + _num(m.group(2)) / 100)
    f2 = (1 - _num(m.group(4)) / 100) if m.group(3) == "obniżk" else (1 + _num(m.group(4)) / 100)
    return [_spec("arithmetic.percent_chain", {"p": "Real"}, [
        ("price_positive", "p > 0", "cena początkowa p > 0"),
        ("restores", f"p * {f1} * {f2} == p", f"p · {f1:g} · {f2:g} = p (cena wraca do początkowej)"),
    ])]


def rule_probability_bounds(query: str) -> list[FormalSpec]:
    m = re.search(r"prawdopodobieństw\w*\s+(?:równe|równym|wynosząc\w*|=|o wartości)\s*(-?\d+(?:[.,]\d+)?)\s*(%?)", query)
    if not m:
        return []
    value = _num(m.group(1)) / (100 if m.group(2) else 1)
    return [_spec("arithmetic.probability", {"p": "Real"}, [
        ("kolmogorov", "And(p >= 0, p <= 1)", "aksjomat: 0 ≤ P ≤ 1"),
        ("stated", f"p == {value}", f"P = {m.group(1)}{m.group(2)}"),
    ])]


def rule_parity(query: str) -> list[FormalSpec]:
    m = re.search(r"każd\w*\s+liczb\w*\s+(nie)?parzyst\w*\s+(?:jest\s+)?podzieln\w*\s+przez\s+2\s+bez\s+reszty", query)
    if not m:
        return []
    remainder = 1 if m.group(1) else 0
    return [_spec("arithmetic.parity", {"n": "Int"}, [
        ("definition", f"n % 2 == {remainder}", "definicja " + ("nieparzystości" if remainder else "parzystości")),
        ("claim", "n % 2 == 0", "n dzieli się przez 2 bez reszty"),
    ])]


def rule_fair_coin_must(query: str) -> list[FormalSpec]:
    if not (re.search(r"uczciw\w+\s+(monet|kostk)", query) and re.search(r"\bmusi\b", query) and re.search(r"niezależn", query)):
        return []
    sides = 6 if "kostk" in query else 2
    return [_spec("probability.gambler", {"p": "Real"}, [
        ("independence", f"p == 1/{sides}", f"niezależność i uczciwość: P = 1/{sides} niezależnie od historii"),
        ("must", "p == 1", "„musi” oznacza P = 1"),
    ])]


# ---------------------------------------------------------------------------
# Cardinality / ordinals
# ---------------------------------------------------------------------------
_ORDINAL_NOUN = re.compile(r"\b(\w+)\s+(bok\w*|planet\w*|element\w*|pozycj\w*|miejsc\w*|wierzchoł\w*|kąt\w*)\b", re.IGNORECASE)


def rule_ordinal_cardinality(query: str) -> list[FormalSpec]:
    specs: list[FormalSpec] = []
    for ord_word, noun in _ORDINAL_NOUN.findall(query):
        k = ordinal_value(ord_word.lower())
        if k is None:
            continue
        noun_stem = noun.lower()[:5]
        size: int | None = None
        container = ""
        # explicit count: "<count> <noun>" e.g. "ośmiu planet"
        for cnt_word, cnt_noun in re.findall(r"\b(\w+)(?=\s+(\w+))", query):
            if cnt_noun.lower().startswith(noun_stem) and cnt_word.lower() in CARDINALS:
                size, container = CARDINALS[cnt_word.lower()], f"{cnt_word} {cnt_noun}"
        if size is None and noun_stem.startswith("bok"):
            after = query[query.lower().find(noun.lower()) + len(noun):]
            shape = re.search(r"\b(\w+)", after)
            if shape:
                for prefix, sides in POLYGON_SIDES.items():
                    if shape.group(1).lower().startswith(prefix):
                        size, container = sides, shape.group(1)
        if size is not None:
            specs.append(_spec("cardinality.ordinal", {"k": "Int", "size": "Int"}, [
                ("size", f"size == {size}", f"{container}: {size} elementów"),
                ("index", f"k == {k}", f"pytanie zakłada pozycję nr {k}"),
                ("bounds", "And(k >= 1, k <= size)", "pozycja musi mieścić się w kolekcji"),
            ]))
    return specs


def rule_polygon_sides(query: str) -> list[FormalSpec]:
    m = re.search(r"\b(\w+)\s+o\s+(\w+)\s+bok", query)
    if not m:
        return []
    shape, count_word = m.group(1).lower(), m.group(2).lower()
    stated = _word_number(count_word)
    for prefix, sides in POLYGON_SIDES.items():
        if shape.startswith(prefix) and stated is not None:
            return [_spec("cardinality.polygon", {"s": "Int"}, [
                ("definition", f"s == {sides}", f"{m.group(1)} ma z definicji {sides} boki"),
                ("stated", f"s == {stated}", f"pytanie zakłada {stated} boków"),
            ])]
    return []


def rule_empty_container(query: str) -> list[FormalSpec]:
    if not re.search(r"pust\w*\s+(list|liś|zbior|zbiór|tablic|krotk|słownik)|(zbior|zbiór|list)\w*\s+pust", query, re.IGNORECASE):
        return []
    if not re.search(r"element|pozycj|któr|pierwsz", query, re.IGNORECASE):
        return []
    return [_spec("cardinality.empty", {"size": "Int", "i": "Int"}, [
        ("empty", "size == 0", "kontener jest pusty"),
        ("member", "And(i >= 1, i <= size)", "pytanie zakłada istnienie elementu"),
    ])]


def rule_race_overtake(query: str) -> list[FormalSpec]:
    m = re.search(r"wyprzedz\w*\s+(?:\w+\s+){0,2}na\s+(\w+)\s+miejscu", query)
    c = re.search(r"zajmuj\w*\s+(?:teraz\s+)?(\w+)\s+miejsce", query)
    if not (m and c):
        return []
    passed, claimed = ordinal_value(m.group(1).lower()), ordinal_value(c.group(1).lower())
    if passed is None or claimed is None:
        return []
    return [_spec("ordering.overtake", {"other": "Int", "me_before": "Int", "me_after": "Int"}, [
        ("other_position", f"other == {passed}", f"wyprzedzana osoba jest na miejscu {passed}"),
        ("behind", "me_before == other + 1", "przed manewrem jesteś bezpośrednio za nią"),
        ("swap", "me_after == other", "po wyprzedzeniu przejmujesz jej pozycję"),
        ("claim", f"me_after == {claimed}", f"pytanie zakłada miejsce {claimed}"),
    ])]


def rule_count_of_defined_none(query: str) -> list[FormalSpec]:
    m = re.search(r"[Ii]le\s+(\w+)\s+ma\b", query)
    if not m:
        return []
    noun = m.group(1).lower()[:6]
    if not re.search(r"nie\s+ma\s+(?:żadn\w+\s+)?" + re.escape(noun), query.lower()):
        return []
    return [_spec("cardinality.defined_none", {"c": "Int"}, [
        ("definition", "c == 0", f"z opisu: brak ({m.group(1)})"),
        ("presupposition", "c >= 1", f"pytanie „ile {m.group(1)} ma” zakłada niezerową liczbę"),
    ])]


# ---------------------------------------------------------------------------
# Inference validity
# ---------------------------------------------------------------------------
def rule_converse(query: str) -> list[FormalSpec]:
    universals = re.findall(r"wszystkie\s+(\w+)\s+są\s+(\w+)", query)
    if len(universals) < 2 or not re.search(r"wynika|implikuje|dowodzi|follows", query):
        return []
    (a, b), (c, d) = universals[0], universals[-1]
    if not (c.lower()[:3] == b.lower()[:3] and d.lower()[:3] == a.lower()[:3]):
        return []
    return [FormalSpec(
        label="inference.converse",
        sorts=["Obj"],
        vars={"x": "Obj"},
        functions={"A": ["Obj", "Bool"], "B": ["Obj", "Bool"]},
        assertions=[Assertion(id="premise", expr="ForAll(x, Implies(A(x), B(x)))", text=f"wszystkie {a} są {b}")],
        entailment_goal="ForAll(x, Implies(B(x), A(x)))",
        target_description={"A": a, "B": b},
    )]


# ---------------------------------------------------------------------------
# Quantity schemas (determinacy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantitySchema:
    name: str
    trigger: Callable[[str], dict[str, float] | None]
    build: Callable[[dict[str, float]], FormalSpec]


def _rx(pattern: str, text: str, flags: int = re.IGNORECASE) -> re.Match[str] | None:
    return re.search(pattern, text, flags)


def _has_time(text: str) -> bool:
    return bool(_rx(r"\d+\s*(h\b|godz|min|sek|s\b)|w\s+(\w+)\s+godzin", text))


def _schema_specs() -> list[QuantitySchema]:
    S: list[QuantitySchema] = []

    def speed(t: str) -> dict[str, float] | None:
        m = _rx(r"(\d+(?:[.,]\d+)?)\s*km\b", t)
        return {"d": _num(m.group(1))} if m and _rx(r"średni\w*\s+prędko", t) and not _has_time(t) else None

    S.append(QuantitySchema("average_speed", speed, lambda v: _spec("schema.average_speed", {"d": "Real", "t": "Real", "v": "Real"}, [
        ("distance", f"d == {v['d']}", f"droga {v['d']:g} km"), ("time_pos", "t > 0", "czas > 0"), ("def", "v * t == d", "v = d / t")],
        target="v", target_description={"t": "czas przejazdu"})))

    def rect(t: str) -> dict[str, float] | None:
        m = _rx(r"obwód\w*\s+(\d+(?:[.,]\d+)?)", t)
        return {"P": _num(m.group(1))} if m and _rx(r"prostokąt", t) and _rx(r"\bpol[eua]", t) else None

    S.append(QuantitySchema("rectangle_area", rect, lambda v: _spec("schema.rectangle_area", {"a": "Real", "b": "Real"}, [
        ("pos", "And(a > 0, b > 0)", "boki dodatnie"), ("perimeter", f"2 * (a + b) == {v['P']}", f"obwód {v['P']:g}")],
        target="a * b", target_description={"a": "długość jednego boku (lub inne ograniczenie)"})))

    def tri(t: str) -> dict[str, float] | None:
        m = _rx(r"podstaw\w*\s+(\d+(?:[.,]\d+)?)", t)
        return {"a": _num(m.group(1))} if m and _rx(r"trójkąt", t) and _rx(r"\bpol[eua]", t) and not _rx(r"wysoko", t) else None

    S.append(QuantitySchema("triangle_area", tri, lambda v: _spec("schema.triangle_area", {"a": "Real", "h": "Real"}, [
        ("base", f"a == {v['a']}", f"podstawa {v['a']:g}"), ("h_pos", "h > 0", "wysokość > 0")],
        target="a * h / 2", target_description={"h": "wysokość opuszczona na podstawę"})))

    def mean_extreme(t: str) -> dict[str, float] | None:
        k = _rx(r"(\w+)\s+(?:osób|liczb|wartości|uczniów|pomiarów)", t)
        m = _rx(r"(?:średni\w*).*?(?:wynosi|równa się|to)\s+(\d+(?:[.,]\d+)?)", t)
        if not (k and m and _rx(r"najmłodsz|najstarsz|najmniejsz|największ|minim|maksim", t)):
            return None
        n = _word_number(k.group(1))
        return {"k": float(n), "m": _num(m.group(1))} if n and 2 <= n <= 12 else None

    def build_mean(v: dict[str, float]) -> FormalSpec:
        k = int(v["k"])
        xs = [f"x{i}" for i in range(k)]
        asserts = [(f"nonneg_{x}", f"{x} >= 0", f"{x} ≥ 0") for x in xs]
        asserts.append(("mean", f"{' + '.join(xs)} == {k * v['m']}", f"średnia {v['m']:g} z {k} wartości"))
        asserts.append(("min_lower", "And(" + ", ".join(f"t <= {x}" for x in xs) + ")", "t ≤ każdej wartości"))
        asserts.append(("min_attained", "Or(" + ", ".join(f"t == {x}" for x in xs) + ")", "t jest jedną z wartości"))
        return _spec("schema.mean_extreme", {**{x: "Real" for x in xs}, "t": "Real"}, asserts, target="t",
                     target_description={"x0": "indywidualne wartości (średnia nie wyznacza minimum)"})

    S.append(QuantitySchema("mean_extreme", mean_extreme, build_mean))

    def prob_free(t: str) -> dict[str, float] | None:
        if not _rx(r"prawdopodobie", t) or _rx(r"\d", t) or _rx(r"uczciw|symetryczn|jednakowo|równo\s*prawdop", t):
            return None
        return {}

    S.append(QuantitySchema("probability_unspecified", prob_free, lambda v: _spec("schema.probability", {"p": "Real"}, [
        ("kolmogorov", "And(p >= 0, p <= 1)", "0 ≤ P ≤ 1")], target="p",
        target_description={"p": "model losowania (rozkład / skład / własności obiektu)"})))

    def profit(t: str) -> dict[str, float] | None:
        m = _rx(r"przych\w*\s+([\d\s]+(?:[.,]\d+)?)", t)
        return {"R": _num(m.group(1).strip())} if m and _rx(r"\bzysk", t) and not _rx(r"koszt", t) else None

    S.append(QuantitySchema("profit", profit, lambda v: _spec("schema.profit", {"R": "Real", "C": "Real"}, [
        ("revenue", f"R == {v['R']}", f"przychód {v['R']:g}"), ("costs", "C >= 0", "koszty ≥ 0")],
        target="R - C", target_description={"C": "koszty oraz definicja zysku i okres"})))

    def currency(t: str) -> dict[str, float] | None:
        m = _rx(r"kosztuje\s+(\d+(?:[.,]\d+)?)(?!\s*(?:zł|pln|eur|usd|\$|€|dolar|euro|funt))", t)
        return {"N": _num(m.group(1))} if m and _rx(r"złot|\bzł\b|pln", t) else None

    S.append(QuantitySchema("currency", currency, lambda v: _spec("schema.currency", {"rate": "Real"}, [
        ("rate_pos", "rate > 0", "kurs > 0")], target=f"{v['N']} * rate", target_description={"rate": "waluta źródłowa oraz kurs i data przeliczenia"})))

    def energy(t: str) -> dict[str, float] | None:
        m = _rx(r"(\w+)\s+godzin", t)
        hours = _word_number(m.group(1)) if m else None
        return {"t": float(hours)} if hours and _rx(r"energi", t) and not _rx(r"\d+\s*(k?W\b|wat)", t) else None

    S.append(QuantitySchema("energy", energy, lambda v: _spec("schema.energy", {"P": "Real"}, [("power", "P >= 0", "moc ≥ 0")],
        target=f"P * {v['t']}", target_description={"P": "moc pobierana w czasie pracy"})))

    def fuel(t: str) -> dict[str, float] | None:
        m = _rx(r"(\d+(?:[.,]\d+)?)\s*km\b", t)
        return {"D": _num(m.group(1))} if m and _rx(r"paliw", t) and not _rx(r"l\s*/\s*100|spala\w*\s+\d", t) else None

    S.append(QuantitySchema("fuel", fuel, lambda v: _spec("schema.fuel", {"c": "Real"}, [("consumption", "c > 0", "spalanie > 0")],
        target=f"{v['D']} * c / 100", target_description={"c": "średnie spalanie lub stan paliwa przed i po"})))

    def tare(t: str) -> dict[str, float] | None:
        m = _rx(r"z\s+zawartości\w*\s+waży\s+(\d+(?:[.,]\d+)?)", t)
        return {"G": _num(m.group(1))} if m and _rx(r"sam\w*\s+zawarto", t) else None

    S.append(QuantitySchema("tare", tare, lambda v: _spec("schema.tare", {"T": "Real"}, [
        ("tare", f"And(T >= 0, T < {v['G']})", "masa opakowania")], target=f"{v['G']} - T",
        target_description={"T": "masa pustego opakowania"})))

    def survey(t: str) -> dict[str, float] | None:
        m = _rx(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:osób|respondent|ankietowan|uczestnik)", t)
        return {"pct": _num(m.group(1))} if m and _rx(r"ile\s+osób|ilu\s+", t) else None

    S.append(QuantitySchema("survey_count", survey, lambda v: _spec("schema.survey", {"N": "Int", "k": "Int"}, [
        ("sample", "N > 0", "liczebność próby > 0"), ("share", f"100 * k == {int(v['pct'])} * N", f"{v['pct']:g}% próby")],
        target="k", target_description={"N": "liczebność próby (i zasady zaokrąglania)"})))

    def records(t: str) -> dict[str, float] | None:
        m = _rx(r"(\d+(?:[.,]\d+)?)\s*(GB|MB|TB|KB)\b", t, 0)
        return {"B": _num(m.group(1))} if m and _rx(r"rekord|wiersz|próbek|przykład", t) else None

    S.append(QuantitySchema("records", records, lambda v: _spec("schema.records", {"n": "Int", "s": "Real"}, [
        ("n_pos", "n > 0", "liczba rekordów > 0"), ("s_pos", "s > 0", "średni rozmiar rekordu > 0"),
        ("size", f"n * s == {v['B']}", f"łączny rozmiar {v['B']:g}")], target="n",
        target_description={"s": "format, kompresja i rozmiary rekordów"})))

    def training(t: str) -> dict[str, float] | None:
        return {} if _rx(r"trening|trenowani|fine-?tun", t) and _rx(r"ile\s+(godzin|czasu|dni)|jak\s+długo", t) else None

    S.append(QuantitySchema("training_time", training, lambda v: _spec("schema.training_time", {"tok": "Real", "thr": "Real"}, [
        ("tokens", "tok > 0", "liczba tokenów × epok > 0"), ("throughput", "thr > 0", "przepustowość tokenów/s > 0")],
        target="tok / thr", target_description={"tok": "architektura, liczba tokenów i kroków", "thr": "sprzęt i zmierzona przepustowość"})))

    def compare_models(t: str) -> dict[str, float] | None:
        m = _rx(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:trafno|dokładno|accuracy|skuteczno)", t)
        return {"acc": _num(m.group(1)) / 100} if m and _rx(r"lepsz|gorsz|poprzedni", t) else None

    S.append(QuantitySchema("model_comparison", compare_models, lambda v: _spec("schema.comparison", {"new": "Real", "old": "Real"}, [
        ("new", f"new == {v['acc']}", f"nowy model: {v['acc']:.0%}"), ("old_range", "And(old >= 0, old <= 1)", "poprzedni wynik ∈ [0,1]")],
        target="new > old", target_description={"old": "wynik poprzedniego modelu na tym samym zbiorze i protokole"})))

    def timezone(t: str) -> dict[str, float] | None:
        dep = _rx(r"(?:startuje|wylatuje|odlatuje)\s+o\s+(\d{1,2})[:.](\d{2})", t)
        dur = _rx(r"leci\s+(\w+)\s+godzin", t)
        hours = _word_number(dur.group(1)) if dur else None
        if dep and hours and _rx(r"lokaln", t):
            return {"dep": int(dep.group(1)) * 60 + int(dep.group(2)), "dur": hours * 60}
        return None

    S.append(QuantitySchema("timezone", timezone, lambda v: _spec("schema.timezone", {"off": "Int", "arr": "Int"}, [
        ("offset", "And(off >= -720, off <= 840)", "różnica stref czasowych (min)"),
        ("arrival", f"arr == ({int(v['dep'])} + {int(v['dur'])} + off) % 1440", "czas lokalny przylotu")],
        target="arr", target_description={"off": "lotniska wylotu i przylotu oraz data lotu (strefy, zmiana czasu)"})))

    def causal(t: str) -> dict[str, float] | None:
        return {} if _rx(r"spowodował|efekt\w*\s+przyczynow|wpływ\w*\s+na", t) and _rx(r"\bpo\s+(zmianie|wprowadzeniu|wdrożeniu)", t) else None

    S.append(QuantitySchema("causal_effect", causal, lambda v: _spec("schema.causal", {"obs": "Real", "cf": "Real"}, [
        ("observed", "obs > 0", "obserwowany wzrost > 0")], target="obs - cf",
        target_description={"cf": "wynik kontrfaktyczny: grupa kontrolna / test A/B i czynniki zakłócające"})))
    return S


QUANTITY_SCHEMAS = _schema_specs()


def rule_quantity_schemas(query: str) -> list[FormalSpec]:
    specs = []
    for schema in QUANTITY_SCHEMAS:
        values = schema.trigger(query)
        if values is not None:
            specs.append(schema.build(values))
    return specs


_EQUATION = re.compile(r"((?:-?\d+\s*\*?\s*)?[a-z](?:\s*[-+]\s*(?:\d+\s*\*?\s*)?[a-z])*)\s*=\s*(-?\d+(?:[.,]\d+)?)")


def rule_linear_system(query: str) -> list[FormalSpec]:
    target = re.search(r"[Ii]le\s+wynosi\s+([a-z])\b|[Ww]yznacz\s+([a-z])\b", query)
    equations = _EQUATION.findall(query)
    if not (target and equations):
        return []
    tvar = target.group(1) or target.group(2)
    names = sorted(set(re.findall(r"\b[a-z]\b", " ".join(lhs for lhs, _ in equations))) | {tvar})
    sort = "Int" if re.search(r"całkowit|naturaln", query) else "Real"
    asserts = []
    for i, (lhs, rhs) in enumerate(equations):
        expr = re.sub(r"(\d)\s*([a-z])", r"\1*\2", lhs)
        asserts.append((f"eq{i}", f"{expr} == {_num(rhs)}", f"{lhs} = {rhs}"))
    others = [n for n in names if n != tvar]
    return [_spec("schema.linear_system", {n: sort for n in names}, asserts, target=tvar,
                  target_description={others[0] if others else tvar: "dodatkowe niezależne równanie lub wartość pozostałych zmiennych"})]


def rule_sequence(query: str) -> list[FormalSpec]:
    m = re.search(r"[Cc]iąg\w*\s+(?:zaczyna się od|rozpoczyna się od|:)\s*((?:-?\d+\s*,\s*)+-?\d+)", query)
    if not m or not re.search(r"(czwart|piąt|następn|kolejn|\w+ty)\s+wyraz", query):
        return []
    values = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    k = len(values)
    coeffs = [f"c{i}" for i in range(k + 1)]

    def poly(n: int) -> str:
        return " + ".join(f"c{i} * {n ** i}" for i in range(k + 1))

    asserts = [(f"term{n}", f"{poly(n)} == {v}", f"a{n} = {v}") for n, v in enumerate(values, start=1)]
    asserts.append(("next", f"nxt == {poly(k + 1)}", f"a{k + 1} z tej samej reguły"))
    return [_spec("schema.sequence", {**{c: "Real" for c in coeffs}, "nxt": "Real"}, asserts, target="nxt",
                  target_description={"c0": "reguła definiująca ciąg (skończony początek pasuje do wielu reguł)"})]


RULES: tuple[RuleFn, ...] = (
    rule_named_self_reference,
    rule_diagonal,
    rule_numeric_existence,
    rule_unbounded_superlative,
    rule_division,
    rule_fraction_comparison,
    rule_percent_chain,
    rule_probability_bounds,
    rule_parity,
    rule_fair_coin_must,
    rule_ordinal_cardinality,
    rule_polygon_sides,
    rule_empty_container,
    rule_race_overtake,
    rule_count_of_defined_none,
    rule_converse,
    rule_linear_system,
    rule_sequence,
    rule_quantity_schemas,
)


class RuleBasedFormalizer:
    """Applies every rule and returns all formal specs it could build."""

    def __init__(self, rules: tuple[RuleFn, ...] = RULES) -> None:
        self.rules = rules

    def formalize(self, query: str) -> list[FormalSpec]:
        specs: list[FormalSpec] = []
        for rule in self.rules:
            specs.extend(rule(query))
        return specs
