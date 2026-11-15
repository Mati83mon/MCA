"""Lightweight, dependency-free text utilities tuned for Polish (and English) input.

These helpers back the offline lexical NLI backend, the hashing embedder and the
rule-based formalizer. They intentionally avoid heavy NLP stacks so the MCA core
runs on CPU-only machines and in CI.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+", re.UNICODE)

STOPWORDS: frozenset[str] = frozenset(
    """
    a aby ale albo ani bo by być był była było czy do dla gdy go i ich ile im
    jak jaka jaki jakie jako je jego jej jest już lub ma mi mnie na nad nas o od
    oraz po pod przez przy się są ta tak tam te tego tej ten to tu tylko tym w we
    więc z za ze że który która które którego której którą których jeśli jeżeli
    co kto czym czego dlaczego czemu
    the of is are be to in on at and or an as it its this that what which why how
    """.split()
)

NEGATIONS: frozenset[str] = frozenset({"nie", "nigdy", "brak", "żaden", "żadna", "żadne", "żadnego", "not", "no", "never", "none"})

ORDINALS: dict[str, int] = {
    "pierwsz": 1, "drug": 2, "trzec": 3, "czwart": 4, "piąt": 5, "szóst": 6,
    "siódm": 7, "ósm": 8, "dziewiąt": 9, "dziesiąt": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
}

CARDINALS: dict[str, int] = {
    "zero": 0, "zera": 0, "jeden": 1, "jedna": 1, "jedno": 1, "jednego": 1, "jednej": 1,
    "dwa": 2, "dwie": 2, "dwóch": 2, "dwu": 2, "dwoma": 2,
    "trzy": 3, "trzech": 3, "cztery": 4, "czterech": 4, "pięć": 5, "pięciu": 5,
    "sześć": 6, "sześciu": 6, "siedem": 7, "siedmiu": 7, "osiem": 8, "ośmiu": 8,
    "dziewięć": 9, "dziewięciu": 9, "dziesięć": 10, "dziesięciu": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
}

# Groups of mutually exclusive concepts (stems). Two claims that agree on most
# content but pick different members of the same group conflict.
EXCLUSIVE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"masy", "masa", "długoś", "czasu", "czas", "temper", "natęże", "świetl", "światło", "liczno"}),
    frozenset({"planet", "gwiazd", "księży"}),
    frozenset({"wielki", "wielką", "mały", "małymi", "małą"}),
    frozenset({"większ", "mniejs"}),
    frozenset({"prawdz", "fałszy"}),
    frozenset({"true", "false"}),
)

_STEM_LEN = 6


def strip_accents(text: str) -> str:
    """Remove diacritics (used only for fuzzy keys, never for display)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Lower-case word/number tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=65536)
def stem(token: str) -> str:
    """Crude prefix stemmer, adequate for Polish inflection in short factual claims."""
    return token if len(token) <= _STEM_LEN else token[:_STEM_LEN]


def split_negated_prefix(token: str) -> tuple[bool, str]:
    """Split Polish ``nie-`` prefixed adjectives (``niemodyfikowalna`` -> (True, ``modyfikowalna``))."""
    if token.startswith("nie") and len(token) > 6 and token not in NEGATIONS:
        return True, token[3:]
    return False, token


def ordinal_value(token: str) -> int | None:
    for prefix, value in ORDINALS.items():
        if token.startswith(prefix):
            return value
    return None


def cardinal_value(token: str) -> int | None:
    if token in CARDINALS:
        return CARDINALS[token]
    if re.fullmatch(r"\d+", token):
        return int(token)
    return None


def content_profile(text: str) -> tuple[frozenset[str], int, frozenset[str]]:
    """Return (content stems, negation parity, numeric/ordinal values) for ``text``."""
    stems: set[str] = set()
    numbers: set[str] = set()
    negations = 0
    for tok in tokenize(text):
        if tok in NEGATIONS:
            negations += 1
            continue
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", tok):
            numbers.add(tok.replace(",", "."))
            continue
        ordv = ordinal_value(tok)
        if ordv is not None:
            numbers.add(f"#{ordv}")
            continue
        if tok in STOPWORDS:
            continue
        negated, base = split_negated_prefix(tok)
        if negated:
            negations += 1
        stems.add(stem(base))
    return frozenset(stems), negations % 2, frozenset(numbers)


def split_clauses(text: str) -> list[str]:
    """Split a passage into clause-sized units for clause-level NLI."""
    parts = re.split(r"(?<=[.;!?])\s+|;\s*|\s+[–—-]\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]
