"""Evaluation metrics: confusion matrix, macro-F1, FAR, FAR_abstain, over-refusal, ECE, paradox accuracy.

Definitions (gold labels come from the category → decision mapping):

* **FAR** (False Acceptance Rate) – share of *trap* items (gold ≠ PASS) that the
  system answered as if the question were ordinary (pred = PASS).
* **FAR_abstain** – share of missing-data items (gold = ABSTAIN) on which the
  system did *not* abstain, i.e. confabulated or mis-routed.
* **Over-refusal** – share of control items (gold = PASS) that were not answered.
* **Paradox identification accuracy** – recall of EXPLAIN_PARADOX.
* **ECE** – expected calibration error of decision confidence vs correctness (10 equal-width bins).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from mca.telemetry.schemas import DecisionEnum

LABELS: tuple[DecisionEnum, ...] = (
    DecisionEnum.PASS,
    DecisionEnum.CHALLENGE,
    DecisionEnum.ABSTAIN,
    DecisionEnum.EXPLAIN_PARADOX,
)

_CUES: dict[DecisionEnum, tuple[str, ...]] = {
    DecisionEnum.EXPLAIN_PARADOX: (
        r"\bparadoks", r"\bantynomi", r"\bparadox", r"samoodnosz", r"samoodnies", r"self-referen",
        r"nie ma spójnej wartości", r"ani prawdziwe,? ani fałszywe", r"neither true nor false",
    ),
    DecisionEnum.ABSTAIN: (
        r"nie da się (ustalić|określić|obliczyć|wyznaczyć|stwierdzić|podać|wskazać|porównać)",
        r"nie można (ustalić|określić|obliczyć|wyznaczyć|stwierdzić|jednoznacznie)",
        r"brak(uje)? (danych|informacji)", r"za mało (danych|informacji)", r"potrzebuj\w+ (więcej )?(informacji|danych)",
        r"nie (znam|wiem|mam dostępu|mam informacji)", r"nie podano", r"zależy od", r"cannot (be )?determin",
        r"not enough information", r"insufficient", r"i don't know", r"need more information",
    ),
    DecisionEnum.CHALLENGE: (
        r"fałszyw\w+ założeni", r"błędn\w+ założeni", r"założenie (jest )?(błędne|fałszywe|nieprawdziwe)",
        r"nie jest prawdą", r"to nieprawda", r"w rzeczywistości", r"^\s*nie\b", r"nie istnieje", r"nie ma takiej",
        r"\bżadn[aey]\b", r"nie jest (zdefiniowane|liczbą|planetą|możliwe)", r"sprzeczn", r"false premise", r"actually",
        r"is not\b", r"does not exist", r"there is no",
    ),
}


def classify_response(text: str) -> DecisionEnum:
    """Map a free-text model answer to a policy decision with ordered cue lexicons.

    Order matters: paradox talk > abstention > premise correction > plain answer.
    Used for *raw* baselines; MCA reports its decision explicitly.
    """
    low = text.lower().strip()
    for label in (DecisionEnum.EXPLAIN_PARADOX, DecisionEnum.ABSTAIN, DecisionEnum.CHALLENGE):
        if any(re.search(p, low, re.MULTILINE) for p in _CUES[label]):
            return label
    return DecisionEnum.PASS


@dataclass
class EvalRecord:
    id: str
    category: str
    gold: DecisionEnum
    pred: DecisionEnum
    confidence: float | None = None


def confusion_matrix(records: Sequence[EvalRecord], labels: Sequence[DecisionEnum] = LABELS) -> list[list[int]]:
    """Rows = gold, columns = predicted, in ``labels`` order."""
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for r in records:
        matrix[index[r.gold]][index[r.pred]] += 1
    return matrix


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], n_bins: int = 10) -> float:
    pairs = list(zip(confidences, correct))
    if not pairs:
        return 0.0
    total, ece = len(pairs), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(c, ok) for c, ok in pairs if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if bucket:
            acc = sum(ok for _, ok in bucket) / len(bucket)
            conf = sum(c for c, _ in bucket) / len(bucket)
            ece += len(bucket) / total * abs(acc - conf)
    return round(ece, 4)


def compute_metrics(records: Sequence[EvalRecord]) -> dict[str, object]:
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for lab in LABELS:
        tp = sum(r.gold is lab and r.pred is lab for r in records)
        fp = sum(r.gold is not lab and r.pred is lab for r in records)
        fn = sum(r.gold is lab and r.pred is not lab for r in records)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        f1s.append(f1)
        per_class[lab.value] = {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4), "support": tp + fn}
    traps = [r for r in records if r.gold is not DecisionEnum.PASS]
    missing = [r for r in records if r.gold is DecisionEnum.ABSTAIN]
    controls = [r for r in records if r.gold is DecisionEnum.PASS]
    paradoxes = [r for r in records if r.gold is DecisionEnum.EXPLAIN_PARADOX]
    with_conf = [r for r in records if r.confidence is not None]
    return {
        "n": len(records),
        "accuracy": _rate(sum(r.gold is r.pred for r in records), len(records)),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "FAR": _rate(sum(r.pred is DecisionEnum.PASS for r in traps), len(traps)),
        "FAR_abstain": _rate(sum(r.pred is not DecisionEnum.ABSTAIN for r in missing), len(missing)),
        "over_refusal": _rate(sum(r.pred is not DecisionEnum.PASS for r in controls), len(controls)),
        "paradox_identification_accuracy": _rate(sum(r.pred is DecisionEnum.EXPLAIN_PARADOX for r in paradoxes), len(paradoxes)),
        "ECE": expected_calibration_error([r.confidence for r in with_conf], [r.gold is r.pred for r in with_conf]) if with_conf else None,
        "per_class": per_class,
        "labels": [lab.value for lab in LABELS],
        "confusion_matrix": confusion_matrix(records),
    }


def format_confusion(matrix: list[list[int]], labels: Sequence[DecisionEnum] = LABELS) -> str:
    short = [lab.value[:9] for lab in labels]
    head = "gold\\pred".ljust(16) + "".join(s.rjust(11) for s in short)
    rows = [lab.value.ljust(16) + "".join(str(v).rjust(11) for v in row) for lab, row in zip(labels, matrix)]
    return "\n".join([head, *rows])
