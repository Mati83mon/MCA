"""Re-apply the current Decision Controller to stored MCA telemetry.

The policy gate is a pure function of :class:`MetacognitiveTelemetry` and :class:`MCAConfig`, so a
rule change can be evaluated exactly on saved runs without re-running any model::

    python -m benchmarks.rescore results/kaggle-2xT4 --out results/kaggle-2xT4/rescored_r8b.csv

This is exact for rules that only change the outcome of rows that reached the slow path (such as R8b):
rows whose fast path ended in a non-PASS decision were saved without entropy samples, but an earlier rule
already decides them, so they come out unchanged.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from benchmarks.runner import summarize
from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.telemetry.schemas import MetacognitiveTelemetry

METRICS = ("accuracy", "macro_f1", "FAR", "FAR_abstain", "over_refusal", "paradox_identification_accuracy", "ECE")


def rescore(src: str, config: MCAConfig | None = None) -> dict[str, tuple[dict, dict, list[dict]]]:
    ctl = DecisionController(config or MCAConfig(nli_backend="lexical", embedding_backend="hashing"))
    out: dict[str, tuple[dict, dict, list[dict]]] = {}
    for f in sorted(glob.glob(str(Path(src) / "mca__*.jsonl"))):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        new, flips = [], []
        for r in rows:
            d = ctl.decide(MetacognitiveTelemetry.model_validate(r["telemetry"]))
            n = dict(r, pred=d.decision.value, confidence=d.confidence, rules=d.triggered_rules, response=d.response_text or r["response"])
            if n["pred"] != r["pred"]:
                flips.append({"id": r["id"], "gold": r["gold"], "old": r["pred"], "new": n["pred"], "rule": d.triggered_rules[0]})
            new.append(n)
        out[Path(f).stem.removeprefix("mca__")] = (summarize(rows), summarize(new), flips)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="directory with mca__*.jsonl files")
    p.add_argument("--out", default=None, help="optional CSV with stored vs. rescored metrics")
    args = p.parse_args(argv)
    res = rescore(args.src)
    rows = []
    for model, (old, new, flips) in res.items():
        print(f"\n{model}")
        for k in METRICS:
            print(f"  {k:32s} {old[k]!s:>8} -> {new[k]}")
        for fl in flips:
            print(f"   {fl['id']}: {fl['old']} -> {fl['new']} ({fl['rule']}, gold {fl['gold']})")
        rows += [{"run": f"stored/{model}", **{k: old[k] for k in METRICS}}, {"run": f"rescored/{model}", **{k: new[k] for k in METRICS}}]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["run", *METRICS])
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
