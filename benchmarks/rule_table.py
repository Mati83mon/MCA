"""Per-rule audit table and offline threshold calibration on stored MCA telemetry.

The ordered policy gate only logs the first rule that fires, so a threshold tuned on end-to-end logs is
tuned on a censored sample. This module exposes every rule's own verdict per item
(:meth:`DecisionController.rule_verdicts`) together with the raw signals behind it. It also re-runs
the (pure) controller over a threshold grid, so calibration needs no new model calls::

    # 1. per-item table (every rule, every item)
    python -m benchmarks.rule_table table results/audit/mca__Qwen2.5-7B-Instruct.jsonl --out table.csv
    # 2. tune on the benchmark, then evaluate the chosen config exactly once on held-out items
    python -m benchmarks.rule_table calibrate results/audit/mca__X.jsonl --heldout results/audit/heldout__X.jsonl

Uncensored R8/R8b verdicts need telemetry recorded with ``MCAConfig(audit_all_layers=True)``; rows
without entropy samples report ``None`` for those rules.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from typing import Any

from benchmarks.runner import summarize
from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.telemetry.schemas import MetacognitiveTelemetry, NLILabel, PremiseKind, PremiseStatus

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
SIMS = (0.5, 0.6, 0.7, 0.8)
ENTS = (0.8, 0.9, 0.95)


def _load(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _base_config(**kw: Any) -> MCAConfig:
    return MCAConfig(nli_backend="lexical", embedding_backend="hashing", **kw)


def signals(t: MetacognitiveTelemetry) -> dict[str, float | None]:
    pres = [v for v in t.premise_verdicts if v.premise.kind is PremiseKind.PRESUPPOSITION]
    supported = [v for v in t.premise_verdicts
                 if v.status is PremiseStatus.SUPPORTED and v.premise.kind is not PremiseKind.REFERENT and v.evidence]
    u = t.uncertainty
    return {
        "max_contradiction": max((v.nli_scores.get(NLILabel.CONTRADICTION, 0.0) for v in pres), default=0.0),
        "normalized_entropy": u.normalized_entropy if u is not None and u.n_samples > 1 else None,
        "best_evidence_similarity": max((v.evidence[0].score for v in supported), default=None),
        "best_evidence_entailment": max((v.nli_scores.get(NLILabel.ENTAILMENT, 0.0) for v in supported), default=None),
    }


def table(rows: list[dict[str, Any]], config: MCAConfig | None = None) -> list[dict[str, Any]]:
    ctl = DecisionController(config or _base_config())
    out = []
    for r in rows:
        t = MetacognitiveTelemetry.model_validate(r["telemetry"])
        verdicts = ctl.rule_verdicts(t)
        d = ctl.decide(t)
        out.append({"id": r["id"], "category": r["category"], "gold": r["gold"], "decision": d.decision.value,
                    "fired": d.triggered_rules[0], **verdicts, **signals(t)})
    return out


def _evaluate(rows: list[dict[str, Any]], config: MCAConfig) -> dict[str, Any]:
    ctl = DecisionController(config)
    new = []
    for r in rows:
        d = ctl.decide(MetacognitiveTelemetry.model_validate(r["telemetry"]))
        new.append(dict(r, pred=d.decision.value, confidence=d.confidence))
    return summarize(new)


def calibrate(rows: list[dict[str, Any]], heldout: list[dict[str, Any]] | None = None, far_budget: float = 0.0) -> dict[str, Any]:
    """Grid-search R8/R8b thresholds on ``rows``; pick max accuracy with FAR ≤ budget; test once on ``heldout``."""
    grid = []
    for tau, sim, ent in itertools.product(TAUS, SIMS, ENTS):
        cfg = _base_config(entropy_tau=tau, evidence_answer_min_similarity=sim, evidence_answer_min_entailment=ent)
        m = _evaluate(rows, cfg)
        grid.append({"entropy_tau": tau, "min_similarity": sim, "min_entailment": ent, **m})
    feasible = [g for g in grid if g["FAR"] <= far_budget] or grid
    best = max(feasible, key=lambda g: (g["accuracy"], -g["over_refusal"], -g["entropy_tau"]))
    result: dict[str, Any] = {"grid": grid, "chosen": best}
    if heldout is not None:
        cfg = _base_config(entropy_tau=best["entropy_tau"], evidence_answer_min_similarity=best["min_similarity"],
                           evidence_answer_min_entailment=best["min_entailment"])
        result["heldout_chosen"] = _evaluate(heldout, cfg)
        result["heldout_default"] = _evaluate(heldout, _base_config())
    return result


STAGE_TAUS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 … 0.95
EARLY_RULES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


def at_entropy_stage(t: MetacognitiveTelemetry, ctl: DecisionController) -> bool:
    """True when no rule before R8 fires, i.e. the item is decided by R8b/R8/R9 alone."""
    verdicts = ctl.rule_verdicts(t)
    return not any(verdicts[r] for r in EARLY_RULES)


def calibrate_stage(tune: list[dict[str, Any]], test: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Stage-restricted calibration of τ (fixed before any audit result was seen).

    1. Stage set = items of ``tune`` on which R1–R7 all stay silent under the default config.
    2. For each τ in :data:`STAGE_TAUS`: trap catch rate (decision ≠ PASS on stage traps) and control
       pass rate (decision = PASS on stage controls); objective = their mean (balanced accuracy).
    3. Pick the best τ; ties go to the higher τ (fewer refusals). R8b thresholds stay at their defaults.
    4. Evaluate the chosen τ and the default τ exactly once on ``test`` with the full pipeline.
    """
    ctl0 = DecisionController(_base_config())
    stage = [r for r in tune if at_entropy_stage(MetacognitiveTelemetry.model_validate(r["telemetry"]), ctl0)]
    traps = [r for r in stage if r["gold"] != "PASS"]
    ctrls = [r for r in stage if r["gold"] == "PASS"]
    curve = []
    for tau in STAGE_TAUS:
        ctl = DecisionController(_base_config(entropy_tau=tau))
        dec = {r["id"]: ctl.decide(MetacognitiveTelemetry.model_validate(r["telemetry"])).decision.value for r in stage}
        catch = sum(dec[r["id"]] != "PASS" for r in traps) / len(traps) if traps else float("nan")
        passed = sum(dec[r["id"]] == "PASS" for r in ctrls) / len(ctrls) if ctrls else float("nan")
        curve.append({"entropy_tau": tau, "trap_catch": round(catch, 4), "control_pass": round(passed, 4),
                      "balanced": round((catch + passed) / 2, 4)})
    usable = [c for c in curve if c["balanced"] == c["balanced"]]  # drop NaN
    chosen = max(usable, key=lambda c: (c["balanced"], c["entropy_tau"])) if traps and ctrls else None
    default_tau = _base_config().entropy_tau
    result: dict[str, Any] = {"n_tune": len(tune), "n_stage_traps": len(traps), "n_stage_controls": len(ctrls),
                              "curve": curve, "chosen": chosen, "default_tau": default_tau,
                              "default_on_stage": next((c for c in curve if abs(c["entropy_tau"] - default_tau) < 1e-9), None)}
    if test is not None and chosen is not None:
        result["test_chosen"] = _evaluate(test, _base_config(entropy_tau=chosen["entropy_tau"]))
        result["test_default"] = _evaluate(test, _base_config())
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("table")
    pt.add_argument("jsonl")
    pt.add_argument("--out", required=True)
    pc = sub.add_parser("calibrate")
    pc.add_argument("jsonl")
    pc.add_argument("--heldout", default=None)
    pc.add_argument("--far-budget", type=float, default=0.0)
    pc.add_argument("--out", default=None, help="optional JSON with the full grid")
    ps = sub.add_parser("calibrate-stage", help="stage-restricted τ calibration on rebuilt traps")
    ps.add_argument("jsonl", nargs="+", help="tuning telemetry (rebuilt traps + their controls)")
    ps.add_argument("--test", nargs="*", default=None, help="untouched held-out telemetry, evaluated once")
    ps.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.cmd == "calibrate-stage":
        tune = [r for f in args.jsonl for r in _load(f)]
        test = [r for f in args.test for r in _load(f)] if args.test else None
        res = calibrate_stage(tune, test)
        print(f"stage set: {res['n_stage_traps']} traps, {res['n_stage_controls']} controls (of {res['n_tune']} tuning items)")
        for c in res["curve"]:
            print(f"  τ={c['entropy_tau']:.2f}  catch={c['trap_catch']:.2f}  pass={c['control_pass']:.2f}  balanced={c['balanced']:.2f}")
        print("chosen:", res["chosen"], "| default τ:", res["default_tau"])
        keys = ("accuracy", "FAR", "over_refusal")
        if "test_chosen" in res:
            print("test, chosen :", {k: res["test_chosen"][k] for k in keys})
            print("test, default:", {k: res["test_default"][k] for k in keys})
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        return 0

    if args.cmd == "table":
        tab = table(_load(args.jsonl))
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tab[0]))
            w.writeheader()
            w.writerows(tab)
        observed = sum(row["R8"] is not None for row in tab)
        print(f"{len(tab)} items, entropy observed on {observed}")
        for rule in DecisionController.RULE_ORDER:
            fires = [row for row in tab if row[rule]]
            print(f"  {rule:4s} fires on {len(fires):3d} items "
                  f"(control facts: {sum(row['gold'] == 'PASS' for row in fires)})")
        return 0

    res = calibrate(_load(args.jsonl), _load(args.heldout) if args.heldout else None, args.far_budget)
    keys = ("accuracy", "FAR", "over_refusal")
    c = res["chosen"]
    print("chosen on benchmark:", {k: c[k] for k in ("entropy_tau", "min_similarity", "min_entailment", *keys)})
    if "heldout_chosen" in res:
        print("held-out, chosen   :", {k: res["heldout_chosen"][k] for k in keys})
        print("held-out, default  :", {k: res["heldout_default"][k] for k in keys})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
