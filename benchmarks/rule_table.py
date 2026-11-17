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
        "answerable": t.answerability.answerable if t.answerability is not None else None,
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


def calibrate_stage(tune: list[dict[str, Any]], test: list[dict[str, Any]] | None = None, **fixed: Any) -> dict[str, Any]:
    """Stage-restricted calibration of τ (fixed before any audit result was seen).

    1. Stage set = items of ``tune`` on which R1–R7 all stay silent under the default config.
    2. For each τ in :data:`STAGE_TAUS`: trap catch rate (decision ≠ PASS on stage traps) and control
       pass rate (decision = PASS on stage controls); objective = their mean (balanced accuracy).
    3. Pick the best τ; ties go to the higher τ (fewer refusals). R8b thresholds stay at their defaults.
    4. Evaluate the chosen τ and the default τ exactly once on ``test`` with the full pipeline.

    ``fixed`` holds config fields kept constant throughout (e.g. ``r7_llm_referents="off"``).
    """
    ctl0 = DecisionController(_base_config(**fixed))
    stage = [r for r in tune if at_entropy_stage(MetacognitiveTelemetry.model_validate(r["telemetry"]), ctl0)]
    traps = [r for r in stage if r["gold"] != "PASS"]
    ctrls = [r for r in stage if r["gold"] == "PASS"]
    curve = []
    for tau in STAGE_TAUS:
        ctl = DecisionController(_base_config(entropy_tau=tau, **fixed))
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
        result["test_chosen"] = _evaluate(test, _base_config(entropy_tau=chosen["entropy_tau"], **fixed))
        result["test_default"] = _evaluate(test, _base_config(**fixed))
    result["fixed"] = fixed
    return result


R7_MODES = ("all", "gated", "specific", "off")  # tie order: status quo first


def _balanced(m: dict[str, Any]) -> float:
    return round(1.0 - (m["FAR"] + m["over_refusal"]) / 2, 4)


def select_r7_mode(dev: list[dict[str, Any]], test: list[dict[str, Any]] | None = None,
                   far_budget: float | None = None) -> dict[str, Any]:
    """Pick the R7 mode on ``dev``, then evaluate it once on ``test`` (fixed before the answerability run).

    1. τ stays at its default; for each mode in :data:`R7_MODES` compute FAR and over-refusal on ``dev``.
    2. Objective: balanced = 1 − (FAR + over-refusal) / 2. Ties go to the earlier mode (``all`` first).
       With ``far_budget`` (v14 protocol) the objective is instead: lowest over-refusal among modes whose dev FAR
       is at most FAR(``all``) + ``far_budget``; ties again go to the earlier mode.
    3. Evaluate on ``test`` exactly once. Primary comparison: the chosen mode vs ``all``; the other modes
       are reported as secondary results.
    Also reports how the answerability probe labelled each gold category.
    """
    keys = ("accuracy", "FAR", "over_refusal")
    curve = []
    for mode in R7_MODES:
        m = _evaluate(dev, _base_config(r7_llm_referents=mode))
        curve.append({"mode": mode, **{k: m[k] for k in keys}, "balanced": _balanced(m)})
    if far_budget is None:
        chosen = max(curve, key=lambda c: (c["balanced"], -R7_MODES.index(c["mode"])))
    else:
        cap = curve[0]["FAR"] + far_budget + 1e-9
        chosen = min((c for c in curve if c["FAR"] <= cap), key=lambda c: (c["over_refusal"], R7_MODES.index(c["mode"])))
    probe: dict[str, dict[str, int]] = {}
    for r in dev + (test or []):
        a = (r["telemetry"].get("answerability") or {}).get("answerable")
        probe.setdefault(r["category"], {"TAK": 0, "NIE": 0, "brak": 0})[{True: "TAK", False: "NIE"}.get(a, "brak")] += 1
    result: dict[str, Any] = {"n_dev": len(dev), "curve": curve, "chosen": chosen, "probe_by_category": probe,
                              "far_budget": far_budget}
    if test is not None:
        for mode in R7_MODES:
            m = _evaluate(test, _base_config(r7_llm_referents=mode))
            result[f"test_{mode}"] = {**{k: m[k] for k in keys}, "balanced": _balanced(m)}
    return result


PROBE_ORDER = ("direct", "fewshot", "missing", "inverse")


def apply_probe(rows: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of ``rows`` whose ``telemetry.answerability`` is replaced by ``verdicts[id]`` (``None`` if absent)."""
    return [dict(r, telemetry=dict(r["telemetry"], answerability=verdicts.get(r["id"]))) for r in rows]


def probe_score(dev: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Balanced probe accuracy on ``dev``: NIE on missing-data items and TAK on control facts (no reply = wrong)."""
    def rate(cat: str, want: bool) -> float:
        ids = [r["id"] for r in dev if r["category"] == cat]
        hits = sum((verdicts.get(i) or {}).get("answerable") is want for i in ids)
        return round(hits / len(ids), 4) if ids else float("nan")
    nie, tak = rate("brak_danych", False), rate("normalny_fakt", True)
    return {"nie_on_missing_data": nie, "tak_on_facts": tak, "balanced": round((nie + tak) / 2, 4)}


def select_probe_and_r7(dev: list[dict[str, Any]], test: list[dict[str, Any]],
                        probes: dict[str, dict[str, dict[str, Any]]], far_budget: float = 0.05) -> dict[str, Any]:
    """v14 protocol (fixed before the run).

    1. Probe: among ``probes`` (candidate name -> {item id -> verdict}), pick the best :func:`probe_score` on
       ``dev``; ties go to the earlier candidate (own ``direct`` first).
    2. R7 mode: :func:`select_r7_mode` on ``dev`` with the chosen probe's verdicts substituted, using the FAR
       budget objective (lowest over-refusal with dev FAR ≤ FAR(``all``) + 0.05).
    3. Test once on ``test`` with the same verdicts: chosen mode vs ``all`` (other modes secondary).
    """
    scores = {name: probe_score(dev, v) for name, v in probes.items()}
    order = list(probes)
    best = max(order, key=lambda n: (scores[n]["balanced"], -order.index(n)))
    res = select_r7_mode(apply_probe(dev, probes[best]), apply_probe(test, probes[best]), far_budget=far_budget)
    return {"probe_scores": scores, "probe_chosen": best,
            "probe_on_test": probe_score(test, probes[best]), **res}


def load_probes(path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """probes JSONL rows {id, variant, answerable, raw} -> {variant -> {id -> verdict}}."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in _load(path):
        out.setdefault(r["variant"], {})[r["id"]] = {"answerable": r["answerable"], "raw": r.get("raw", ""), "variant": r["variant"]}
    return out


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
    ps.add_argument("--r7-llm-referents", choices=("all", "specific", "off", "gated"), default="all")
    ps.add_argument("--out", default=None)
    pr = sub.add_parser("select-r7", help="pick the R7 mode on dev telemetry, test once on fresh held-out")
    pr.add_argument("jsonl", nargs="+", help="dev telemetry (already-seen sets)")
    pr.add_argument("--test", nargs="*", default=None, help="fresh held-out telemetry, evaluated once")
    pr.add_argument("--out", default=None)
    pp = sub.add_parser("select-probe-r7", help="v14: pick probe wording, then R7 mode, on dev; test once")
    pp.add_argument("jsonl", nargs="+", help="dev telemetry (already-seen sets)")
    pp.add_argument("--test", nargs="+", required=True, help="fresh held-out telemetry, evaluated once")
    pp.add_argument("--probes", required=True, help="this model's probe JSONL (all variants)")
    pp.add_argument("--judge-probes", default=None, help="another model's probe JSONL, offered as external judge")
    pp.add_argument("--judge-name", default="judge")
    pp.add_argument("--far-budget", type=float, default=0.05)
    pp.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.cmd == "select-probe-r7":
        dev = [r for f in args.jsonl for r in _load(f)]
        test = [r for f in args.test for r in _load(f)]
        own = load_probes(args.probes)
        cands = {f"own:{v}": own[v] for v in PROBE_ORDER if v in own}
        if args.judge_probes:
            judge = load_probes(args.judge_probes)
            cands.update({f"{args.judge_name}:{v}": judge[v] for v in PROBE_ORDER if v in judge})
        res = select_probe_and_r7(dev, test, cands, args.far_budget)
        for name, sc in res["probe_scores"].items():
            print(f"  probe {name:22s} NIE|missing={sc['nie_on_missing_data']:.2f}  TAK|facts={sc['tak_on_facts']:.2f}  balanced={sc['balanced']:.3f}")
        print("probe chosen:", res["probe_chosen"], "| on test:", res["probe_on_test"])
        for c in res["curve"]:
            print(f"  dev {c['mode']:8s} FAR={c['FAR']:.3f}  over-refusal={c['over_refusal']:.3f}  balanced={c['balanced']:.3f}")
        print("mode chosen:", res["chosen"]["mode"])
        for k, v in res.items():
            if k.startswith("test_"):
                print(f"{k}: FAR={v['FAR']:.3f}  over-refusal={v['over_refusal']:.3f}  balanced={v['balanced']:.3f}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        return 0

    if args.cmd == "select-r7":
        dev = [r for f in args.jsonl for r in _load(f)]
        test = [r for f in args.test for r in _load(f)] if args.test else None
        res = select_r7_mode(dev, test)
        print(f"dev: {res['n_dev']} items")
        for c in res["curve"]:
            print(f"  {c['mode']:8s} FAR={c['FAR']:.3f}  over-refusal={c['over_refusal']:.3f}  balanced={c['balanced']:.3f}")
        print("chosen:", res["chosen"]["mode"])
        print("probe (TAK/NIE/brak) by category:", res["probe_by_category"])
        for k, v in res.items():
            if k.startswith("test_"):
                print(f"{k}: FAR={v['FAR']:.3f}  over-refusal={v['over_refusal']:.3f}  balanced={v['balanced']:.3f}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        return 0

    if args.cmd == "calibrate-stage":
        tune = [r for f in args.jsonl for r in _load(f)]
        test = [r for f in args.test for r in _load(f)] if args.test else None
        res = calibrate_stage(tune, test, r7_llm_referents=args.r7_llm_referents)
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
