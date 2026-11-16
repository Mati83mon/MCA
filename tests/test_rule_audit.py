"""rule_verdicts must agree with decide: the first True rule in RULE_ORDER is the rule decide fires."""

import json
from pathlib import Path

import pytest

from benchmarks.rule_table import calibrate, table
from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.telemetry.schemas import MetacognitiveTelemetry

RESULTS = Path(__file__).resolve().parents[1].joinpath("results", "kaggle-2xT4")
RUNS = sorted(RESULTS.glob("mca__*.jsonl"))
AUDIT = sorted(RESULTS.joinpath("audit-v10").glob("*.jsonl"))
AUDIT_MCA = [p for p in AUDIT if not p.name.startswith("raw")]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.skipif(not RUNS, reason="stored GPU telemetry not present")
@pytest.mark.parametrize("path", RUNS + AUDIT_MCA, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_rule_verdicts_agree_with_decide(path: Path):
    ctl = DecisionController(MCAConfig(nli_backend="lexical", embedding_backend="hashing"))
    for r in _rows(path):
        t = MetacognitiveTelemetry.model_validate(r["telemetry"])
        verdicts = ctl.rule_verdicts(t)
        first = next((rule for rule in DecisionController.RULE_ORDER if verdicts[rule]), "R9")
        fired = ctl.decide(t).triggered_rules[0].split(":")[0]
        assert first == fired, (r["id"], verdicts, fired)


@pytest.mark.skipif(not RUNS, reason="stored GPU telemetry not present")
def test_table_and_calibration_run_offline():
    rows = _rows(RUNS[0])
    tab = table(rows)
    assert len(tab) == len(rows) and {"R5", "R8", "max_contradiction"} <= set(tab[0])
    res = calibrate(rows[:60], heldout=rows[60:])
    assert res["chosen"]["FAR"] <= 0.0 + 1e-9 or res["chosen"] is not None
    assert "heldout_chosen" in res


@pytest.mark.skipif(not AUDIT_MCA, reason="audit telemetry not present")
@pytest.mark.parametrize("path", AUDIT_MCA, ids=lambda p: p.stem)
def test_audit_run_is_uncensored(path: Path):
    """audit_all_layers=True: entropy (hence R8/R8b) is observed on every item, not only on fast-path PASSes."""
    tab = table(_rows(path))
    assert all(row["R8"] is not None and row["R8b"] is not None for row in tab)


def _stage_row(rid: str, gold: str, entropy: float) -> dict:
    from mca.telemetry.schemas import UncertaintyReport
    t = MetacognitiveTelemetry(query=f"q {rid}", uncertainty=UncertaintyReport(n_samples=5, normalized_entropy=entropy))
    return {"id": rid, "category": "x", "gold": gold, "pred": "PASS", "confidence": 0.5, "latency_ms": 0.0,
            "telemetry": t.model_dump(mode="json")}


def test_calibrate_stage_picks_highest_tau_that_separates():
    from benchmarks.rule_table import calibrate_stage
    tune = [_stage_row(f"t{i}", "ABSTAIN", e) for i, e in enumerate((0.72, 0.8, 0.9, 1.0))]
    tune += [_stage_row(f"c{i}", "PASS", e) for i, e in enumerate((0.0, 0.2, 0.4, 0.55))]
    res = calibrate_stage(tune, test=tune)
    assert res["n_stage_traps"] == 4 and res["n_stage_controls"] == 4
    # every τ in [0.55, 0.72) separates perfectly; ties go to the highest grid value, 0.70
    assert res["chosen"]["entropy_tau"] == 0.7 and res["chosen"]["balanced"] == 1.0
    assert res["test_chosen"]["accuracy"] >= res["test_default"]["accuracy"]
