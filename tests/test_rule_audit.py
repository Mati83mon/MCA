"""rule_verdicts must agree with decide: the first True rule in RULE_ORDER is the rule decide fires."""

import json
from pathlib import Path

import pytest

from benchmarks.rule_table import calibrate, table
from mca.config import MCAConfig
from mca.core.decision_controller import DecisionController
from mca.telemetry.schemas import MetacognitiveTelemetry

RUNS = sorted(Path(__file__).resolve().parents[1].joinpath("results", "kaggle-2xT4").glob("mca__*.jsonl"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.skipif(not RUNS, reason="stored GPU telemetry not present")
@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
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
