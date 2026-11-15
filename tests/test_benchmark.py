"""End-to-end offline evaluation + metric definitions."""

from __future__ import annotations

import asyncio

from benchmarks.metrics import EvalRecord, classify_response, compute_metrics, expected_calibration_error
from benchmarks.runner import DEFAULT_DATASET, load_questions, summarize, _evaluate_mca
from mca.telemetry.schemas import DecisionEnum as D

HELDOUT = DEFAULT_DATASET.parent / "heldout_probes.json"


def test_metrics_definitions():
    recs = [
        EvalRecord("1", "x", D.PASS, D.PASS, 0.9), EvalRecord("2", "x", D.CHALLENGE, D.PASS, 0.8),
        EvalRecord("3", "x", D.ABSTAIN, D.CHALLENGE, 0.6), EvalRecord("4", "x", D.EXPLAIN_PARADOX, D.EXPLAIN_PARADOX, 0.9),
    ]
    m = compute_metrics(recs)
    assert m["accuracy"] == 0.5 and m["FAR"] == round(1 / 3, 4) and m["FAR_abstain"] == 1.0
    assert m["paradox_identification_accuracy"] == 1.0 and m["over_refusal"] == 0.0
    assert m["confusion_matrix"][1][0] == 1
    assert expected_calibration_error([1.0, 1.0], [True, False]) == 0.5


def test_response_classifier():
    assert classify_response("To klasyczny paradoks kłamcy.") is D.EXPLAIN_PARADOX
    assert classify_response("Nie da się ustalić tego bez czasu przejazdu.") is D.ABSTAIN
    assert classify_response("Nie, Mars jest czwartą planetą.") is D.CHALLENGE
    assert classify_response("Wenus.") is D.PASS


def test_offline_pipeline_on_benchmark(orchestrator):
    rows = asyncio.run(_evaluate_mca(orchestrator, load_questions(), 8))
    metrics = summarize(rows)
    # The bundled KB and rule templates cover the benchmark's topics, so this is an
    # in-distribution regression check, not a generalisation estimate.
    assert metrics["accuracy"] >= 0.95
    assert metrics["FAR"] <= 0.05


def test_offline_pipeline_on_heldout_probes(orchestrator):
    rows = asyncio.run(_evaluate_mca(orchestrator, load_questions(HELDOUT), 8))
    metrics = summarize(rows)
    # Held-out probes were written after the pipeline was frozen; the floor documents
    # current offline generalisation (see README) and guards against regressions.
    assert metrics["accuracy"] >= 0.6
