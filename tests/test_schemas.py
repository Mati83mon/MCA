from __future__ import annotations

import pytest
from pydantic import ValidationError

from mca.telemetry.schemas import BenchmarkDataset, Category, DecisionEnum, QuestionItem


def test_dataset_validates(dataset_raw):
    ds = BenchmarkDataset.model_validate(dataset_raw)
    assert len(ds.pytania) == 100
    assert {q.kategoria for q in ds.pytania} == set(Category)


def test_category_mapping():
    assert Category.PARADOKS.expected_decision is DecisionEnum.EXPLAIN_PARADOX
    assert Category.BRAK_DANYCH.expected_decision is DecisionEnum.ABSTAIN


def _item(**kw):
    base = dict(id="Q001", kategoria="normalny_fakt", temat="t", pytanie="Ile?", odpowiedz_wzorcowa="a", wyjasnienie="w")
    base.update(kw)
    return base


@pytest.mark.parametrize("bad", [
    {"id": "X1"},
    {"kategoria": "brak_danych"},
    {"brakujace_dane": ["x"]},
    {"zrodla": ["ftp://x"]},
    {"unknown_field": 1},
])
def test_item_validation_errors(bad):
    with pytest.raises(ValidationError):
        QuestionItem.model_validate(_item(**bad))


def test_dataset_count_mismatch(dataset_raw):
    broken = {**dataset_raw, "pytania": dataset_raw["pytania"][:-1]}
    with pytest.raises(ValidationError):
        BenchmarkDataset.model_validate(broken)
