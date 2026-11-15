from __future__ import annotations

import datetime as dt
import json

from scripts.ingest_daily_json import Corpus, ingest

from tests.conftest import DATASET


def _item(i, q):
    return {"id": f"Q{i:03d}", "kategoria": "normalny_fakt", "temat": "t", "pytanie": q, "odpowiedz_wzorcowa": "a", "wyjasnienie": "w"}


def test_ingest_appends_dedupes_renumbers_and_rejects(tmp_path):
    inbox, root = tmp_path / "inbox", tmp_path / "corpus"
    inbox.mkdir()
    (inbox / "a.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
        _item(1, "Jaka jest stolica Polski?"),                       # id collides -> renumbered
        _item(500, "Która planeta jest najbliżej Słońca?"),          # duplicate of Q077
    ]), encoding="utf-8")
    (inbox / "bad.json").write_text(json.dumps([{"id": "nope"}]), encoding="utf-8")
    summary = ingest(inbox, root, seed=DATASET, date=dt.date(2026, 11, 15))
    assert summary["added"] == 1 and summary["duplicates"] == 1 and summary["rejected"] == ["bad.json"]
    corpus = Corpus.load(root)
    assert len(corpus.items) == 101 and corpus.items[-1].id == "Q101"
    assert (root / "data" / "daily" / "2026-11-15.jsonl").exists()
    assert (inbox / "rejected" / "bad.json").exists()
    again = ingest(inbox, root, seed=DATASET, date=dt.date(2026, 11, 16))
    assert again["files"] == [] and again["added"] == 0
