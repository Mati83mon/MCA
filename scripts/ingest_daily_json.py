"""Daily auto-ingestion of question files (e.g. the 04:00 task drop) into the MCA benchmark.

Pipeline for every ``*.json`` / ``*.jsonl`` file in ``--inbox`` that has not been
processed before (tracked by SHA-256 in ``manifest.json``):

1. parse (whole-dataset JSON with ``metadane``/``pytania``, a JSON list, or JSONL);
2. validate every item against :class:`mca.telemetry.schemas.QuestionItem`;
3. de-duplicate against the existing corpus by normalised question text;
   colliding IDs are re-numbered to the next free ``Q###``;
4. append to ``data/pytania_podchwytliwe.jsonl`` and write a dated partition
   ``data/daily/YYYY-MM-DD.jsonl``;
5. optionally push the changed files to the Hugging Face dataset repo in one commit.

Invalid files are moved to ``<inbox>/rejected/`` with a ``.errors.json`` report;
nothing partial is written.

Cron example (04:15 daily, after the 04:00 drop)::

    15 4 * * *  cd /opt/mca && HF_TOKEN=... python -m scripts.ingest_daily_json \
        --inbox /data/inbox --corpus-dir /data/mca-corpus \
        --push Mati83moni/mca-adversarial-benchmarks >> /var/log/mca-ingest.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mca.telemetry.schemas import BenchmarkDataset, QuestionItem

logger = logging.getLogger("mca.ingest")
CORPUS_FILE = Path("data") / "pytania_podchwytliwe.jsonl"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", " ", text).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_file(path: Path) -> list[QuestionItem]:
    """Parse and validate a single incoming file; raises ``ValueError`` with all item errors."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        raw: list[Any] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, dict) and "pytania" in data:
            return BenchmarkDataset.model_validate(data).pytania
        raw = data if isinstance(data, list) else [data]
    items, errors = [], []
    for i, obj in enumerate(raw):
        try:
            items.append(QuestionItem.model_validate(obj))
        except ValidationError as exc:
            errors.append({"index": i, "errors": json.loads(exc.json())})
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    return items


@dataclass
class Corpus:
    root: Path
    items: list[QuestionItem] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.root / CORPUS_FILE

    @classmethod
    def load(cls, root: Path, seed: Path | None = None) -> "Corpus":
        corpus = cls(root)
        if corpus.path.exists():
            corpus.items = [QuestionItem.model_validate_json(l) for l in corpus.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        elif seed is not None and seed.exists():
            corpus.items = parse_file(seed)
        return corpus

    def next_id(self) -> str:
        nums = [int(i.id[1:]) for i in self.items]
        return f"Q{(max(nums) + 1) if nums else 1:03d}"

    def merge(self, new_items: list[QuestionItem]) -> tuple[list[QuestionItem], int]:
        seen_text = {_norm(i.pytanie) for i in self.items}
        seen_ids = {i.id for i in self.items}
        added, skipped = [], 0
        for item in new_items:
            if _norm(item.pytanie) in seen_text:
                skipped += 1
                continue
            if item.id in seen_ids:
                item = item.model_copy(update={"id": self.next_id()})
            self.items.append(item)
            seen_text.add(_norm(item.pytanie))
            seen_ids.add(item.id)
            added.append(item)
        return added, skipped

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(i.model_dump_json() + "\n" for i in self.items), encoding="utf-8")
        tmp.replace(self.path)


def push_to_hub(repo_id: str, corpus_root: Path, paths: list[Path], message: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(corpus_root)), path_or_fileobj=str(p)) for p in paths]
    info = api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=message)
    return info.commit_url


def ingest(inbox: Path, corpus_root: Path, *, seed: Path | None = None, push: str | None = None,
           date: dt.date | None = None, dry_run: bool = False) -> dict[str, Any]:
    date = date or dt.date.today()
    manifest_path = corpus_root / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"processed": {}}
    corpus = Corpus.load(corpus_root, seed)
    summary: dict[str, Any] = {"date": date.isoformat(), "files": [], "added": 0, "duplicates": 0, "rejected": []}
    daily: list[QuestionItem] = []
    for path in sorted(list(inbox.glob("*.json")) + list(inbox.glob("*.jsonl"))):
        digest = _sha256(path)
        if digest in manifest["processed"]:
            continue
        try:
            items = parse_file(path)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            logger.error("rejecting %s: %s", path.name, str(exc)[:300])
            summary["rejected"].append(path.name)
            if not dry_run:
                rejected = inbox / "rejected"
                rejected.mkdir(exist_ok=True)
                shutil.move(str(path), rejected / path.name)
                (rejected / f"{path.name}.errors.json").write_text(str(exc), encoding="utf-8")
            continue
        added, dupes = corpus.merge(items)
        daily.extend(added)
        summary["files"].append({"file": path.name, "items": len(items), "added": len(added), "duplicates": dupes})
        summary["added"] += len(added)
        summary["duplicates"] += dupes
        manifest["processed"][digest] = {"file": path.name, "date": date.isoformat(), "added": [i.id for i in added]}
    if dry_run or not summary["files"]:
        return summary
    corpus.save()
    changed = [corpus.path, manifest_path]
    if daily:
        part = corpus_root / "data" / "daily" / f"{date.isoformat()}.jsonl"
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open("a", encoding="utf-8") as fh:
            fh.writelines(i.model_dump_json() + "\n" for i in daily)
        changed.append(part)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["total"] = len(corpus.items)
    if push:
        summary["commit"] = push_to_hub(push, corpus_root, changed, f"Daily ingest {date.isoformat()}: +{summary['added']} items")
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Validate and append daily question files to the MCA benchmark corpus.")
    p.add_argument("--inbox", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, default=Path("."))
    p.add_argument("--seed", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks" / "datasets" / "pytania_podchwytliwe_100.json")
    p.add_argument("--push", metavar="REPO_ID", default=None, help="HF dataset repo to push to (needs HF_TOKEN)")
    p.add_argument("--date", type=dt.date.fromisoformat, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    summary = ingest(args.inbox, args.corpus_dir, seed=args.seed, push=args.push, date=args.date, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["rejected"] else 0


if __name__ == "__main__":
    sys.exit(main())
