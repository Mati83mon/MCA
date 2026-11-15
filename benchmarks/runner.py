"""Multi-model evaluation harness: raw LLM vs MCA-wrapped LLM.

Examples::

    # zero-download pipeline (lexical NLI, hashing embeddings, KB responder)
    python -m benchmarks.runner --mode mca-offline

    # raw baseline and MCA-wrapped Qwen2.5 on GPU, dataset pulled from the Hub
    python -m benchmarks.runner --mode raw --model Qwen/Qwen2.5-7B-Instruct \
        --dataset hf://datasets/Mati83moni/mca-adversarial-benchmarks
    python -m benchmarks.runner --mode mca --model Qwen/Qwen2.5-7B-Instruct \
        --nli-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.metrics import EvalRecord, classify_response, compute_metrics, format_confusion
from mca.config import MCAConfig
from mca.telemetry.schemas import BenchmarkDataset, Category, QuestionItem

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "pytania_podchwytliwe_100.json"
HUB_FILE = "data/pytania_podchwytliwe.jsonl"

logger = logging.getLogger("mca.bench")


def load_questions(source: str | Path = DEFAULT_DATASET) -> list[QuestionItem]:
    """Load + validate questions from a local JSON/JSONL file or ``hf://datasets/OWNER/NAME[/path]``."""
    src = str(source)
    if src.startswith("hf://datasets/"):
        from huggingface_hub import hf_hub_download

        parts = src.removeprefix("hf://datasets/").split("/")
        repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:]) or HUB_FILE
        src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    path = Path(src)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [QuestionItem.model_validate_json(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "pytania" in data:
        return BenchmarkDataset.model_validate(data).pytania
    return [QuestionItem.model_validate(item) for item in data]


async def _evaluate_mca(orchestrator: Any, items: list[QuestionItem], concurrency: int) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: QuestionItem) -> dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            result = await orchestrator.run(item.pytanie)
            return {
                "id": item.id, "category": item.kategoria.value, "question": item.pytanie,
                "gold": item.kategoria.expected_decision.value, "pred": result.decision.decision.value,
                "confidence": result.decision.confidence, "rules": result.decision.triggered_rules,
                "response": result.final_response, "reference": item.odpowiedz_wzorcowa,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "telemetry": result.telemetry.model_dump(mode="json"),
            }

    return list(await asyncio.gather(*(one(i) for i in items)))


async def _evaluate_raw(llm: Any, items: list[QuestionItem], max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        t0 = time.perf_counter()
        answer = await llm.generate(item.pytanie, temperature=0.0, max_new_tokens=max_new_tokens)
        rows.append({
            "id": item.id, "category": item.kategoria.value, "question": item.pytanie,
            "gold": item.kategoria.expected_decision.value, "pred": classify_response(answer).value,
            "confidence": None, "response": answer, "reference": item.odpowiedz_wzorcowa,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from mca.telemetry.schemas import DecisionEnum

    records = [EvalRecord(r["id"], r["category"], DecisionEnum(r["gold"]), DecisionEnum(r["pred"]), r.get("confidence")) for r in rows]
    metrics = compute_metrics(records)
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["gold"] == r["pred"])
    metrics["accuracy_by_category"] = {c: round(sum(v) / len(v), 4) for c, v in by_cat.items()}
    lat = sorted(r["latency_ms"] for r in rows)
    metrics["latency_ms_p50"] = lat[len(lat) // 2] if lat else 0.0
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["mca-offline", "raw", "mca"], default="mca-offline")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--nli-model", default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--categories", nargs="*", choices=[c.value for c in Category], default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-uncertainty", action="store_true", help="disable semantic-entropy sampling")
    p.add_argument("--out", default="results")
    return p


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    items = load_questions(args.dataset)
    if args.categories:
        items = [i for i in items if i.kategoria.value in args.categories]
    if args.limit:
        items = items[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.mode if args.mode == "mca-offline" else f"{args.mode}__{args.model.replace('/', '__')}"

    if args.mode == "mca-offline":
        from mca import build_offline_orchestrator

        orch = build_offline_orchestrator(enable_uncertainty=not args.no_uncertainty)
        rows = await _evaluate_mca(orch, items, args.concurrency)
        orch.close()
    else:
        from mca.llm.transformers_llm import TransformersLLM

        llm = TransformersLLM(args.model, load_in_4bit=args.load_in_4bit)
        if args.mode == "raw":
            rows = await _evaluate_raw(llm, items, args.max_new_tokens)
        else:
            from mca.core.orchestrator import MCAOrchestrator

            overrides: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
            if args.nli_model:
                overrides["nli_model"] = args.nli_model
            if args.embedding_model:
                overrides["embedding_model"] = args.embedding_model
            orch = MCAOrchestrator(llm, MCAConfig.from_env(**overrides), enable_uncertainty=not args.no_uncertainty)
            rows = await _evaluate_mca(orch, items, 1)
            orch.close()

    metrics = summarize(rows)
    metrics["mode"], metrics["model"] = args.mode, None if args.mode == "mca-offline" else args.model
    with (out_dir / f"{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"{tag}.metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    metrics = asyncio.run(amain(args))
    from mca.telemetry.schemas import DecisionEnum

    print(format_confusion(metrics["confusion_matrix"], [DecisionEnum(x) for x in metrics["labels"]]))
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"confusion_matrix", "per_class", "labels"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
