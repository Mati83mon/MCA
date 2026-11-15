# MCA – Metacognitive Control Architecture

Neuro-symbolic middleware that sits between a user and an LLM and decides **whether the model
should answer at all** – and if not, *why*. It blocks false-premise sycophancy, refuses to
confabulate under missing information, and explains paradoxes instead of "solving" them.

```
query ─┬─► base LLM stream ──► FIFO buffer ─────────────────────────────┐
       │                                                                 ▼
       ├─► L1 Premise Validator  (presuppositions, referents, NLI, oracles)   flush after 250 ms
       ├─► L3 Logic Engine       (Z3: SAT / UNSAT core / PARADOX_SELF_REF /   if no violation,
       │                          UNDERDETERMINED, Kleene K3 / Łukasiewicz Ł3) else abort + replace
       ├─► L4 Evidence Verifier  (dense retrieval, support ratio,             ▲
       │                          epistemic void vs logical inconsistency)    │
       └─► L2 Uncertainty        (semantic entropy, N samples, bi-NLI) ──► L5 Policy Gate
                                                           PASS · CHALLENGE · ABSTAIN · EXPLAIN_PARADOX
```

| Layer | Module | What it does |
|---|---|---|
| L1 | `mca/core/premise_validator.py` | Presupposition triggers (`Dlaczego X?` ⇒ *X*; `Która N jest <Name>?` ⇒ *<Name> jest N*; wh ⇒ ∃), unresolved-referent detection (possessives, deixis, unknown proper names), optional LLM JSON decomposition, validation via symbolic oracles then NLI against retrieved evidence. |
| L2 | `mca/core/uncertainty.py` | Kuhn et al. semantic entropy: N samples → bidirectional-entailment clusters → Shannon entropy; speculative early stop after 2 equivalent samples. |
| L3 | `mca/core/logic_engine.py`, `mca/utils/z3_helpers.py`, `mca/core/formalizer.py` | Safe AST→Z3 DSL (private `z3.Context` per check), tracked assertions + deletion-minimised `unsat_core()`, determinacy check (two models with different target), entailment check with counter-model, 3-valued fixed points for `P ⇔ ¬P`. |
| L4 | `mca/core/evidence_verifier.py` | `EvidenceStore` (sentence-transformers or hashing embedder, FAISS `IndexFlatIP` or NumPy), paradox catalog, `support_ratio`, *epistemic void* vs *logical inconsistency*. |
| L5 | `mca/core/decision_controller.py` | Pure function of `MetacognitiveTelemetry` (Pydantic v2) → `PolicyDecision`, ordered audit-logged rules R1–R9. |
| — | `mca/core/orchestrator.py` | Async speculative fast/slow pipeline: stream into a buffer, verify concurrently, flush after `flush_window_ms`, cancel generation on violation, late-violation correction, background entropy. |
| — | `mca/core/oracles.py` | AST-whitelisted Python execution (`all([])`, `t[0] = 7`), number-theory oracle. |

### Policy rules (first match wins)

| Rule | Signal | Decision |
|---|---|---|
| R1 | Z3 `PARADOX_SELF_REF` (classical UNSAT, only Kleene gap fixed point) | `EXPLAIN_PARADOX` |
| R2 | `UNSAT` core among presuppositions / failed entailment | `CHALLENGE` |
| R3 | Oracle refutes a presupposition | `CHALLENGE` |
| R4 | Query matches a catalogued paradox (similarity ≥ `paradox_similarity`) | `EXPLAIN_PARADOX` |
| R5 | NLI contradiction of a (non-existential) presupposition | `CHALLENGE` |
| R6 | Target quantity `UNDERDETERMINED` | `ABSTAIN` |
| R7 | Unresolved referent | `ABSTAIN` |
| R8 | Normalised semantic entropy > `entropy_tau` | `ABSTAIN` |
| R9 | none of the above | `PASS` |

## Install & run

```bash
pip install -e ".[dev]"                 # core: pydantic, numpy, z3-solver (CPU, zero downloads)
pip install -e ".[ml,faiss,hub]"        # + DeBERTa NLI, sentence-transformers, FAISS, HF Hub

pytest -q                               # 57 tests (Z3, NLI, oracles, entropy, streaming, metrics, ingest)
python -m benchmarks.runner --mode mca-offline
python -m benchmarks.runner --mode raw --model Qwen/Qwen2.5-7B-Instruct
python -m benchmarks.runner --mode mca --model Qwen/Qwen2.5-7B-Instruct \
       --nli-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

```python
import asyncio
from mca import build_offline_orchestrator

orch = build_offline_orchestrator()
res = asyncio.run(orch.run("Dlaczego Mars jest drugą planetą od Słońca?"))
print(res.decision.decision, res.final_response)
# DecisionEnum.CHALLENGE  Pytanie opiera się na fałszywym założeniu: „Mars jest drugą planetą od Słońca”. Sprzeczne z: …

async def demo():
    async for ev in orch.stream("Która planeta jest druga od Słońca?"):
        print(ev.kind, ev.t_ms, ev.text)
asyncio.run(demo())
```

With a real model: `MCAOrchestrator(TransformersLLM("Qwen/Qwen2.5-7B-Instruct"), MCAConfig.from_env())`.
All thresholds live in `mca/config.py` and can be overridden with `MCA_*` environment variables.

## Results (offline backends: lexical NLI, hashing embeddings, KB responder, CPU)

| Set | n | Acc | macro-F1 | FAR | FAR_abstain | Over-refusal | Paradox ID |
|---|---|---|---|---|---|---|---|
| `pytania_podchwytliwe_100` | 100 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| `heldout_probes` (24 items written after freezing the pipeline) | 24 | 0.67 | 0.68 | 0.33 | 0.50 | 0.17 | 0.67 |

**Read the first row as a regression test, not a generalisation claim.** The bundled knowledge base
(`mca/knowledge/seed_kb.jsonl`), the paradox catalog and the formalizer templates were written with the
benchmark's topics in view, so the 100-item score is in-distribution. The held-out probes show where the
offline symbolic stack stops: paraphrased paradoxes without catalog vocabulary, facts outside the KB
(over-refusal on unknown proper names), and quantity questions without a matching schema. Closing that gap
is the job of the neural path – LLM decomposition into the Z3 DSL, a trained NLI cross-encoder and
semantic entropy over real samples – which the Kaggle notebook measures on GPU.

**Calibration notes.** `paradox_similarity` (0.30) and `retrieval_min_similarity` (0.35) are calibrated for the
`HashingEmbedder`; recalibrate them when switching to a sentence-transformer. The default
`cross-encoder/nli-deberta-v3-large` is English-only; for the Polish benchmark use a multilingual NLI model
(the Space and notebook use `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`).

## Metrics (`benchmarks/metrics.py`)

* **FAR** – share of trap items (gold ≠ PASS) answered as ordinary questions.
* **FAR_abstain** – share of missing-data items where the system did not abstain.
* **Over-refusal** – share of control facts not answered.
* **Paradox identification accuracy** – recall of `EXPLAIN_PARADOX`.
* **ECE** – calibration of decision confidence (10 bins). Raw models are mapped to decisions with an ordered
  Polish/English cue lexicon (`classify_response`).

## Repository layout

```
mca/            config, telemetry schemas, 5 layers + orchestrator, utils (NLI, Z3, embeddings), LLM backends, seed KB
benchmarks/     runner.py (raw / mca / mca-offline), metrics.py, datasets/ (100 items + 24 held-out probes)
spaces/         Gradio ZeroGPU explorer (raw vs MCA + telemetry dashboard)
kaggle/         mca_gpu_evaluator.ipynb – 2×T4 / P100 batch benchmark (Qwen2.5, Llama-3, DeepSeek)
scripts/        ingest_daily_json.py – validated, de-duplicated daily ingestion (+ optional Hub push)
tests/          pytest suite
```

## Daily ingestion

```bash
HF_TOKEN=... python -m scripts.ingest_daily_json --inbox /data/inbox --corpus-dir /data/mca-corpus \
    --push Mati83moni/mca-adversarial-benchmarks
# cron: 15 4 * * *  (runs after the 04:00 task drop)
```

Each file is validated against `QuestionItem`; invalid files go to `inbox/rejected/` with an error report,
duplicates (normalised question text) are skipped, colliding IDs are renumbered, and a dated partition
`data/daily/YYYY-MM-DD.jsonl` is written next to the cumulative corpus.

## Hugging Face

* Dataset: [`Mati83moni/mca-adversarial-benchmarks`](https://huggingface.co/datasets/Mati83moni/mca-adversarial-benchmarks)
* Space: [`Mati83moni/mca-metacognitive-explorer`](https://huggingface.co/spaces/Mati83moni/mca-metacognitive-explorer)

## Security

Formal specs (including LLM-produced ones) are compiled from an AST whitelist, never `eval`-ed. The Python
oracle executes only whitelisted node types and built-ins (no attributes except `json.dumps/loads`, no
dunder names, no `**`).

## License

Apache-2.0.
