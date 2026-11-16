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
| R8b | Entropy > `entropy_tau`, but a KB passage NLI-entails a premise (sim ≥ 0.6, entail ≥ 0.9) | `PASS`, answer = passage |
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

## Results on GPU (Kaggle 2×T4, real LLMs, mDeBERTa XNLI, n=5 entropy samples)

Same 100-item benchmark, each model evaluated raw (direct answer, classified into PASS/CHALLENGE/ABSTAIN/
EXPLAIN_PARADOX) and wrapped by MCA. fp16 sharded over 2×T4, `max_new_tokens=256`. Raw per-item outputs:
[`results/kaggle-2xT4/`](results/kaggle-2xT4/), notebook: [`kaggle/mca_gpu_evaluator.ipynb`](kaggle/mca_gpu_evaluator.ipynb).

| Run | Accuracy | Macro-F1 | FAR | FAR (abstain) | Over-refusal | Paradox ID | ECE | p50 latency |
|---|---|---|---|---|---|---|---|---|
| raw Qwen2.5-7B-Instruct | 0.48 | 0.45 | 0.56 | 0.44 | 0.04 | 0.28 | – | 4.1 s |
| **MCA + Qwen2.5-7B** | **0.94** | **0.94** | **0.00** | **0.00** | 0.20 | 0.96 | 0.05 | 17.6 s |
| raw DeepSeek-R1-Distill-Qwen-7B | 0.36 | 0.30 | 0.65 | 0.92 | 0.08 | 0.16 | – | 16.7 s |
| **MCA + DeepSeek-R1-7B** | **0.90** | **0.90** | **0.00** | **0.00** | 0.36 | 1.00 | 0.10 | 44.6 s |
| raw Meta-Llama-3-8B-Instruct | 0.30 | 0.20 | 0.89 | 0.92 | 0.00 | 0.04 | – | 1.8 s |
| **MCA + Llama-3-8B** | **0.85** | **0.84** | **0.01** | **0.04** | 0.52 | 1.00 | 0.09 | 19.8 s |

Per category (MCA, correct / 25 – fact · paradox · false premise · missing data):
Qwen 20·24·25·25, DeepSeek 16·25·24·25, Llama-3 12·25·24·24.

What the numbers say:

- **The wrapper removes almost all false acceptance.** Raw models answer 56–89 % of the 75 trap questions
  as if the premise were fine; with MCA that drops to 0 / 0 / 1 item (Llama-3 accepted Q048, a JSON
  `True` capitalisation premise that the KB does not cover).
- **The price is over-refusal on ordinary facts** (raw models: 0–8 %). The remaining refusals come from
  two rules: R8 semantic entropy with only 5 samples (Qwen 4, DeepSeek 9, Llama-3 7 facts) and, for
  Llama-3, R5 NLI contradiction against its own verbose answers (6 facts). Threshold calibration on these
  logs is the next step; no GPU needed.
- **Latency is 4–10× raw** (entropy sampling + NLI + Z3), so the fast/slow orchestrator's fast path matters
  in production.
- Same caveat as the CPU table: the KB and paradox catalog were written with these topics in view.

**Update: rule R8b (evidence answer), rescored on the same telemetry.** A ZeroGPU Space test
("Która planeta jest druga od Słońca?") showed R8 abstaining although the KB passage "Wenus jest drugą
planetą od Słońca." supported the question at similarity 0.88 / entailment 0.99. R8b now answers from that
passage when entropy is high but a KB passage NLI-entails a non-referent premise (similarity ≥ 0.6,
entailment ≥ 0.9); the answer is the passage, not the model's sample. Because the policy gate is a pure
function of the stored telemetry, `python -m benchmarks.rescore results/kaggle-2xT4` re-applies it exactly,
without re-running any model ([`rescored_r8b.csv`](results/kaggle-2xT4/rescored_r8b.csv)):

| MCA run | Accuracy | FAR | Over-refusal | Paradox ID |
|---|---|---|---|---|
| Qwen2.5-7B | 0.94 → **0.97** | 0.00 → 0.00 | 0.20 → **0.08** | 0.96 → 0.96 |
| DeepSeek-R1-7B | 0.90 → **0.97** | 0.00 → 0.00 | 0.36 → **0.08** | 1.00 → 1.00 |
| Llama-3-8B | 0.85 → **0.89** | 0.01 → 0.01 | 0.52 → **0.36** | 1.00 → 1.00 |

14 abstentions on control facts flip to correct answers and no trap item changes. Read this with two caveats:
the two thresholds were chosen after inspecting these 21 R8 abstentions, so the rescored numbers are
in-sample; and R8b can only answer facts the KB covers. Llama-3's remaining over-refusal comes from R5
(NLI contradiction on verbose answers), which R8b does not touch.

**Auditing every rule (uncensored calibration).** The ordered gate logs only the first rule that fires, so
end-to-end logs are a censored sample: in the runs above semantic entropy was recorded for only 20–26 of
100 items, because the slow path is cancelled once the fast path decides. With
`MCAConfig(audit_all_layers=True)` every layer runs on every item without changing the decision, and
`DecisionController.rule_verdicts()` returns each rule's own verdict (a test checks it reproduces
`decide()` on all stored rows). `python -m benchmarks.rule_table table|calibrate` then builds the per-item
rule table and grid-searches the thresholds offline. It tunes on the 100 items and evaluates the chosen
config exactly once on the held-out probes. (Suggested by a reviewer on the HF forum thread.)

**Audit run (Kaggle v10, R8b live, `audit_all_layers=True`)** – entropy observed on 100/100 benchmark and
24/24 held-out items per model. Data: [`results/kaggle-2xT4/audit-v10/`](results/kaggle-2xT4/audit-v10/)
(per-item rule tables and calibration results in `rule_tables/`).

| Model | Benchmark acc / FAR / over-refusal | Held-out MCA acc / FAR / over-refusal | Held-out raw acc / FAR |
|---|---|---|---|
| Qwen2.5-7B | 0.96 / 0.00 / 0.12 | 0.79 / 0.06 / 0.17 | 0.46 / 0.67 |
| DeepSeek-R1-7B | 0.98 / 0.00 / 0.04 | 0.83 / 0.00 / 0.17 | 0.33 / 0.78 |
| Llama-3-8B | 0.89 / 0.00 / 0.36 | 0.71 / 0.00 / 0.67 | 0.25 / 0.89 |

What the uncensored sample shows:

* **The fresh run confirms the R8b rescoring** (predicted 0.97 / 0.97 / 0.89, observed 0.96 / 0.98 / 0.89, FAR 0 for all three).
* **On the 100 items R8 is nearly redundant as a trap detector.** Traps that pass R1–R7 are 0 (Qwen), 1 (DeepSeek) and 1 (Llama). In-distribution its net effect is over-refusal. On held-out it does real work: for DeepSeek 4 traps reach the entropy stage and R8 catches all 4.
* **Entropy separates traps from control facts only moderately**: AUROC 0.68 / 0.80 / 0.76 on the benchmark and 0.79 / 0.87 / 0.85 on held-out (n = 5 samples). Llama-3 has high entropy on 13/25 control facts; R8b rescues 10 of them.
* **Calibrating τ on the benchmark overfits.** The grid picked τ = 0.9 for DeepSeek: 0.99 on the benchmark, but held-out FAR rises from 0.00 to 0.22. That happens because the benchmark has almost no trap that only R8 can catch, so raising τ costs nothing there. For Qwen the calibrated config (τ = 0.5) gives identical held-out results. For Llama the calibrated τ = 0.6 / similarity 0.5 / entailment 0.8 improves held-out accuracy from 0.71 to 0.75 and lowers over-refusal from 0.67 to 0.50. The defaults stay unchanged.
* **Remaining errors:** the dominant held-out error is R7 (unresolved referents outside the KB), and Llama's in-distribution errors are R5 (6 control facts flagged by NLI). With only 6 control and 18 trap held-out items per model (one item = 0.17 over-refusal), these are directional signals, not estimates.

**Before the fixes (first GPU run, v3)** MCA+Qwen scored acc 0.75 with over-refusal **1.00** (every LLM-
extracted referent hit R7) and MCA+DeepSeek acc 0.80 / over-refusal 0.76 (entropy computed on truncated
`<think>` traces, plus LLM formal specs with infix `And`/`Or` failing to parse). Commit `05084ba` grounds
LLM referents in the KB, skips/strips reasoning traces for sampling and normalises the DSL; metrics of
that run are kept in [`results/kaggle-2xT4/v3-prefix/`](results/kaggle-2xT4/v3-prefix/).

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

## Community

* Kaggle notebook: [`mateuszpiesiak/mca-gpu-evaluator`](https://www.kaggle.com/code/mateuszpiesiak/mca-gpu-evaluator)
* Kaggle discussion: <https://www.kaggle.com/discussions/general/759501>
* Hugging Face post: <https://huggingface.co/posts/Mati83moni/245309888731952>

Feedback, adversarial questions that fool MCA and results on other models are welcome. Open an issue or reply in either thread.

## Security

Formal specs (including LLM-produced ones) are compiled from an AST whitelist, never `eval`-ed. The Python
oracle executes only whitelisted node types and built-ins (no attributes except `json.dumps/loads`, no
dunder names, no `**`).

## License

Apache-2.0.
