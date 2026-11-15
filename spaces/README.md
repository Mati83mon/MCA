---
title: MCA Metacognitive Explorer
emoji: 🧭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Raw LLM vs MCA-gated LLM with live metacognitive telemetry
models:
  - Qwen/Qwen2.5-1.5B-Instruct
  - MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
datasets:
  - Mati83moni/mca-adversarial-benchmarks
---

# MCA Metacognitive Explorer

Side-by-side comparison of a raw instruction-tuned LLM and the same model wrapped by the
**Metacognitive Control Architecture** (premise validation · semantic entropy · Z3 SMT logic ·
evidence grounding · deterministic policy gate).

* **Left** – direct generation (baseline).
* **Right** – MCA output and telemetry: premise verdicts, Z3 status (`SAT`/`UNSAT` + unsat core /
  `PARADOX_SELF_REF` + Kleene assignment / `UNDERDETERMINED`), semantic-entropy gauge, evidence
  support ratio, final decision and per-layer latency.

Runs on ZeroGPU (`@spaces.GPU`). Set `MCA_SPACE_OFFLINE=1` to run a zero-download CPU variant.
Code: the `mca` package in this Space is a copy of the framework repository.
