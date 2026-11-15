"""MCA Metacognitive Explorer – Hugging Face ZeroGPU Gradio Space.

Left pane: raw LLM answer. Right pane: MCA-orchestrated answer plus a telemetry
dashboard (premises, Z3 logic, semantic entropy, evidence grounding, decision).

Environment overrides:
  MCA_SPACE_LLM        causal LM            (default Qwen/Qwen2.5-1.5B-Instruct)
  MCA_SPACE_NLI        NLI cross-encoder    (default multilingual mDeBERTa XNLI – the dataset is Polish)
  MCA_SPACE_OFFLINE=1  zero-download CPU mode (KB responder + lexical NLI)
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):  # Space root holds ./mca; in the repo it's ../mca
    if (candidate / "mca" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

import gradio as gr  # noqa: E402

try:
    import spaces  # noqa: E402

    GPU = spaces.GPU(duration=120)
except (ImportError, AttributeError):  # local run: no ZeroGPU runtime (or the repo's ./spaces dir shadows it)
    def GPU(fn):  # type: ignore[no-redef]
        return fn

from mca import MCAConfig, MCAOrchestrator, build_offline_orchestrator  # noqa: E402
from mca.core.evidence_verifier import EvidenceStore  # noqa: E402
from mca.telemetry.schemas import MCAResult, PremiseStatus  # noqa: E402
from mca.utils.embeddings import HashingEmbedder  # noqa: E402

LLM_NAME = os.environ.get("MCA_SPACE_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
NLI_NAME = os.environ.get("MCA_SPACE_NLI", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
OFFLINE = os.environ.get("MCA_SPACE_OFFLINE") == "1"


def _build() -> tuple[MCAOrchestrator, object]:
    if OFFLINE:
        orch = build_offline_orchestrator(MCAConfig(nli_backend="lexical", embedding_backend="hashing", n_samples=3))
        return orch, orch.llm
    import torch

    from mca.llm.transformers_llm import TransformersLLM
    from mca.utils.nli_client import NLIClient, TransformersNLIBackend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = TransformersLLM(LLM_NAME, device_map=device)
    nli = NLIClient(TransformersNLIBackend(NLI_NAME, device=device))
    # Retrieval uses the calibrated hashing embedder (thresholds in MCAConfig are tuned for it).
    cfg = MCAConfig(nli_backend="transformers", nli_model=NLI_NAME, embedding_backend="hashing", n_samples=5, max_new_tokens=200)
    store = EvidenceStore.from_seed(HashingEmbedder(), use_faiss=cfg.use_faiss)
    return MCAOrchestrator(llm, cfg, store=store, nli=nli), llm


ORCH, RAW_LLM = _build()

EXAMPLES = [
    "Dlaczego Mars jest drugą planetą od Słońca?",
    "Zdanie L brzmi: „Zdanie L jest fałszywe”. Czy L jest prawdziwe, czy fałszywe?",
    "Samochód przejechał 120 km. Jaka była jego średnia prędkość?",
    "Która planeta jest druga od Słońca?",
    "Dlaczego all([]) w Pythonie zwraca False?",
    "Achilles biegnie 10 m/s, żółw 1 m/s i ma 100 m przewagi. Czy dogoni go dopiero po nieskończonym czasie?",
    "Ile lat ma Anna?",
    "Która dodatnia liczba całkowita jest mniejsza od zera?",
]

DECISION_COLOR = {"PASS": "#1f9d55", "CHALLENGE": "#d97706", "ABSTAIN": "#2563eb", "EXPLAIN_PARADOX": "#7c3aed"}
STATUS_COLOR = {PremiseStatus.SUPPORTED: "#1f9d55", PremiseStatus.CONTRADICTED: "#dc2626", PremiseStatus.UNVERIFIED: "#6b7280"}

CSS = """
.mca-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 12px;margin-bottom:8px}
.mca-card h4{margin:0 0 6px 0;font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;opacity:.75}
.mca-badge{display:inline-block;color:#fff;border-radius:999px;padding:2px 10px;font-weight:600;font-size:.8rem}
.mca-bar{height:10px;border-radius:5px;background:var(--neutral-200);overflow:hidden}
.mca-bar>div{height:100%}
.mca-mono{font-family:ui-monospace,monospace;font-size:.78rem;white-space:pre-wrap;word-break:break-word}
"""


def _bar(value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f'<div class="mca-bar"><div style="width:{pct:.0f}%;background:{color}"></div></div>'


def render_dashboard(result: MCAResult) -> str:
    t, d = result.telemetry, result.decision
    color = DECISION_COLOR[d.decision.value]
    parts = [
        f'<div class="mca-card"><h4>Decision state</h4><span class="mca-badge" style="background:{color}">{d.decision.value}</span>'
        f' &nbsp;confidence {d.confidence:.2f} &nbsp;<span class="mca-mono">{html.escape(", ".join(d.triggered_rules))}</span></div>'
    ]
    rows = "".join(
        f'<li><span class="mca-badge" style="background:{STATUS_COLOR[v.status]}">{v.status.value}</span> '
        f'<b>{html.escape(v.premise.kind.value)}</b>{" ∃" if v.premise.existential else ""}: {html.escape(v.premise.text)}'
        f'<br><small>{html.escape(v.explanation)}</small></li>'
        for v in t.premise_verdicts
    ) or "<li>brak wyodrębnionych założeń</li>"
    parts.append(f'<div class="mca-card"><h4>Premise status</h4><ul style="margin:0;padding-left:18px">{rows}</ul></div>')
    logic = t.logic
    core = f"<br>unsat_core: <span class='mca-mono'>{html.escape(str(logic.unsat_core))}</span>" if logic.unsat_core else ""
    kleene = f"<br>Kleene K3: <span class='mca-mono'>{html.escape(str(logic.kleene_assignment))}</span>" if logic.kleene_assignment else ""
    formulas = html.escape("\n".join(logic.formulas[:8]))
    parts.append(
        f'<div class="mca-card"><h4>Z3 logic status</h4><b>{logic.status.value}</b>{core}{kleene}'
        f'<br><small>{html.escape(logic.explanation)}</small><div class="mca-mono">{formulas}</div></div>'
    )
    u = t.uncertainty
    if u is not None:
        ent_color = "#dc2626" if u.normalized_entropy > ORCH.config.entropy_tau else "#1f9d55"
        parts.append(
            f'<div class="mca-card"><h4>Semantic entropy</h4>{_bar(u.normalized_entropy, ent_color)}'
            f'H = {u.semantic_entropy:.3f} nats · normalised {u.normalized_entropy:.2f} (τ = {ORCH.config.entropy_tau}) · '
            f'{u.n_samples} samples · {len(u.clusters)} clusters{" · early stop" if u.early_stopped else ""}</div>'
        )
    else:
        parts.append('<div class="mca-card"><h4>Semantic entropy</h4>pominięta – fast path zatrzymał generowanie przed próbkowaniem</div>')
    ev = t.evidence
    top = "".join(f"<li>[{e.score:.2f}] {html.escape(e.text)}</li>" for e in ev.retrieved[:3])
    void = " · <b>epistemic void</b>" if ev.epistemic_void else ""
    incons = " · <b>logical inconsistency</b>" if ev.logical_inconsistency else ""
    missing = f"<br>Brakujące zmienne: {html.escape('; '.join(ev.missing_variables))}" if ev.missing_variables else ""
    paradox = f"<br>Paradox catalog match: {html.escape(ev.paradox_match.doc_id)} ({ev.paradox_match.score:.2f})" if ev.paradox_match else ""
    parts.append(
        f'<div class="mca-card"><h4>Evidence grounding</h4>{_bar(ev.support_ratio, "#2563eb")}support ratio {ev.support_ratio:.2f}'
        f'{void}{incons}{missing}{paradox}<ul style="margin:4px 0 0 0;padding-left:18px">{top}</ul></div>'
    )
    lat = " · ".join(f"{k} {v:.0f} ms" for k, v in t.latency_ms.items())
    parts.append(f'<div class="mca-card"><h4>Latency</h4><span class="mca-mono">{html.escape(lat)}</span></div>')
    return "".join(parts)


@GPU
def compare(question: str) -> tuple[str, str, str, str]:
    question = (question or "").strip()
    if not question:
        return "", "", "", "{}"

    async def _both() -> tuple[str, MCAResult]:
        raw = await RAW_LLM.generate(question, temperature=0.0, max_new_tokens=200)
        mca = await ORCH.run(question)
        return raw, mca

    raw, result = asyncio.run(_both())
    return raw, result.final_response, render_dashboard(result), json.dumps(result.telemetry.model_dump(mode="json"), ensure_ascii=False, indent=2)


with gr.Blocks(title="MCA Metacognitive Explorer", css=CSS) as demo:
    gr.Markdown(
        "# MCA – Metacognitive Control Architecture\n"
        f"Raw **{'KB responder (offline)' if OFFLINE else LLM_NAME}** vs. the same model wrapped by MCA: premise validation (NLI), "
        "semantic entropy, Z3 logic, evidence grounding and a deterministic policy gate "
        "(`PASS` / `CHALLENGE` / `ABSTAIN` / `EXPLAIN_PARADOX`)."
    )
    question = gr.Textbox(label="Pytanie", lines=2, placeholder="Zadaj pytanie – najlepiej podchwytliwe…")
    run = gr.Button("Porównaj", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Raw LLM (baseline)")
            raw_out = gr.Textbox(label="Odpowiedź bez MCA", lines=8)
        with gr.Column():
            gr.Markdown("### MCA-orchestrated")
            mca_out = gr.Textbox(label="Odpowiedź z MCA", lines=8)
            dashboard = gr.HTML()
    with gr.Accordion("Pełna telemetria (JSON)", open=False):
        telemetry = gr.Code(language="json")
    gr.Examples(EXAMPLES, inputs=question)
    run.click(compare, inputs=question, outputs=[raw_out, mca_out, dashboard, telemetry])
    question.submit(compare, inputs=question, outputs=[raw_out, mca_out, dashboard, telemetry])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
