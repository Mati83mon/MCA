"""Hugging Face ``transformers`` causal-LM backend (Qwen2.5, Llama-3, DeepSeek, ...)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator

DEFAULT_SYSTEM_PROMPT = "Jesteś pomocnym asystentem. Odpowiadaj zwięźle i rzeczowo, w języku pytania."
THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Drop a ``<think>…</think>`` trace (DeepSeek-R1 style) and keep the final answer."""
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1].strip()
    return text.strip()


class TransformersLLM:
    """Chat-template aware wrapper with async generate / stream / batched sampling.

    ``device_map="auto"`` shards larger checkpoints over all visible GPUs
    (e.g. Kaggle 2×T4). A lock serialises ``model.generate`` calls because HF
    models are not re-entrant.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | dict[str, Any] = "auto",
        dtype: str = "auto",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # "auto"/dict -> accelerate sharding (multi-GPU); a plain device string ("cuda", "cpu")
        # loads then moves the model, which is what Hugging Face ZeroGPU expects.
        move_to = device_map if isinstance(device_map, str) and device_map != "auto" else None
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if move_to is None:
            kwargs["device_map"] = device_map
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16 if dtype == "auto" and torch.cuda.is_available() else (
                getattr(torch, dtype) if dtype != "auto" else torch.float32
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if move_to is not None:
            self.model = self.model.to(move_to)
        self._lock = threading.Lock()
        # Reasoning models (R1 distills) ship a chat template that handles </think>.
        self.reasoning = THINK_CLOSE in (getattr(self.tokenizer, "chat_template", None) or "")

    # -- prompt handling -------------------------------------------------------
    def _encode(self, prompt: str, think: bool = True) -> Any:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.reasoning and not think:
                # Prefill an empty reasoning block so the model answers directly.
                text += "\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
        else:
            text = f"{self.system_prompt}\n\nPytanie: {prompt}\nOdpowiedź:"
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def _gen_kwargs(self, temperature: float, max_new_tokens: int) -> dict[str, Any]:
        kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        return kw

    def _generate_sync(self, prompt: str, temperature: float, max_new_tokens: int, n: int, seed: int | None, think: bool = True) -> list[str]:
        torch = self._torch
        inputs = self._encode(prompt, think)
        with self._lock, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            out = self.model.generate(**inputs, num_return_sequences=n, **self._gen_kwargs(temperature if n == 1 else max(temperature, 1e-3), max_new_tokens))
        prompt_len = inputs["input_ids"].shape[1]
        return [strip_reasoning(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)) for seq in out]

    # -- LLMBackend API ----------------------------------------------------------
    async def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256, seed: int | None = None) -> str:
        return (await asyncio.to_thread(self._generate_sync, prompt, temperature, max_new_tokens, 1, seed))[0]

    async def generate_many(self, prompt: str, n: int, *, temperature: float = 0.8, max_new_tokens: int = 96) -> list[str]:
        """Batched sampling via ``num_return_sequences`` (one prefill for all samples).

        Semantic entropy must compare *answers*: reasoning models are sampled with the
        thinking phase skipped, otherwise truncated traces land in singleton clusters.
        """
        return await asyncio.to_thread(self._generate_sync, prompt, temperature, max_new_tokens, n, None, False)

    async def stream(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> AsyncIterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._encode(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0)
        kwargs = dict(**inputs, streamer=streamer, **self._gen_kwargs(temperature, max_new_tokens))

        def _run() -> None:
            with self._lock, self._torch.inference_mode():
                self.model.generate(**kwargs)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        loop = asyncio.get_running_loop()
        iterator = iter(streamer)
        sentinel = object()
        while True:
            piece = await loop.run_in_executor(None, next, iterator, sentinel)
            if piece is sentinel:
                break
            if piece:
                yield piece
        await asyncio.to_thread(worker.join)
