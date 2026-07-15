"""Unified registry of the 9 translation systems under test.

Every system exposes the same call:  translate(text, context=None) -> str
returning normalized Uzbek-Latin. Heavy backends (torch, openai, google-genai)
are imported lazily inside each adapter, so importing this module is cheap and
running only NLLB doesn't require the LLM SDKs.

Serving (see METHODOLOGY.md §2):
  - NLLB 1.3B / 3.3B ....... local, via the project's NllbTranslator wrapper
  - gemma4 12b/26b ......... local Ollama (OpenAI-compatible :11434/v1)
  - gemma4:31b-cloud ....... Ollama Cloud (ollama.com/v1, OLLAMA_API_KEY)
  - translategemma 12b/27b . local Ollama
  - gemini-3.5-flash ....... Google GenAI (GEMINI_API_KEY), minimal thinking

Env: OLLAMA_LOCAL_URL (default http://localhost:11434/v1), OLLAMA_CLOUD_URL
(default https://ollama.com/v1), OLLAMA_API_KEY (cloud), GEMINI_API_KEY.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .nllb_translator import NllbTranslator, normalize_uz

_STRIP_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# HF generation config for the small Uzbek-tuned models (mirrors model_eval.py:
# greedy + repetition guards keep 4-8B models from looping).
HF_MAX_NEW = 256
HF_REP_PENALTY = 1.3
HF_NO_REPEAT_NGRAM = 4

# ── the fixed translation prompt (user-specified; identical across LLM systems) ─
SOURCE_LANG, SOURCE_CODE = "English", "en"
TARGET_LANG, TARGET_CODE = "Uzbek", "uz"
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_THINKING_BUDGET = 0        # minimal thinking mode (0 = off; raise if desired)

_PROMPT = (
    "You are a professional {sl} ({sc}) to {tl} ({tc}) translator. Your goal is to "
    "accurately convey the meaning and nuances of the original {sl} text while "
    "adhering to {tl} grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the {tl} translation, without any additional explanations or "
    "commentary. Please translate the following {sl} text into {tl}:\n\n\n{text}"
)


def build_prompt(text: str) -> str:
    return _PROMPT.format(
        sl=SOURCE_LANG, sc=SOURCE_CODE, tl=TARGET_LANG, tc=TARGET_CODE, text=text
    )


def _context_preamble(context: str | None) -> str:
    """Prior dialogue turn as context, not to be translated."""
    if not context:
        return ""
    return (
        f"Conversation context — the previous turn, for reference only, do NOT "
        f"translate it:\n{context}\n\n"
    )


# ── base ─────────────────────────────────────────────────────────────────────
class System:
    key: str

    def translate(self, text: str, context: str | None = None) -> str:
        raise NotImplementedError


# ── NLLB (local, enc-dec MT) ─────────────────────────────────────────────────
class NllbSystem(System):
    """Wraps the production NllbTranslator (entity-masking + sentence-split).

    NLLB is sentence-level and gets its real deployment config; dialogue context
    is not used (it has no chat interface). Output is already normalized.
    """

    def __init__(self, key: str, model: str):
        self.key = key
        self._model_name = model
        self._t: NllbTranslator | None = None

    def _lazy(self) -> NllbTranslator:
        if self._t is None:
            self._t = NllbTranslator(target="uz", model=self._model_name)
        return self._t

    def translate(self, text: str, context: str | None = None) -> str:
        return self._lazy().translate(text)


# ── OpenAI-compatible chat (Ollama local & cloud) ────────────────────────────
class OpenAIChatSystem(System):
    """Any OpenAI-compatible chat endpoint — used for all Ollama models."""

    def __init__(self, key: str, model: str, base_url: str, api_key: str):
        self.key = key
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _lazy(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def translate(self, text: str, context: str | None = None) -> str:
        client = self._lazy()
        prompt = _context_preamble(context) + build_prompt(text)
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return normalize_uz((resp.choices[0].message.content or "").strip())


# ── Gemini (Google GenAI) ────────────────────────────────────────────────────
class GeminiSystem(System):
    def __init__(self, key: str, model: str = GEMINI_MODEL):
        self.key = key
        self._model = model
        self._client = None

    def _lazy(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def translate(self, text: str, context: str | None = None) -> str:
        from google.genai import types

        client = self._lazy()
        prompt = _context_preamble(context) + build_prompt(text)
        resp = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=GEMINI_THINKING_BUDGET
                ),
            ),
        )
        return normalize_uz((resp.text or "").strip())


# ── HF causal LM (local, Uzbek-tuned models: NeuronAI-Uzbek, behbudiy) ────────
class HFLocalSystem(System):
    """Local HuggingFace causal LM used through its chat template.

    Uses the SAME fixed translation prompt as every other LLM system (identical
    prompt across systems for fair comparison), applied as the user turn.
    Qwen3-based models (NeuronAI-Uzbek) get enable_thinking=False; greedy decode
    with repetition guards, then <think> is stripped and script is normalized.
    """

    def __init__(self, key: str, model: str):
        self.key = key
        self._model_name = model
        self._tok = None
        self._model = None
        self._torch = None

    def _lazy(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
        return self._tok, self._model

    def translate(self, text: str, context: str | None = None) -> str:
        tok, model = self._lazy()
        prompt = _context_preamble(context) + build_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,           # Qwen3 thinking off; ignored elsewhere
            )
        except TypeError:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tok(rendered, return_tensors="pt").to(model.device)
        with self._torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=HF_MAX_NEW, do_sample=False,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=HF_REP_PENALTY,
                no_repeat_ngram_size=HF_NO_REPEAT_NGRAM,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return normalize_uz(_STRIP_THINK.sub("", gen).strip())


# ── registry ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Spec:
    key: str
    kind: str                       # nllb | ollama_local | ollama_cloud | gemini
    model: str
    extra: dict = field(default_factory=dict)


SPECS: list[Spec] = [
    Spec("nllb-1.3b", "nllb", "facebook/nllb-200-distilled-1.3B"),
    Spec("nllb-3.3b", "nllb", "facebook/nllb-200-3.3B"),
    Spec("gemma4-12b", "ollama_local", "gemma4:12b"),
    Spec("gemma4-26b", "ollama_local", "gemma4:26b"),
    Spec("gemma4-31b-cloud", "ollama_cloud", "gemma4:31b-cloud"),
    Spec("translategemma-12b", "ollama_local", "translategemma:12b"),
    Spec("translategemma-27b", "ollama_local", "translategemma:27b"),
    Spec("gemini-3.5-flash", "gemini", GEMINI_MODEL),
    # Uzbek-specialized HF causal LM
    Spec("neuronai-uzbek", "hf_local", "NeuronUz/NeuronAI-Uzbek"),
]

SPEC_BY_KEY = {s.key: s for s in SPECS}
QUALITY_BOARD = [s.key for s in SPECS]                       # all 9


def _ollama_local_url() -> str:
    return os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434/v1")


def _ollama_cloud_url() -> str:
    return os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com/v1")


def build_system(key: str) -> System:
    spec = SPEC_BY_KEY[key]
    if spec.kind == "nllb":
        return NllbSystem(key, spec.model)
    if spec.kind == "ollama_local":
        # local Ollama ignores the key; any non-empty string works
        return OpenAIChatSystem(key, spec.model, _ollama_local_url(), "ollama")
    if spec.kind == "ollama_cloud":
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY required for the Ollama Cloud model")
        return OpenAIChatSystem(key, spec.model, _ollama_cloud_url(), api_key)
    if spec.kind == "gemini":
        return GeminiSystem(key, spec.model)
    if spec.kind == "hf_local":
        return HFLocalSystem(key, spec.model)
    raise ValueError(f"unknown system kind {spec.kind!r}")


def resolve_keys(arg: str | None) -> list[str]:
    """'all' / None -> every system; else a comma-separated list of keys."""
    if not arg or arg == "all":
        return list(QUALITY_BOARD)
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    for k in keys:
        if k not in SPEC_BY_KEY:
            raise ValueError(f"unknown system {k!r}; known: {list(SPEC_BY_KEY)}")
    return keys
