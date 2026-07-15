"""Production EN->UZ translator: NLLB-200 + entity-masking + sentence-splitting.

Proven config from the Phase 4a eval (22/22 on the scenario suite). Reused by
both the eval harness and the dataset translation CLI.

  * entity-masking  -> numbers / phones / URLs / {{vars}} / codes survive verbatim
  * sentence-split  -> NLLB is sentence-level; compound inputs don't get truncated
  * per-sentence cache (on masked text) -> high reuse across repetitive call lines

License note: NLLB-200 is CC-BY-NC-4.0. Translated output inherits a
non-commercial restriction (accepted for this project).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .entity_mask import mask, unmask, unrestored

DEFAULT_MODEL = "facebook/nllb-200-distilled-1.3B"
LANG_CODES = {                     # dataset lang -> NLLB FLORES code
    "uz": "uzn_Latn",
    "ru": "rus_Cyrl",
    "kk": "kaz_Cyrl",
    "en": "eng_Latn",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# Apostrophe variants NLLB emits inconsistently for Uzbek. Canonical Uzbek Latin
# uses oʻ / gʻ with U+02BB (modifier letter turned comma) and the glottal-stop
# "tutuq belgisi" U+02BC. Standardize so TTS and tokenization stay consistent.
_TURNED_COMMA = "ʻ"   # ʻ  -> for oʻ / gʻ
_TUTUQ = "ʼ"          # ʼ  -> glottal stop (maʼno)
# All apostrophe variants (for the o/g rule).
_APOS_ALL = "['‘’ʼʻ`´′‵]"
# Glottal rule must NOT re-touch the canonical chars (ʻ, ʼ) it/we set.
_APOS_VARIANT = "['‘’`´′‵]"
_OG_APOS = re.compile(r"([oOgG])" + _APOS_ALL)
_GLOTTAL = re.compile(r"(?<=[a-zA-Z])" + _APOS_VARIANT + r"(?=[a-zA-Z])")


def normalize_uz(text: str) -> str:
    """Standardize Uzbek apostrophes: oʻ/gʻ -> U+02BB, in-word glottal -> U+02BC."""
    text = _OG_APOS.sub(r"\1" + _TURNED_COMMA, text)   # oʻ / gʻ first (canonical)
    text = _GLOTTAL.sub(_TUTUQ, text)                  # remaining maʼno / saʼva
    return text


class NllbTranslator:
    def __init__(self, target: str = "uz", *, model: str = DEFAULT_MODEL,
                 cache_dir: Path | None = None, num_beams: int = 4,
                 max_new: int = 256, src: str = "eng_Latn"):
        if target not in LANG_CODES:
            raise ValueError(f"unsupported target lang {target!r}")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.target, self.src = target, src
        self.tgt_code = LANG_CODES[target]
        self.num_beams, self.max_new = num_beams, max_new
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, src_lang=src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto").eval()
        self.tgt_id = self.tok.convert_tokens_to_ids(self.tgt_code)

        self._cache: dict[str, str] = {}
        self.cache_path = (cache_dir / f"nllb_{target}.json") if cache_dir else None
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())
        self.stats = {"calls": 0, "cache_hits": 0, "cyrillic_warn": 0, "dropped_entities": 0}

    # ── core ────────────────────────────────────────────────────────────────
    def _key(self, masked_sentence: str) -> str:
        return hashlib.sha1(f"{self.tgt_code}|{masked_sentence}".encode()).hexdigest()

    def _gen_batch(self, sentences: list[str]) -> list[str]:
        """Translate a batch of masked sentences in a single padded forward pass."""
        inp = self.tok(sentences, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_new).to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inp, forced_bos_token_id=self.tgt_id,
                                       max_new_tokens=self.max_new, num_beams=self.num_beams)
        self.stats["calls"] += len(sentences)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def _gen(self, masked_sentence: str) -> str:
        key = self._key(masked_sentence)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]
        text = self._gen_batch([masked_sentence])[0]
        self._cache[key] = text
        return text

    def _finalize(self, sents: list[str], translated: list[str],
                  mapping: dict[str, str]) -> str:
        out = " ".join(translated) if translated else ""
        if self.target == "uz":
            out = normalize_uz(out)
        out = unmask(out, mapping).strip()
        if _CYRILLIC.search(out) and self.target == "uz":
            self.stats["cyrillic_warn"] += 1
        if mapping and unrestored(out, mapping):
            self.stats["dropped_entities"] += 1
        return out

    def translate(self, text: str, *, do_mask: bool = True) -> str:
        """Translate one English string to the target language.

        do_mask=False bypasses entity masking — used to measure the masking
        layer's impact on surrounding text quality.
        """
        if not text or not text.strip():
            return text
        masked, mapping = mask(text) if do_mask else (text, {})
        sents = [s for s in _SENT_SPLIT.split(masked.strip()) if s]
        return self._finalize(sents, [self._gen(s) for s in sents], mapping)

    def translate_many(self, texts: list[str], *, batch_size: int = 32,
                       do_mask: bool = True) -> list[str]:
        """Batch-translate many strings, maximizing GPU utilization.

        Masks + sentence-splits every input, pools all uncached sentences into
        padded batches for the GPU, then reassembles each output (normalize +
        unmask). Identical results to translate(), just far higher throughput.
        """
        plans: list[tuple[dict[str, str], list[str]] | None] = []
        pending: list[str] = []          # unique uncached masked sentences
        seen: set[str] = set()
        for t in texts:
            if not t or not t.strip():
                plans.append(None)
                continue
            masked, mapping = mask(t) if do_mask else (t, {})
            sents = [s for s in _SENT_SPLIT.split(masked.strip()) if s]
            plans.append((mapping, sents))
            for s in sents:
                if self._key(s) in self._cache:
                    self.stats["cache_hits"] += 1
                elif s not in seen:
                    seen.add(s)
                    pending.append(s)

        # Length-bucket before batching: sort uncached sentences by length so each
        # padded batch is uniform. A batch of all-long sentences would otherwise spike
        # VRAM past the 12 GB budget (the corpus has 100+-turn dialogues); bucketing
        # caps the peak and eliminates padding waste (also faster). Cache is keyed, so
        # processing order does not affect results.
        pending.sort(key=len)
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            for s, tr in zip(chunk, self._gen_batch(chunk)):
                self._cache[self._key(s)] = tr

        out: list[str] = []
        for plan, t in zip(plans, texts):
            if plan is None:
                out.append(t)
                continue
            mapping, sents = plan
            translated = [self._cache[self._key(s)] for s in sents]
            out.append(self._finalize(sents, translated, mapping))
        return out

    def flush(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False))
