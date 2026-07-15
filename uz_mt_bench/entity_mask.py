"""Entity masking for translation fidelity.

Voice-agent text is full of tokens that MUST survive translation byte-for-byte:
phone numbers, codes, amounts, URLs, e-mails, and {{placeholder}} variables.
NMT/LLM translators routinely localize or hallucinate these. The fix is to
replace each with an opaque sentinel BEFORE translation and restore it AFTER:

    mask("Call +998 71 200 70 07")  -> ("Call ⟦0⟧", {"⟦0⟧": "+998 71 200 70 07"})
    unmask("⟦0⟧ ga qoʻngʻiroq", map) -> "+998 71 200 70 07 ga qoʻngʻiroq"

Sentinels are chosen to be copy-through friendly and the restore step is
whitespace-tolerant in case the model nudges spacing around the brackets.
"""

from __future__ import annotations

import re

# Order matters: most specific first so a phone isn't split by the number rule.
# NOTE: we deliberately DO NOT mask short bare integers (1-2 digits). NLLB
# translates small inline numbers in date/time idioms ("half past 6", "March
# 10th") correctly on its own; masking them breaks the idiom and the model drops
# or mutates the sentinel. Only high-value entities — phones, URLs, e-mails,
# {{vars}}, grouped/decimal amounts, and >=3-digit codes/IDs — get masked.
_PATTERNS = [
    re.compile(r"\{\{.*?\}\}"),                       # {{placeholder}}
    re.compile(r"https?://[^\s]+"),                   # URLs
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),          # e-mails
    re.compile(r"\+?\d[\d\s()\-]{5,}\d"),             # phone numbers
    re.compile(r"\d{1,3}(?:[.,]\d+)+"),               # grouped/decimal amounts: 47.50, 1,250,000
    re.compile(r"\d{3,}"),                            # long codes / IDs (>=3 digits)
]


def _sentinel(i: int) -> str:
    """Digit-free, uppercase, copy-through-friendly sentinel.

    NMT models strip bracket punctuation (⟦ ⟧) and translate bare digits, so the
    sentinel must be an opaque alpha token the model copies verbatim. Index is
    base-26 letters: 0->A, 1->B, ... 26->BA. 'Xq' delimiters keep it from fusing
    with adjacent words while staying a single copyable unit.
    """
    n, s = i, ""
    while True:
        s = chr(65 + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            break
    return f"Xq{s}qX"


def _index_re(token: str) -> str:
    """Loose regex for a sentinel, tolerant of case / spacing the model may add."""
    core = token[2:-2]  # strip Xq ... qX
    return r"[Xx]q\s*" + r"\s*".join(re.escape(c) for c in core) + r"\s*q[Xx]"


def mask(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected entities with sentinels. Returns (masked, mapping)."""
    spans: list[tuple[int, int, str]] = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), m.group()))

    # Resolve overlaps: earliest start wins, then longest match.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, g in spans:
        if s >= last_end:
            chosen.append((s, e, g))
            last_end = e

    out: list[str] = []
    mapping: dict[str, str] = {}
    prev = 0
    for idx, (s, e, g) in enumerate(chosen):
        out.append(text[prev:s])
        token = _sentinel(idx)
        out.append(token)
        mapping[token] = g
        prev = e
    out.append(text[prev:])
    return "".join(out), mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore sentinels. Tolerant of case/spacing the model may introduce."""
    for token, value in mapping.items():
        if token in text:
            text = text.replace(token, value)
        else:
            text = re.sub(_index_re(token), value, text)
    return text


def unrestored(text: str, mapping: dict[str, str]) -> list[str]:
    """Return entity values whose sentinel never made it back (lost in translation)."""
    return [value for value in mapping.values() if value not in text]
