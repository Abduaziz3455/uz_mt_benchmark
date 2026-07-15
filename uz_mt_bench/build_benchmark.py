"""Phase 0a — build the stratified EN->UZ evaluation set.

Samples turn-level English segments from the SFT corpus, stratified by source
section (12) and text category (5), labels each with the high-value entities
that must survive translation, and writes a frozen, deterministic benchmark.

Categories (priority order, most specific first):
  edge        - bitext ALLCAPS placeholders / profanity  (rare, must-sample)
  tool_spoken - contains masked entities (numbers, URLs, IDs, times)
  kb_passage  - long informational answer (>= KB_MIN_WORDS words)
  agent_reply - assistant turn (default)
  caller_turn - user turn (default)

Every scored unit is one turn (self-contained), with the previous turn kept as
optional context. Sampling is reservoir-based per (section, category) bucket so
memory stays bounded and the result is reproducible for a fixed seed.

Usage:
    python -m uz_mt_bench.build_benchmark
    python -m uz_mt_bench.build_benchmark \
        --in data/train.sft.jsonl --out data/eval/uz_mt_benchmark.jsonl \
        --n 600 --smoke 60 --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .entity_mask import mask

CATEGORIES = ["caller_turn", "agent_reply", "kb_passage", "tool_spoken", "edge"]
KB_MIN_WORDS = 40          # assistant turn this long counts as a KB passage
MIN_WORDS = 3              # drop shorter fragments (except edge)
POOL_CAP = 3000            # reservoir size per (section, category) bucket
MAX_WORDS = 400            # skip pathologically long turns (p99 is ~ a few hundred)

_PLACEHOLDER = re.compile(r"\b[A-Z]{6,}\b")     # bitext WEBSITEURL / CLAIMSECTION / ...
_PROFANITY = re.compile(
    r"\b(fuck\w*|shit\w*|damn|bitch\w*|asshole\w*|crap)\b", re.IGNORECASE
)
# A grouped/decimal amount ($40,000, 14.55) is NOT a byte-for-byte survivor: a
# faithful Uzbek translation reformats it (decimal comma, space thousands sep, or
# spells it out). It only round-trips verbatim behind the mask/unmask pipeline,
# which the benchmark deliberately does not apply. Keeping such tokens in
# preserve[] penalizes correctly-localized output, so we drop them from the gate
# set — the truly-opaque entities (URLs, e-mails, phones, {{vars}}, digit IDs)
# stay. See structural_checks entity_keep.
_AMOUNT = re.compile(r"\d{1,3}(?:[.,]\d+)+")


def _verbatim_entities(entities: list[str]) -> list[str]:
    """Drop localizable numeric amounts; keep opaque must-survive tokens."""
    return [e for e in entities if not _AMOUNT.fullmatch(e)]


def _categorize(role: str, text: str, entities: list[str], placeholders: list[str]) -> str:
    """Assign a category by content signals, then fall back to the turn role."""
    if placeholders or _PROFANITY.search(text):
        return "edge"
    if entities:
        return "tool_spoken"
    if role == "assistant" and len(text.split()) >= KB_MIN_WORDS:
        return "kb_passage"
    return "agent_reply" if role == "assistant" else "caller_turn"


def _iter_segments(path: Path):
    """Yield (section, domain, dialogue_idx, turn_idx, role, text, prev_text)."""
    with path.open() as fh:
        for dialogue_idx, line in enumerate(fh):
            obj = json.loads(line)
            section = obj.get("source", "unknown")
            domain = obj.get("domain", "")
            prev_text = None
            for turn_idx, msg in enumerate(obj["messages"]):
                role, text = msg["role"], (msg.get("content") or "").strip()
                if role == "system" or not text:
                    prev_text = text or prev_text
                    continue
                yield section, domain, dialogue_idx, turn_idx, role, text, prev_text
                prev_text = text


def build(in_path: Path, n: int, seed: int):
    rng = random.Random(seed)
    # Reservoir per (section, category); count tracks items seen for that bucket.
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    seen_text: set[str] = set()

    for section, domain, d_idx, t_idx, role, text, prev in _iter_segments(in_path):
        words = text.split()
        key_norm = " ".join(words).lower()
        if key_norm in seen_text:
            continue

        entities = list(mask(text)[1].values())
        placeholders = sorted(set(_PLACEHOLDER.findall(text)))
        category = _categorize(role, text, entities, placeholders)

        if len(words) > MAX_WORDS:
            continue
        if len(words) < MIN_WORDS and category != "edge":
            continue

        seen_text.add(key_norm)
        rec = {
            "id": f"{section}-{d_idx:06d}-t{t_idx}",
            "section": section,
            "domain": domain,
            "category": category,
            "en_text": text,
            "context_prev": prev,
            # preserve = opaque entities + bitext placeholders (verbatim-survival
            # set); localizable amounts are excluded (see _verbatim_entities).
            "preserve": sorted(set(_verbatim_entities(entities)) | set(placeholders)),
            "char_len": len(text),
            "word_len": len(words),
        }

        bucket = (section, category)
        counts[bucket] += 1
        pool = pools[bucket]
        if len(pool) < POOL_CAP:
            pool.append(rec)
        else:  # reservoir replacement
            j = rng.randint(0, counts[bucket] - 1)
            if j < POOL_CAP:
                pool[j] = rec

    return _stratified_sample(pools, n, rng)


def _stratified_sample(pools: dict[tuple[str, str], list[dict]], n: int, rng) -> list[dict]:
    """Balance across sections, then round-robin across categories within each."""
    sections = sorted({s for s, _ in pools})
    per_section = max(1, n // len(sections))

    # Shuffle every bucket once (deterministic given seed) so pops are unbiased.
    for pool in pools.values():
        rng.shuffle(pool)

    chosen: list[dict] = []
    for section in sections:
        cats = [c for c in CATEGORIES if pools.get((section, c))]
        if not cats:
            continue
        cursors = {c: 0 for c in cats}
        picked, i = 0, 0
        while picked < per_section:
            cat = cats[i % len(cats)]
            i += 1
            pool = pools[(section, cat)]
            cur = cursors[cat]
            if cur < len(pool):
                chosen.append(pool[cur])
                cursors[cat] += 1
                picked += 1
            elif all(cursors[c] >= len(pools[(section, c)]) for c in cats):
                break  # section exhausted
    rng.shuffle(chosen)
    return chosen[:n]


def _write(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _report(records: list[dict], label: str):
    by_section = Counter(r["section"] for r in records)
    by_cat = Counter(r["category"] for r in records)
    print(f"\n{label}: {len(records)} segments")
    print("  by section:", dict(sorted(by_section.items())))
    print("  by category:", dict(sorted(by_cat.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="data/train.sft.jsonl")
    ap.add_argument("--out", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--smoke", type=int, default=60, help="also write an N-segment smoke subset")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    print(f"building benchmark from {in_path} (n={args.n}, seed={args.seed}) ...")

    records = build(in_path, args.n, args.seed)
    _write(records, out_path)
    _report(records, f"wrote {out_path}")

    if args.smoke:
        # Deterministic per-section slice for the smoke subset.
        rng = random.Random(args.seed + 1)
        by_section: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_section[r["section"]].append(r)
        per = max(1, args.smoke // len(by_section))
        smoke: list[dict] = []
        for section in sorted(by_section):
            pool = by_section[section][:]
            rng.shuffle(pool)
            smoke.extend(pool[:per])
        rng.shuffle(smoke)
        smoke = smoke[: args.smoke]
        smoke_path = out_path.with_suffix(".smoke.jsonl")
        _write(smoke, smoke_path)
        _report(smoke, f"wrote {smoke_path}")


if __name__ == "__main__":
    main()
