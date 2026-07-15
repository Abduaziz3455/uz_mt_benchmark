"""Phase 3 — XCOMET-XL scoring (primary automated adequacy).

XCOMET-XL (Unbabel/XCOMET-XL, 3.5B) is the current large, up-to-date neural MT
metric. It runs in two modes from the same checkpoint:

  * QE (reference-free):  score(src=en, mt=uz)            -> all systems, in-domain
  * reference mode:       score(src=en, mt=uz, ref=human) -> on the human refsets

QE is the unbiased backbone that decides the in-domain ranking (no references
needed). Reference mode is the cross-check on FLORES/NTREX/turkic_xwmt.

Needs `pip install "unbabel-comet>=2.2.0"`, a GPU, and `huggingface-cli login`
(XCOMET-XL is a gated repo). XXL (Unbabel/XCOMET-XXL, 10.7B) via --model if VRAM
allows.

Usage (in-domain QE, all systems):
    python -m uz_mt_bench.comet_qe \
        --bench data/eval/uz_mt_benchmark.jsonl \
        --candidates data/eval/candidates/uz_mt_benchmark

    # reference mode on a human refset (needs the refset's own uz_text as reference)
    python -m ...comet_qe --bench data/eval/refsets/ntrex.jsonl \
        --candidates data/eval/candidates/ntrex --reference

Output: data/eval/scores/xcomet_<tag>.jsonl  (per row: id, system, score[, span_count])
        + per-system and per-category means printed.

LONG-SEGMENT CHUNKING
---------------------
XCOMET's encoder (XLM-R) truncates at 512 subword tokens. Anything past the cut
is invisible to the metric, so a long-but-correct translation scores near zero:
in the first full run, segments over 200 EN words averaged QE 0.295 *for every
system*, which flattened the leaderboard (shared floor scores compress the
between-system gaps). We therefore split any source over --max-words into
sentence-aligned chunks, score each chunk, and report the source-word-weighted
mean. Pass --no-chunk to reproduce the old truncating behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_MODEL = "Unbabel/XCOMET-XL"
DEFAULT_MAX_WORDS = 80          # ~ safely under XLM-R's 512-subword window

_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def _load_bench(path: Path) -> dict[str, dict]:
    out = {}
    skipped = 0
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  !! {path.name}:{n} unparseable, skipping ({exc})")
                skipped += 1
                continue
            out[o["id"]] = {
                "en_text": o["en_text"],
                "uz_ref": o.get("uz_text"),          # present only for refsets
                "category": o.get("category", o.get("domain", "all")),
            }
    if skipped:
        print(f"  !! {skipped} malformed line(s) in {path} were skipped")
    return out


def _iter_candidates(path: Path):
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── long-segment chunking ────────────────────────────────────────────────────
def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p and p.strip()]
    return parts or [text.strip()]


def _group_by_words(sents: list[str], max_words: int) -> list[str]:
    """Greedily pack sentences into chunks of <= max_words (never splits a sentence)."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for s in sents:
        w = len(s.split())
        if cur and cur_w + w > max_words:
            chunks.append(" ".join(cur))
            cur, cur_w = [], 0
        cur.append(s)
        cur_w += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_k(sents: list[str], k: int) -> list[str]:
    """Split a sentence list into exactly k contiguous, non-empty, char-balanced groups.

    Used to cut the translation at sentence boundaries so its chunks line up with
    the source's. Assumes k <= len(sents).
    """
    k = max(1, min(k, len(sents)))
    if k == 1:
        return [" ".join(sents)]
    lens = [len(s) for s in sents]
    total = float(sum(lens)) or 1.0
    groups: list[str] = []
    cur: list[str] = []
    acc = 0.0
    for i, (s, ln) in enumerate(zip(sents, lens)):
        cur.append(s)
        acc += ln
        remaining = len(sents) - i - 1          # sentences after this one
        still_needed = k - len(groups) - 1      # groups to open after closing this one
        if len(groups) < k - 1 and (acc / total >= (len(groups) + 1) / k
                                    or remaining <= still_needed):
            groups.append(" ".join(cur))
            cur = []
    if cur:
        groups.append(" ".join(cur))
    return groups


def _chunk_pair(src: str, mt: str, ref: str | None,
                max_words: int) -> list[tuple[str, str, str | None]]:
    """Sentence-align src/mt(/ref) into parallel chunks small enough to encode."""
    if len(src.split()) <= max_words:
        return [(src, mt, ref)]
    src_sents = _sentences(src)
    mt_sents = _sentences(mt)
    if len(src_sents) < 2 or len(mt_sents) < 2:
        return [(src, mt, ref)]                 # nothing to align on; score as-is

    src_chunks = _group_by_words(src_sents, max_words)
    # can't produce more chunks than the translation has sentences to give
    k = min(len(src_chunks), len(mt_sents))
    if ref is not None:
        k = min(k, len(_sentences(ref)))
    if k < 2:
        return [(src, mt, ref)]
    if k != len(src_chunks):
        src_chunks = _split_k(src_sents, k)
    mt_chunks = _split_k(mt_sents, k)
    ref_chunks = _split_k(_sentences(ref), k) if ref is not None else [None] * k
    return list(zip(src_chunks, mt_chunks, ref_chunks))


def _load_model(model_name: str, batch_size: int, gpus: int):
    import torch
    from comet import download_model, load_from_checkpoint

    # A100 Tensor Cores: enable TF32 for float32 matmuls. Silences Lightning's
    # perf warning; ~1.5-2x faster scoring with a negligible effect on XCOMET
    # values (well below the score deltas between systems). 'high'=TF32 keeps
    # more precision than 'medium'=bf16 — the right trade for a scoring metric.
    torch.set_float32_matmul_precision("high")

    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--candidates", default="data/eval/candidates/uz_mt_benchmark")
    ap.add_argument("--out-dir", default="data/eval/scores")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reference", action="store_true",
                    help="reference mode: use the refset's uz_text as reference")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                    help="split sources longer than this into sentence-aligned chunks")
    ap.add_argument("--no-chunk", action="store_true",
                    help="disable chunking (old behaviour: XCOMET truncates long inputs)")
    args = ap.parse_args()

    bench_path = Path(args.bench)
    bench = _load_bench(bench_path)
    cand_dir = Path(args.candidates)
    files = sorted(cand_dir.glob("*.jsonl"))
    if not files:
        print(f"no candidate files in {cand_dir}")
        return

    # Build one flat scoring batch across all systems (efficient single GPU pass),
    # remembering which (system, id, category) each *chunk* belongs to plus its
    # source-word weight, so chunks can be recombined into one score per segment.
    data, meta = [], []
    n_chunked = 0
    empties: dict[str, int] = defaultdict(int)
    for f in files:
        system = f.stem
        # A retried segment appends a second row; keep the last non-empty one so a
        # stale empty (or a duplicate) can't be averaged into the fresh translation.
        latest: dict[str, dict] = {}
        for cand in _iter_candidates(f):
            if (cand.get("uz_text") or "").strip() or cand["id"] not in latest:
                latest[cand["id"]] = cand
        for cand in latest.values():
            b = bench.get(cand["id"])
            uz = (cand.get("uz_text") or "").strip()
            if b is not None and not uz:
                empties[system] += 1
            if b is None or not uz:
                continue
            ref = b.get("uz_ref") if args.reference else None
            if args.reference and not ref:
                continue
            if args.no_chunk:
                pieces = [(b["en_text"], uz, ref)]
            else:
                pieces = _chunk_pair(b["en_text"], uz, ref, args.max_words)
            if len(pieces) > 1:
                n_chunked += 1
            for src_c, mt_c, ref_c in pieces:
                item = {"src": src_c, "mt": mt_c}
                if args.reference:
                    item["ref"] = ref_c
                data.append(item)
                meta.append((system, cand["id"], b["category"],
                             max(1, len(src_c.split())), len(pieces)))

    if not data:
        print("nothing to score (no non-empty candidates matched the benchmark)")
        return

    if empties:
        print("  !! empty translations EXCLUDED from the mean (rerun translate_all "
              "to fill them; an unscored failure flatters the system):")
        for s, n in sorted(empties.items(), key=lambda kv: -kv[1]):
            print(f"       {s}: {n}")

    print(f"scoring {len(data)} rows with {args.model} "
          f"({'reference' if args.reference else 'QE'} mode) ...")
    if not args.no_chunk:
        print(f"  chunking: {n_chunked} segments exceeded {args.max_words} words "
              f"and were sentence-aligned into multiple chunks")
    model = _load_model(args.model, args.batch_size, args.gpus)
    out = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
    scores = out["scores"] if isinstance(out, dict) else out.scores

    tag = args.tag or bench_path.stem
    mode = "ref" if args.reference else "qe"
    out_path = Path(args.out_dir) / f"xcomet_{mode}_{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # recombine chunks: source-word-weighted mean per (system, id)
    agg: dict[tuple[str, str], dict] = {}
    for (system, sid, cat, weight, n_chunks), sc in zip(meta, scores):
        rec = agg.setdefault((system, sid), {"cat": cat, "num": 0.0, "den": 0.0,
                                             "n_chunks": n_chunks})
        rec["num"] += sc * weight
        rec["den"] += weight

    by_system: dict[str, list[float]] = defaultdict(list)
    by_sys_cat: dict[tuple[str, str], list[float]] = defaultdict(list)
    with out_path.open("w") as fh:
        for (system, sid), rec in agg.items():
            sc = rec["num"] / rec["den"]
            fh.write(json.dumps({"id": sid, "system": system, "score": sc,
                                 "n_chunks": rec["n_chunks"]}) + "\n")
            by_system[system].append(sc)
            by_sys_cat[(system, rec["cat"])].append(sc)

    print(f"\nwrote {out_path}")
    print(f"\n{'system':20s} {'n':>4} {'mean':>7}")
    print("-" * 34)
    for system in sorted(by_system, key=lambda s: -sum(by_system[s]) / len(by_system[s])):
        vals = by_system[system]
        print(f"{system:20s} {len(vals):>4} {sum(vals)/len(vals):>7.4f}")

    # per-category breakdown (drives the real decision)
    cats = sorted({c for _, c in by_sys_cat})
    print(f"\nper-category means:\n{'system':20s} " + " ".join(f"{c[:10]:>10}" for c in cats))
    for system in sorted(by_system):
        cells = []
        for c in cats:
            vals = by_sys_cat.get((system, c))
            cells.append(f"{sum(vals)/len(vals):>10.3f}" if vals else f"{'-':>10}")
        print(f"{system:20s} " + " ".join(cells))


if __name__ == "__main__":
    main()
