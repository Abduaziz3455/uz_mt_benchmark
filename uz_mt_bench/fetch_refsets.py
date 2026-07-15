"""Phase 0b — fetch external HUMAN EN->UZ reference sets.

These are the unbiased, professionally/native-translated anchors for the
reference-based cross-check (chrF++ / reference-XCOMET / spBLEU). They replace
the dropped Gemini golden set (which would have been circular: Gemini is already
a system under test AND the judge). See docs/uz_mt_benchmark_plan.md §4.2.

Three sources, all normalized to a common schema and written one file each:

  refsets/ntrex.jsonl   NTREX-128   news (WMT19)   ~1997   raw GitHub files (stdlib)
  refsets/flores.jsonl  FLORES-200  wiki/news      ~1012   facebook/flores (HF, gated)
  refsets/xwmt.jsonl    turkic_xwmt talks+news     ~1-2k   turkic_xwmt (HF, remote code)

Common record: {"id","set","domain","en_text","uz_text"}  (uz_text normalized).

NTREX needs only stdlib + network. FLORES and turkic_xwmt need `datasets`
(`pip install datasets`) and, for FLORES, `huggingface-cli login` (gated repo).
Each source is independent — a failure in one is logged and the others proceed.

Usage:
    python -m uz_mt_bench.fetch_refsets            # all three
    python -m uz_mt_bench.fetch_refsets --only ntrex
    python -m uz_mt_bench.fetch_refsets --limit 50 # quick check
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from .nllb_translator import normalize_uz

NTREX_BASE = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
NTREX_EN = f"{NTREX_BASE}/newstest2019-src.eng.txt"
NTREX_UZ = f"{NTREX_BASE}/newstest2019-ref.uzb.txt"

FLORES_REPO = "facebook/flores"
FLORES_CONFIG = "eng_Latn-uzn_Latn"       # pair config → sentence_eng_Latn / sentence_uzn_Latn
FLORES_SPLIT = "devtest"

XWMT_REPO = "turkic-interlingua/turkic_xwmt"
XWMT_CONFIG = "en-uz"                       # translation.{en,uz}; split "test"


def _clean(en: str, uz: str) -> dict | None:
    en, uz = (en or "").strip(), (uz or "").strip()
    if not en or not uz:
        return None
    return {"en_text": en, "uz_text": normalize_uz(uz)}


# ── NTREX (stdlib, no heavy deps) ────────────────────────────────────────────
def fetch_ntrex(limit: int | None) -> list[dict]:
    def lines(url: str) -> list[str]:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8").splitlines()

    en_lines, uz_lines = lines(NTREX_EN), lines(NTREX_UZ)
    if len(en_lines) != len(uz_lines):
        print(f"  ! NTREX line mismatch: en={len(en_lines)} uz={len(uz_lines)} — zipping to min")
    out = []
    for i, (en, uz) in enumerate(zip(en_lines, uz_lines)):
        rec = _clean(en, uz)
        if rec:
            out.append({"id": f"ntrex-{i:05d}", "set": "ntrex", "domain": "news", **rec})
        if limit and len(out) >= limit:
            break
    return out


# ── FLORES-200 (HF, gated) ───────────────────────────────────────────────────
def fetch_flores(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    try:
        ds = load_dataset(FLORES_REPO, FLORES_CONFIG, split=FLORES_SPLIT)
        pairs = ((r["sentence_eng_Latn"], r["sentence_uzn_Latn"]) for r in ds)
    except Exception as exc:  # pair config unavailable → load two single configs and zip
        print(f"  (pair config failed: {str(exc)[:80]}; falling back to single-lang zip)")
        en = load_dataset(FLORES_REPO, "eng_Latn", split=FLORES_SPLIT)
        uz = load_dataset(FLORES_REPO, "uzn_Latn", split=FLORES_SPLIT)
        pairs = ((a["sentence"], b["sentence"]) for a, b in zip(en, uz))

    out = []
    for i, (en, uz) in enumerate(pairs):
        rec = _clean(en, uz)
        if rec:
            out.append({"id": f"flores-{i:05d}", "set": "flores", "domain": "wiki", **rec})
        if limit and len(out) >= limit:
            break
    return out


# ── turkic_xwmt (HF, script-based) ───────────────────────────────────────────
def fetch_xwmt(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    try:
        ds = load_dataset(XWMT_REPO, XWMT_CONFIG, split="test", trust_remote_code=True)
        get = lambda r: (r["translation"]["en"], r["translation"]["uz"])
    except Exception as exc:  # try the reversed pair config
        print(f"  (config {XWMT_CONFIG} failed: {str(exc)[:80]}; trying uz-en)")
        ds = load_dataset(XWMT_REPO, "uz-en", split="test", trust_remote_code=True)
        get = lambda r: (r["translation"]["en"], r["translation"]["uz"])

    out = []
    for i, row in enumerate(ds):
        en, uz = get(row)
        rec = _clean(en, uz)
        if rec:
            out.append({"id": f"xwmt-{i:05d}", "set": "xwmt", "domain": "talks_news", **rec})
        if limit and len(out) >= limit:
            break
    return out


SOURCES = {"ntrex": fetch_ntrex, "flores": fetch_flores, "xwmt": fetch_xwmt}


def _write(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/eval/refsets")
    ap.add_argument("--only", choices=list(SOURCES), help="fetch just one source")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per source (quick check)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    targets = [args.only] if args.only else list(SOURCES)
    summary = {}
    for name in targets:
        print(f"fetching {name} ...")
        try:
            recs = SOURCES[name](args.limit)
        except ImportError:
            print(f"  ✗ {name}: needs `pip install datasets` (skipped)")
            continue
        except Exception as exc:
            print(f"  ✗ {name}: {type(exc).__name__}: {str(exc)[:140]} (skipped)")
            continue
        path = out_dir / f"{name}.jsonl"
        _write(recs, path)
        summary[name] = len(recs)
        print(f"  ✓ wrote {path}: {len(recs)} pairs")

    print("\nsummary:", summary or "(nothing fetched)")


if __name__ == "__main__":
    main()
