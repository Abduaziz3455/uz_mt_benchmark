"""Diagnostic — list the exact preserve[] tokens each system dropped.

structural_checks reports entity_keep as a rate; this shows *which* entities
were lost and in *what* output, so you can eyeball whether a call-agent-critical
token (WEBSITEURL, phone, code) went missing. Joins bench (en_text + preserve[])
to each system's candidate file by id.

Duplicate ids in a candidate file (e.g. leftover rows from a resumed translate
run) are collapsed to the last-written row, and the dup count is reported — that
also explains an inflated `n` in the structural summary.

Usage:
    python -m uz_mt_bench.dropped_entities \
        --bench data/eval/uz_mt_benchmark.smoke.jsonl \
        --candidates data/eval/candidates/uz_mt_benchmark.smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_bench(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            out[o["id"]] = {"en_text": o["en_text"], "preserve": o.get("preserve", [])}
    return out


def _load_candidates(path: Path) -> tuple[dict[str, str], int]:
    """id -> uz_text (last wins). Returns (map, duplicate_row_count)."""
    seen: dict[str, str] = {}
    dups = 0
    with path.open() as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = o.get("id")
            if cid in seen:
                dups += 1
            seen[cid] = o.get("uz_text", "")
    return seen, dups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--candidates", default="data/eval/candidates/uz_mt_benchmark")
    ap.add_argument("--max-out", type=int, default=200,
                    help="max Uzbek output chars to print per row")
    args = ap.parse_args()

    bench = _load_bench(Path(args.bench))
    files = sorted(Path(args.candidates).glob("*.jsonl"))
    if not files:
        print(f"no candidate files in {args.candidates}")
        return

    for f in files:
        system = f.stem
        cand, dups = _load_candidates(f)
        rows_with_ent = 0
        offenders = []
        for cid, uz in cand.items():
            b = bench.get(cid)
            if b is None or not b["preserve"]:
                continue
            rows_with_ent += 1
            missing = [tok for tok in b["preserve"] if tok not in uz]
            if missing:
                offenders.append((cid, missing, b["en_text"], uz))

        print(f"\n=== {system} ===")
        note = f" ({dups} duplicate rows collapsed)" if dups else ""
        print(f"unique ids: {len(cand)}{note} · segments with entities: {rows_with_ent} "
              f"· dropped in: {len(offenders)}")
        for cid, missing, en, uz in offenders:
            uz_short = uz[: args.max_out] + ("…" if len(uz) > args.max_out else "")
            print(f"  [{cid}] MISSING {missing}")
            print(f"      en: {en[: args.max_out]}")
            print(f"      uz: {uz_short}")


if __name__ == "__main__":
    main()
