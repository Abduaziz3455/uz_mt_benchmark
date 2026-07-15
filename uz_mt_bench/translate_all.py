"""Phase 1 — translate an input set through every system.

Generic over any JSONL with `id` + `en_text` (+ optional `context_prev`), so it
serves both the in-domain benchmark and the external reference sets:

    # in-domain (the 600-segment benchmark)
    python -m uz_mt_bench.translate_all
    # smoke first
    python -m ...translate_all --input data/eval/uz_mt_benchmark.smoke.jsonl
    # one system, or a subset
    python -m ...translate_all --systems nllb-1.3b,gemini-3.5-flash
    # a reference set (English side) for reference-based scoring
    python -m ...translate_all --input data/eval/refsets/ntrex.jsonl

Output: data/eval/candidates/<tag>/<system>.jsonl, one line per segment:
    {"id", "system", "uz_text", "latency_ms", "error"}

**Resumable:** already-translated ids in an existing candidate file are skipped,
so an interrupted run (OOM, rate limit, Ctrl-C) continues where it left off.
Each translation is flushed immediately.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from .systems import build_system, resolve_keys


def _load_input(path: Path) -> list[dict]:
    recs = []
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            recs.append(
                {
                    "id": o["id"],
                    "en_text": o["en_text"],
                    "context_prev": o.get("context_prev"),
                }
            )
    return recs


def _done_ids(path: Path) -> set[str]:
    """Ids already translated successfully — everything else is retried on rerun.

    A row counts as done only if it carries non-empty text. A generation timeout
    lands as {"uz_text": "", "error": null}, which used to look like success here
    (skipped forever on rerun) while comet_qe silently dropped it from the mean —
    so a system was effectively rewarded for failing to translate a segment.
    """
    if not path.exists():
        return set()
    ids = set()
    with path.open() as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not o.get("error") and (o.get("uz_text") or "").strip():
                ids.add(o["id"])
    return ids


def run_system(key: str, records: list[dict], out_path: Path, report_every: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [r for r in records if r["id"] not in done]
    print(f"\n[{key}] {len(done)} done, {len(todo)} to translate -> {out_path}")
    if not todo:
        return

    system = build_system(key)          # lazy backend init happens on first call
    ok, err, t0 = 0, 0, time.monotonic()
    with out_path.open("a") as fh:
        for i, rec in enumerate(todo, 1):
            row = {"id": rec["id"], "system": key}
            t = time.monotonic()
            try:
                uz = system.translate(rec["en_text"], rec.get("context_prev"))
                row["uz_text"] = uz
                row["latency_ms"] = round((time.monotonic() - t) * 1000)
                # an empty generation is a failure, not a success with no output
                row["error"] = None if uz.strip() else "EmptyGeneration"
                if uz.strip():
                    ok += 1
                else:
                    err += 1
            except Exception as exc:  # keep going; record the failure for a later retry
                row["uz_text"] = ""
                row["latency_ms"] = round((time.monotonic() - t) * 1000)
                row["error"] = f"{type(exc).__name__}: {exc}"
                err += 1
                if err <= 3:
                    traceback.print_exc()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % report_every == 0:
                rate = i / (time.monotonic() - t0)
                print(f"  [{key}] {i}/{len(todo)}  ok={ok} err={err}  {rate:.1f}/s")
    print(f"  [{key}] finished: ok={ok} err={err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--systems", default="all",
                    help="'all' (9) or a comma-separated list of system keys")
    ap.add_argument("--out-root", default="data/eval/candidates")
    ap.add_argument("--tag", default=None, help="subfolder name (default: input stem)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=25)
    args = ap.parse_args()

    in_path = Path(args.input)
    records = _load_input(in_path)
    if args.limit:
        records = records[: args.limit]
    tag = args.tag or in_path.stem
    out_dir = Path(args.out_root) / tag
    keys = resolve_keys(args.systems)

    print(f"input {in_path} ({len(records)} segments) -> {out_dir}")
    print(f"systems: {keys}")
    for key in keys:
        try:
            run_system(key, records, out_dir / f"{key}.jsonl", args.report_every)
        except Exception as exc:
            # a whole-system failure (e.g. missing OLLAMA_API_KEY, model not pulled)
            # shouldn't sink the other systems.
            print(f"  ✗ [{key}] aborted: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
