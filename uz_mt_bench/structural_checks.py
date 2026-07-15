"""Phase 2 — structural hard gates on candidate translations.

Cheap, no-GPU checks run on 100% of candidates. A fluent translation that drops
`WEBSITEURL`, leaks English, or drifts to Cyrillic fails the call-agent use case
regardless of its COMET score — this phase surfaces exactly that.

Per (segment, system):
  empty            - blank output
  copy_through     - output == source (untranslated)
  cyrillic         - contains Cyrillic (Uzbek here must be Latin)
  english_leak     - output looks like English (stopword heuristic, or langid)
  entity_keep      - fraction of preserve[] tokens present verbatim (1.0 = all kept)
  len_ratio        - char-length ratio uz/en (outlier if <0.3 or >3.0)
  degenerate       - n-gram repetition loop

Reads the benchmark (for en_text + preserve[]) joined by id with each system's
candidate file. Writes per-row results and prints a per-system summary.

Optional stronger language-ID: `pip install lingua-language-detector` is used
automatically if present; otherwise a dependency-free stopword heuristic runs.

Usage:
    python -m uz_mt_bench.structural_checks
    python -m ...structural_checks --bench data/eval/uz_mt_benchmark.jsonl \
        --candidates data/eval/candidates/uz_mt_benchmark --out data/eval/scores/structural.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# common English function words unlikely to survive a real Uzbek translation
_EN_STOP = {
    "the", "and", "you", "your", "for", "with", "that", "this", "have", "will",
    "please", "can", "are", "our", "not", "from", "would", "could", "should",
}
_LEN_LOW, _LEN_HIGH = 0.3, 3.0

# Optional: lingua language detector (loaded once if installed).
_LINGUA = None
try:  # pragma: no cover - optional dependency
    from lingua import Language, LanguageDetectorBuilder

    _LINGUA = (
        LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.UZBEK)
        .build()
    )
except Exception:
    _LINGUA = None


def _english_leak(uz: str, en: str) -> bool:
    if not uz:
        return False
    if _LINGUA is not None:
        lang = _LINGUA.detect_language_of(uz)
        return lang is not None and lang.name == "ENGLISH"
    toks = re.findall(r"[a-zA-Z']+", uz.lower())
    if not toks:
        return False
    hits = sum(1 for t in set(toks) if t in _EN_STOP)
    return hits >= 3          # heuristic: several English function words present


def _entity_keep(uz: str, preserve: list[str]) -> float | None:
    if not preserve:
        return None
    kept = sum(1 for tok in preserve if tok in uz)
    return kept / len(preserve)


def _degenerate(uz: str) -> bool:
    words = uz.split()
    if len(words) < 8:
        return False
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    top = Counter(trigrams).most_common(1)
    return bool(top) and top[0][1] >= 4        # same 3-gram >=4 times = loop


def check_row(en: str, uz: str, preserve: list[str]) -> dict:
    uz_s = (uz or "").strip()
    en_s = (en or "").strip()
    ek = _entity_keep(uz_s, preserve)
    lr = (len(uz_s) / len(en_s)) if en_s else 0.0
    return {
        "empty": uz_s == "",
        "copy_through": uz_s != "" and uz_s.lower() == en_s.lower(),
        "cyrillic": bool(_CYRILLIC.search(uz_s)),
        "english_leak": _english_leak(uz_s, en_s),
        "entity_keep": ek,                       # None if no entities to keep
        "len_ratio": round(lr, 3),
        "len_outlier": uz_s != "" and (lr < _LEN_LOW or lr > _LEN_HIGH),
        "degenerate": _degenerate(uz_s),
    }


def _load_bench(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            out[o["id"]] = {"en_text": o["en_text"], "preserve": o.get("preserve", [])}
    return out


def _iter_candidates(path: Path):
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _latest_candidates(path: Path) -> list[dict]:
    """One row per id, preferring the last non-empty one.

    translate_all appends on retry, so a segment that first came back empty and was
    later refilled has two rows. Counting both would gate the system on a translation
    it no longer produces. comet_qe.py resolves this the same way; the two must agree
    or the leaderboard's `empty` column will contradict its `n`.
    """
    latest: dict[str, dict] = {}
    for cand in _iter_candidates(path):
        if (cand.get("uz_text") or "").strip() or cand["id"] not in latest:
            latest[cand["id"]] = cand
    return list(latest.values())


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ent = [r["entity_keep"] for r in rows if r["entity_keep"] is not None]

    def rate(flag: str) -> float:
        return round(sum(1 for r in rows if r[flag]) / n, 4)

    passed = sum(
        1
        for r in rows
        if not (r["empty"] or r["copy_through"] or r["cyrillic"]
                or r["english_leak"] or r["degenerate"]
                or (r["entity_keep"] is not None and r["entity_keep"] < 1.0))
    )
    return {
        "n": n,
        "empty": rate("empty"),
        "copy_through": rate("copy_through"),
        "cyrillic": rate("cyrillic"),
        "english_leak": rate("english_leak"),
        "len_outlier": rate("len_outlier"),
        "degenerate": rate("degenerate"),
        "entity_keep_mean": round(sum(ent) / len(ent), 4) if ent else None,
        "entity_perfect_rate": round(sum(1 for e in ent if e == 1.0) / len(ent), 4) if ent else None,
        "pass_all_rate": round(passed / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--candidates", default="data/eval/candidates/uz_mt_benchmark",
                    help="dir of <system>.jsonl candidate files")
    ap.add_argument("--out", default="data/eval/scores/structural.jsonl")
    args = ap.parse_args()

    bench = _load_bench(Path(args.bench))
    cand_dir = Path(args.candidates)
    files = sorted(cand_dir.glob("*.jsonl"))
    if not files:
        print(f"no candidate files in {cand_dir}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_system: dict[str, list[dict]] = {}
    with out_path.open("w") as out_fh:
        for f in files:
            system = f.stem
            rows = []
            for cand in _latest_candidates(f):
                b = bench.get(cand["id"])
                if b is None:
                    continue
                res = check_row(b["en_text"], cand.get("uz_text", ""), b["preserve"])
                rec = {"id": cand["id"], "system": system, **res}
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(res)
            per_system[system] = rows

    print(f"lang-id backend: {'lingua' if _LINGUA else 'stopword-heuristic'}")
    print(f"wrote {out_path}\n")
    hdr = f"{'system':20s} {'n':>4} {'pass':>6} {'ent_kept':>8} {'copy':>5} {'cyr':>5} {'en_leak':>7} {'lenout':>6} {'degen':>5}"
    print(hdr)
    print("-" * len(hdr))
    for system, rows in per_system.items():
        s = summarize(rows)
        if not s["n"]:
            continue
        em = s["entity_keep_mean"]
        print(f"{system:20s} {s['n']:>4} {s['pass_all_rate']:>6} "
              f"{(em if em is not None else 0):>8} {s['copy_through']:>5} "
              f"{s['cyrillic']:>5} {s['english_leak']:>7} {s['len_outlier']:>6} "
              f"{s['degenerate']:>5}")


if __name__ == "__main__":
    main()
