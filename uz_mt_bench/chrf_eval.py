"""chrF++ / spBLEU on the human reference sets (reference-based cross-check).

chrF++ is the field standard for morphologically rich Turkic languages like
Uzbek; spBLEU (FLORES SentencePiece BLEU) is reported alongside for continuity
with FLORES/NLLB literature. Both need a human reference, so this runs ONLY on
the external refsets (NTREX / FLORES / turkic_xwmt), never on the in-domain
benchmark (which has no references — that's what XCOMET-QE + GEMBA-MQM are for).

Reads a refset (id -> reference uz_text) and each system's candidate file for
that refset, aligns by id, and reports corpus-level chrF++ and spBLEU per system.

Needs `pip install sacrebleu`.

Usage:
    python -m uz_mt_bench.chrf_eval --refset ntrex
    python -m ...chrf_eval --refset flores --candidates data/eval/candidates/flores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_refs(path: Path) -> dict[str, str]:
    refs = {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("uz_text"):
                refs[o["id"]] = o["uz_text"]
    return refs


def _load_hyps(path: Path) -> dict[str, str]:
    hyps = {}
    with path.open() as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("uz_text"):
                hyps[o["id"]] = o["uz_text"]
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refset", default="ntrex", help="refset name (ntrex/flores/xwmt)")
    ap.add_argument("--refset-file", default=None,
                    help="path to the refset jsonl (default: data/eval/refsets/<refset>.jsonl)")
    ap.add_argument("--candidates", default=None,
                    help="dir of candidate files (default: data/eval/candidates/<refset>)")
    ap.add_argument("--out-dir", default="data/eval/scores")
    args = ap.parse_args()

    import sacrebleu

    ref_path = Path(args.refset_file or f"data/eval/refsets/{args.refset}.jsonl")
    cand_dir = Path(args.candidates or f"data/eval/candidates/{args.refset}")
    refs = _load_refs(ref_path)
    files = sorted(cand_dir.glob("*.jsonl"))
    if not refs:
        print(f"no references in {ref_path}")
        return
    if not files:
        print(f"no candidate files in {cand_dir}")
        return

    out_path = Path(args.out_dir) / f"chrf_{args.refset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Restrict to segments every system translated: NLLB ran on the full refset
    # while the paid LLMs were capped, so per-system corpus scores over each
    # system's own segments are not comparable (same rule as aggregate.py uses
    # for reference-mode XCOMET).
    sys_hyps = {f.stem: _load_hyps(f) for f in files}
    sys_hyps = {s: h for s, h in sys_hyps.items() if h}
    shared = [i for i in refs if all(i in h for h in sys_hyps.values())]
    if not shared:
        print(f"no segment is translated by every system in {cand_dir}")
        return
    dropped = {s: len(h) - len(shared) for s, h in sys_hyps.items() if len(h) > len(shared)}
    if dropped:
        print("restricted to shared segments; excluded per system: "
              + ", ".join(f"{s} {n}" for s, n in sorted(dropped.items())))

    results = []
    for system, hyps in sys_hyps.items():
        ids = shared
        hyp_list = [hyps[i] for i in ids]
        ref_list = [refs[i] for i in ids]
        chrfpp = sacrebleu.corpus_chrf(hyp_list, [ref_list], word_order=2).score  # chrF++
        spbleu = sacrebleu.corpus_bleu(
            hyp_list, [ref_list], tokenize="flores200"
        ).score
        # Corpus-level chrF/BLEU aggregate n-gram counts over all segments, so a
        # single runaway output (e.g. a repetition loop hundreds of times the
        # reference length) can sink the whole corpus score. Count such outputs
        # so the table is never read without that context.
        degenerate = sum(1 for h, r in zip(hyp_list, ref_list)
                         if len(h) > 200 and len(h) > 3 * len(r))
        results.append({"system": system, "n": len(ids),
                        "chrf_pp": round(chrfpp, 2), "spbleu": round(spbleu, 2),
                        "degenerate": degenerate})

    results.sort(key=lambda r: -r["chrf_pp"])
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    print(f"refset={args.refset}  wrote {out_path}\n")
    print(f"{'system':20s} {'n':>5} {'chrF++':>7} {'spBLEU':>7}")
    print("-" * 42)
    for r in results:
        print(f"{r['system']:20s} {r['n']:>5} {r['chrf_pp']:>7} {r['spbleu']:>7}")


if __name__ == "__main__":
    main()
