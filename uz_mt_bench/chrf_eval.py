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

    results = []
    for f in files:
        system = f.stem
        hyps = _load_hyps(f)
        ids = [i for i in refs if i in hyps]        # aligned, order-stable
        if not ids:
            continue
        hyp_list = [hyps[i] for i in ids]
        ref_list = [refs[i] for i in ids]
        chrfpp = sacrebleu.corpus_chrf(hyp_list, [ref_list], word_order=2).score  # chrF++
        spbleu = sacrebleu.corpus_bleu(
            hyp_list, [ref_list], tokenize="flores200"
        ).score
        results.append({"system": system, "n": len(ids),
                        "chrf_pp": round(chrfpp, 2), "spbleu": round(spbleu, 2)})

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
