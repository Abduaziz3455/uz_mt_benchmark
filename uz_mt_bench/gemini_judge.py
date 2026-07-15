"""Phase 3 — GEMBA-MQM judge (Gemini, reference-free).

Implements the GEMBA-MQM method (Kocmi & Federmann, WMT23): prompt an LLM to mark
MQM error spans with a category and severity, then convert to a numeric score.
Per the v2 finding that a single judgment is noisy, each (segment, system) is
judged N times and averaged.

  score(segment) = max(MQM_FLOOR, sum of -weight per error)
      minor -1, major -5, critical -10; floor -25 (0 = flawless)

Reference-free: judge sees English source + Uzbek candidate (+ dialogue context),
no human reference. Also returns adequacy/fluency 1-5 as a secondary readout.

NOTE on pairwise: we do NOT make separate pairwise API calls. Because every
system is scored per-segment here, `aggregate.py` derives paired win-rates
(vs the NLLB-1.3B baseline and vs the leader) directly from these scores — the
same paired signal, at no extra cost.

Needs `pip install google-genai` and GEMINI_API_KEY. Resumable per (system, id).

Usage:
    python -m uz_mt_bench.gemini_judge \
        --bench data/eval/uz_mt_benchmark.jsonl \
        --candidates data/eval/candidates/uz_mt_benchmark
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .systems import GEMINI_MODEL

JUDGE_MODEL = GEMINI_MODEL
JUDGE_RUNS = 3                      # GEMBA-MQM v2: aggregate multiple judgments
JUDGE_THINKING_BUDGET = 0          # minimal thinking mode (per project decision)
# Multi-run averaging only reduces noise if the runs can differ. At temperature 0
# the N decodes are identical, `mqm_std` is always 0.0, and the self-consistency
# check in the report measures nothing. Sample instead when runs > 1.
JUDGE_TEMPERATURE = 0.3
WEIGHTS = {"critical": 10, "major": 5, "minor": 1}
MQM_FLOOR = -25.0
MAX_RETRIES = 4                     # transient API / malformed-response retries

# Constraining the decode is what actually fixes the JSONDecodeErrors: without a
# schema the model paraphrases the shape sketched in the prompt (emitting e.g.
# `"adequacy": int (1-5)`), which is not valid JSON. With response_schema set,
# Gemini is decode-constrained and cannot emit an unparseable object.
_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "errors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING"},
                    "severity": {"type": "STRING",
                                 "enum": ["critical", "major", "minor"]},
                    "span": {"type": "STRING"},
                },
                "required": ["category", "severity", "span"],
            },
        },
        "adequacy": {"type": "INTEGER"},
        "fluency": {"type": "INTEGER"},
    },
    "required": ["errors", "adequacy", "fluency"],
}

_SYSTEM_INSTRUCTION = (
    "You are an expert English→Uzbek (Latin script) translation evaluator using "
    "the MQM framework. Identify translation errors in the Uzbek candidate with "
    "respect to the English source. For each error give a category "
    "(mistranslation, omission, addition, grammar, spelling, terminology, "
    "wrong-script, entity-dropped, style, other) and a severity "
    "(critical, major, minor). Judge only the candidate turn; any provided "
    "context is background. Be strict and consistent."
)

_TASK = (
    "{ctx}English source:\n{src}\n\nUzbek candidate:\n{mt}\n\n"
    "List every translation error as an object with `category`, `severity` "
    "(critical, major or minor) and `span` (the exact substring of the Uzbek "
    "candidate that is wrong). Then rate `adequacy` and `fluency` as integers "
    "from 1 to 5. If the translation is flawless, return an empty errors list."
)


def _mqm_score(errors: list[dict]) -> float:
    penalty = sum(WEIGHTS.get((e.get("severity") or "").lower(), 1) for e in errors)
    return max(MQM_FLOOR, -float(penalty))


def _parse_json(text: str) -> dict:
    """Parse the judge's reply, tolerating code fences and trailing prose.

    The schema makes this near-unnecessary, but a decode that hits the output
    token cap can still truncate; failing loudly per-call (and retrying) beats
    silently recording an error row.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {t[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError(f"unbalanced JSON object in response: {t[:200]!r}")


def _load_bench(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            out[o["id"]] = {
                "en_text": o["en_text"],
                "context_prev": o.get("context_prev"),
                "category": o.get("category", o.get("domain", "all")),
            }
    return out


def _iter_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for o in _iter_jsonl(path):
        if not o.get("error"):
            keys.add((o["system"], o["id"]))
    return keys


class Judge:
    def __init__(self, temperature: float = JUDGE_TEMPERATURE):
        from google import genai

        import os
        self._genai = genai
        self._temperature = temperature
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _one(self, src: str, mt: str, ctx: str | None) -> dict:
        from google.genai import types

        ctx_block = (
            f"Dialogue context (previous turn, background only):\n{ctx}\n\n" if ctx else ""
        )
        prompt = _TASK.format(ctx=ctx_block, src=src, mt=mt)
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTION,
                        temperature=self._temperature,
                        response_mime_type="application/json",
                        response_schema=_RESPONSE_SCHEMA,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=JUDGE_THINKING_BUDGET
                        ),
                    ),
                )
                data = _parse_json(resp.text)
                errors = data.get("errors") or []
                return {
                    "mqm": _mqm_score(errors),
                    "errors": errors,
                    "adequacy": data.get("adequacy"),
                    "fluency": data.get("fluency"),
                }
            except Exception as exc:                      # transient API or decode
                last = exc
                if attempt < MAX_RETRIES - 1:
                    # jittered backoff; rate limits are the common case at concurrency
                    time.sleep(min(2 ** attempt + random.random(), 30.0))
        raise last                                        # type: ignore[misc]

    def judge(self, src: str, mt: str, ctx: str | None, runs: int) -> dict:
        """Average `runs` independent judgments (GEMBA-MQM v2 noise reduction).

        Also records the spread across runs (`mqm_std`) so aggregate.py can
        report judge self-consistency — the cheap reliability check for a
        low-resource LLM judge.
        """
        import statistics

        got = [self._one(src, mt, ctx) for _ in range(runs)]
        mqm_vals = [g["mqm"] for g in got]
        mqm = sum(mqm_vals) / len(mqm_vals)
        adq = [g["adequacy"] for g in got if g["adequacy"] is not None]
        flu = [g["fluency"] for g in got if g["fluency"] is not None]
        # keep the error list from the harshest run for the failure gallery
        worst = min(got, key=lambda g: g["mqm"])
        return {
            "mqm": round(mqm, 3),
            "mqm_std": round(statistics.pstdev(mqm_vals), 3) if len(mqm_vals) > 1 else 0.0,
            "mqm_runs": mqm_vals,
            "adequacy": round(sum(adq) / len(adq), 3) if adq else None,
            "fluency": round(sum(flu) / len(flu), 3) if flu else None,
            "errors": worst["errors"],
            "runs": runs,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--candidates", default="data/eval/candidates/uz_mt_benchmark")
    ap.add_argument("--out-dir", default="data/eval/scores")
    ap.add_argument("--runs", type=int, default=JUDGE_RUNS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--report-every", type=int, default=25)
    ap.add_argument("--systems", default=None,
                    help="comma-separated system keys to judge (default: all). "
                         "The judge is the tiebreaker for the top cluster — "
                         "restricting it to the contenders cuts the API bill.")
    ap.add_argument("--sample", type=int, default=0,
                    help="judge only N segments (the SAME N for every system, so "
                         "the paired comparison stays valid). 0 = all.")
    ap.add_argument("--seed", type=int, default=13, help="sampling seed")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=JUDGE_TEMPERATURE)
    args = ap.parse_args()

    bench_path = Path(args.bench)
    bench = _load_bench(bench_path)
    cand_dir = Path(args.candidates)
    files = sorted(cand_dir.glob("*.jsonl"))
    if args.systems:
        want = {s.strip() for s in args.systems.split(",") if s.strip()}
        missing = want - {f.stem for f in files}
        if missing:
            raise SystemExit(f"no candidate file for: {sorted(missing)}")
        files = [f for f in files if f.stem in want]
    if not files:
        print(f"no candidate files in {cand_dir}")
        return

    tag = args.tag or bench_path.stem
    out_path = Path(args.out_dir) / f"gemini_mqm_{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)

    # Sample a single shared id set across systems, otherwise the paired win-rates
    # in aggregate.py would be computed over different segments per system.
    keep_ids: set[str] | None = None
    if args.sample:
        per_system = [{c["id"] for c in _iter_jsonl(f)
                       if (c.get("uz_text") or "").strip()} for f in files]
        shared = sorted(set.intersection(*per_system) & set(bench))
        rng = random.Random(args.seed)
        keep_ids = set(rng.sample(shared, min(args.sample, len(shared))))
        print(f"sampling {len(keep_ids)} of {len(shared)} shared segments (seed={args.seed})")

    work = []
    for f in files:
        system = f.stem
        for cand in _iter_jsonl(f):
            uz = (cand.get("uz_text") or "").strip()
            b = bench.get(cand["id"])
            if b is None or not uz or (system, cand["id"]) in done:
                continue
            if keep_ids is not None and cand["id"] not in keep_ids:
                continue
            work.append((system, cand["id"], b, uz))
    print(f"judging {len(work)} (system,segment) pairs × {args.runs} runs "
          f"with {JUDGE_MODEL} (concurrency={args.concurrency}, "
          f"temperature={args.temperature}) -> {out_path}")
    if not work:
        print("nothing to do (all judged).")
        return

    judge = Judge(temperature=args.temperature)
    by_system: dict[str, list[float]] = defaultdict(list)
    ok = err = 0
    t0 = time.monotonic()
    write_lock = threading.Lock()

    def _run(item):
        system, sid, b, uz = item
        row = {"id": sid, "system": system}
        try:
            res = judge.judge(b["en_text"], uz, b.get("context_prev"), args.runs)
            row.update(res)
            row["error"] = None
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    with out_path.open("a") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(_run, it) for it in work]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row.get("error"):
                err += 1
            else:
                ok += 1
                by_system[row["system"]].append(row["mqm"])
            with write_lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if i % args.report_every == 0:
                rate = i / (time.monotonic() - t0)
                print(f"  {i}/{len(work)}  ok={ok} err={err}  {rate:.1f}/s")

    print(f"\nfinished: ok={ok} err={err}")
    if err:
        print(f"  ⚠ {err} pairs failed after {MAX_RETRIES} retries — rerun to "
              "resume (successful rows are skipped).")
    print(f"\n{'system':20s} {'n':>4} {'mean_MQM':>9}   (0 = flawless, more negative = worse)")
    print("-" * 40)
    for system in sorted(by_system, key=lambda s: -sum(by_system[s]) / len(by_system[s])):
        vals = by_system[system]
        print(f"{system:20s} {len(vals):>4} {sum(vals)/len(vals):>9.3f}")


if __name__ == "__main__":
    main()
