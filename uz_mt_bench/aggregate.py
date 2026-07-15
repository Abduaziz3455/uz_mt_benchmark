"""Phase 5 — aggregate every score file into LEADERBOARD.md.

Pulls together what the earlier phases wrote:

  Leaderboard (translation quality, all 9 systems)
    - structural pass-rate + entity-preservation  (data/eval/scores/structural.jsonl)
    - XCOMET-QE mean, with per-category breakdown  (xcomet_qe_<bench>.jsonl)
    - GEMBA-MQM mean, if the judge was run         (gemini_mqm_<bench>.jsonl)
    - paired win-rate vs the NLLB-1.3B baseline and vs the leader
    - reference-based XCOMET / chrF++ on the human refsets

Bootstrap 95% CIs; paired comparison on shared segments. Sections whose inputs
are absent are reported as "not run" rather than silently rendered as blanks —
a leaderboard column full of em-dashes reads as data, and it isn't.

Usage:
    python -m uz_mt_bench.aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .systems import QUALITY_BOARD

BASELINE = "nllb-1.3b"


# ── io helpers ───────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _bootstrap_ci(xs: list[float], iters: int, rng: random.Random) -> tuple[float, float] | None:
    """95% percentile bootstrap CI of the mean (deterministic given seed)."""
    if len(xs) < 2:
        return None
    n = len(xs)
    means = []
    for _ in range(iters):
        s = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return round(lo, 4), round(hi, 4)


def _paired_bootstrap_p(a: dict[str, float], b: dict[str, float],
                        iters: int, rng: random.Random) -> float | None:
    """One-sided paired-bootstrap p-value for H1: mean(A) > mean(B) on shared
    segments. p = fraction of resamples where the mean paired diff <= 0."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    diffs = [a[i] - b[i] for i in shared]
    if sum(diffs) <= 0:            # A isn't ahead on the point estimate
        return 1.0
    n = len(diffs)
    le0 = 0
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        if s <= 0:
            le0 += 1
    return round(le0 / iters, 4)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else None


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _judge_consistency(rows: list[dict]) -> dict[str, float]:
    """Mean per-segment GEMBA-MQM std across the N judge runs, per system."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("mqm_std") is not None and not r.get("error"):
            per[r["system"]].append(r["mqm_std"])
    return {s: round(sum(v) / len(v), 3) for s, v in per.items() if v}


# ── loaders keyed by system → {id: score} ───────────────────────────────────
def _by_system_scores(rows: list[dict], value: str = "score") -> dict[str, dict[str, float]]:
    d: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.get(value) is not None and not r.get("error"):
            d[r["system"]][r["id"]] = r[value]
    return d


def _structural_summary(rows: list[dict]) -> dict[str, dict]:
    per: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per[r["system"]].append(r)
    out = {}
    for sysname, rs in per.items():
        n = len(rs)
        ent = [r["entity_keep"] for r in rs if r.get("entity_keep") is not None]
        passed = sum(
            1 for r in rs
            if not (r["empty"] or r["copy_through"] or r["cyrillic"]
                    or r["english_leak"] or r["degenerate"]
                    or (r["entity_keep"] is not None and r["entity_keep"] < 1.0))
        )
        out[sysname] = {
            "pass_all": round(passed / n, 4) if n else None,
            "entity_keep": round(sum(ent) / len(ent), 4) if ent else None,
            "empty": sum(1 for r in rs if r.get("empty")),
        }
    return out


def _paired_winrate(scores: dict[str, dict[str, float]], a: str, b: str) -> float | None:
    """Fraction of shared segments where a's score > b's (ties = 0.5)."""
    if a not in scores or b not in scores:
        return None
    shared = set(scores[a]) & set(scores[b])
    if not shared:
        return None
    wins = sum(
        1.0 if scores[a][i] > scores[b][i] else 0.5 if scores[a][i] == scores[b][i] else 0.0
        for i in shared
    )
    return round(wins / len(shared), 4)


def _paired_wld(a: dict[str, float], b: dict[str, float],
                eps: float = 0.01) -> tuple[int, int, int] | None:
    """Win / loss / tie counts on shared segments (|diff| <= eps counts as a tie).

    The headline means hide this: two systems can sit 0.004 apart on the mean and
    still disagree on hundreds of segments, or sit far apart and agree everywhere.
    """
    shared = set(a) & set(b)
    if not shared:
        return None
    w = sum(1 for i in shared if a[i] - b[i] > eps)
    l = sum(1 for i in shared if b[i] - a[i] > eps)
    return w, l, len(shared) - w - l


def _shared_ids(per_system: dict[str, dict[str, float]]) -> set[str]:
    """Segment ids every system has a score for.

    Essential for the refsets: NLLB was scored on all 1997 NTREX segments while
    the paid LLMs were capped at 500, so raw per-system means are not comparable.
    """
    if not per_system:
        return set()
    return set.intersection(*(set(v) for v in per_system.values()))


# ── report ───────────────────────────────────────────────────────────────────
def build_report(scores_dir: Path, bench_path: Path, bench_tag: str,
                 refsets: list[str], boot_iters: int, seed: int) -> str:
    rng = random.Random(seed)
    L = []
    A = lambda s="": L.append(s)

    structural = _structural_summary(_read_jsonl(scores_dir / "structural.jsonl"))
    qe = _by_system_scores(_read_jsonl(scores_dir / f"xcomet_qe_{bench_tag}.jsonl"))
    mqm = _by_system_scores(_read_jsonl(scores_dir / f"gemini_mqm_{bench_tag}.jsonl"), "mqm")
    n_bench = sum(1 for _ in bench_path.open()) if bench_path.exists() else 0

    # Segments every system has a score for. Empty translations are unscorable, so
    # a system that failed to emit output would otherwise be averaged over fewer
    # (and easier) segments than its rivals. The paired mean fixes the comparison.
    shared = _shared_ids(qe)

    qe_mean = {s: _mean(list(v.values())) for s, v in qe.items()}
    paired_mean = {s: _mean([qe[s][i] for i in shared]) for s in qe} if shared else {}

    # rank by the paired mean when it is available — it is the apples-to-apples number
    key_mean = paired_mean or qe_mean
    ranked = sorted(
        [s for s in QUALITY_BOARD if key_mean.get(s) is not None],
        key=lambda s: -key_mean[s],
    )
    leader = ranked[0] if ranked else None
    has_mqm = bool(mqm)

    A("# EN→UZ Translation-Quality Benchmark — Leaderboard\n")
    A("Generated by `aggregate.py`; do not edit by hand. "
      "Methodology: [METHODOLOGY.md](METHODOLOGY.md).\n")

    A("## In-domain leaderboard (reference-free, 600 call-agent segments)\n")
    A(f"Ranked by **XCOMET-QE (paired)** — the mean over the {len(shared)} segments "
      f"every system scored. Baseline `{BASELINE}`; leader `{leader or '—'}`.\n")
    A("`p(base)` is the one-sided paired-bootstrap p-value that the system beats the "
      "baseline on XCOMET-QE (`*` = p<0.05). `W/L/T vs base` counts segments where the "
      "system's XCOMET-QE beats / loses to / ties (within ±0.01) the baseline's. Read "
      "W/L/T before the means: overlapping CIs with a significant `p(base)` is the "
      "normal signature of a real but small per-segment effect, not a tie.\n")
    A("`empty` counts segments where the system returned nothing. Empty outputs cannot "
      "be scored by XCOMET, so they are excluded from `XCOMET-QE (all)` — see the "
      "robustness note below the table.\n")

    head = ("| system | n | empty | struct pass | entity keep | XCOMET-QE (paired) | "
            "XCOMET-QE (all) | QE 95% CI | p(base) | W/L/T vs base |")
    rule = "|---|---|---|---|---|---|---|---|---|---|"
    if has_mqm:
        head += " GEMBA-MQM | win vs baseline | win vs leader |"
        rule += "---|---|---|"
    A(head)
    A(rule)
    for s in ranked or QUALITY_BOARD:
        st = structural.get(s, {})
        ci = _bootstrap_ci(list(qe[s].values()), boot_iters, rng) if s in qe else None
        if s == BASELINE:
            pcell, wldcell = "baseline", "baseline"
        else:
            p = _paired_bootstrap_p(qe.get(s, {}), qe.get(BASELINE, {}), boot_iters, rng)
            pcell = "—" if p is None else (f"{p}*" if p < 0.05 else f"{p}")
            wld = _paired_wld(qe.get(s, {}), qe.get(BASELINE, {}))
            wldcell = f"{wld[0]}/{wld[1]}/{wld[2]}" if wld else "—"
        row = "| {sys} | {n} | {em} | {pa} | {ek} | {qp} | {qe} | {ci} | {p} | {wld} |".format(
            sys=s,
            n=len(qe.get(s, {})) or "—",
            em=st.get("empty", "—"),
            pa=_fmt(st.get("pass_all")),
            ek=_fmt(st.get("entity_keep")),
            qp=_fmt(paired_mean.get(s)),
            qe=_fmt(qe_mean.get(s)),
            ci=f"[{ci[0]}, {ci[1]}]" if ci else "—",
            p=pcell,
            wld=wldcell,
        )
        if has_mqm:
            mm = _mean(list(mqm[s].values())) if s in mqm else None
            wb = _paired_winrate(mqm, s, BASELINE)
            wl = _paired_winrate(mqm, s, leader) if leader else None
            row += " {mm} | {wb} | {wl} |".format(
                mm=_fmt(mm),
                wb="baseline" if s == BASELINE else _fmt(wb),
                wl="leader" if s == leader else _fmt(wl),
            )
        A(row)
    A()

    _empty_robustness(A, qe, qe_mean, paired_mean, structural, ranked, n_bench)

    # Long inputs beyond XCOMET's 512-subword window are invisible to the metric,
    # which floors long segments for every system and compresses the leaderboard.
    qe_rows = _read_jsonl(scores_dir / f"xcomet_qe_{bench_tag}.jsonl")
    chunked = sum(1 for r in qe_rows if (r.get("n_chunks") or 1) > 1)
    if chunked:
        A(f"_Long-segment chunking active: {chunked} of {len(qe_rows)} scored "
          "(system, segment) pairs were sentence-aligned into multiple chunks so "
          "XCOMET's 512-token window never truncates them._\n")
    elif qe_rows:
        A("_⚠ **Scored without long-segment chunking** — sources beyond XCOMET's "
          "512-subword window were truncated, which floors long segments for every "
          "system and compresses the leaderboard. These scores are not comparable to "
          "a chunked run. Rerun `comet_qe` (chunking is the default)._\n")

    # ---- per-category QE ----
    A("### XCOMET-QE by category\n")
    _per_category(A, scores_dir, bench_path, bench_tag)

    # ---- reference-based cross-check ----
    A("## Reference-based cross-check (external human refsets, formal domain)\n")
    A("> Bias note: FLORES is NLLB's home benchmark (mildly favors NLLB); use "
      "NTREX + turkic_xwmt too. These are formal-domain — the in-domain decision "
      "stays on the leaderboard above.\n")
    A("Scored with XCOMET-XL in **reference mode** against the human reference. "
      "Restricted to segments every system translated, so the means are "
      "comparable (systems were run on different slice sizes).\n")

    ref_means: dict[str, list[float]] = defaultdict(list)
    for rs in refsets:
        rows = _by_system_scores(_read_jsonl(scores_dir / f"xcomet_ref_{rs}.jsonl"))
        if not rows:
            continue
        shared = _shared_ids(rows)
        if not shared:
            A(f"_({rs}: no segment is scored for every system; skipping)_\n")
            continue
        means = {s: _mean([rows[s][i] for i in shared]) for s in rows}
        A(f"**{rs}** — XCOMET-XL (reference mode), n={len(shared)} shared segments\n")
        A("| system | XCOMET-ref |")
        A("|---|---|")
        for s in sorted(means, key=lambda k: -means[k]):
            A(f"| {s} | {_fmt(round(means[s], 4))} |")
            ref_means[s].append(means[s])
        A()

        dropped = {s: len(rows[s]) - len(shared) for s in rows}
        if any(dropped.values()):
            A("_Segments scored but excluded as not-shared: "
              + ", ".join(f"`{s}` {n}" for s, n in sorted(dropped.items()) if n)
              + "._\n")

    # legacy chrF++ tables, if that scorer was ever run
    for rs in refsets:
        rows = _read_jsonl(scores_dir / f"chrf_{rs}.jsonl")
        if not rows:
            continue
        A(f"**{rs}** (chrF++ / spBLEU)\n")
        A("| system | n | chrF++ | spBLEU |")
        A("|---|---|---|---|")
        for r in sorted(rows, key=lambda x: -x["chrf_pp"]):
            A(f"| {r['system']} | {r['n']} | {r['chrf_pp']} | {r['spbleu']} |")
        A()

    # ---- metric agreement (meta-evaluation) ----
    A("## Metric agreement\n")
    A("Do the independent metrics agree? A weakly-calibrated metric on a "
      "low-resource language must be cross-checked, not trusted on its own.\n")

    ref_avg = {s: sum(v) / len(v) for s, v in ref_means.items()}
    common_r = [s for s in qe_mean if s in ref_avg and qe_mean[s] is not None]
    if len(common_r) >= 2:
        A(f"- **System-level QE-mean ↔ XCOMET-ref (refsets)** (n={len(common_r)} systems): "
          f"Pearson {_pearson([qe_mean[s] for s in common_r], [ref_avg[s] for s in common_r])}. "
          "Reference-free and reference-based scoring agreeing on the system ranking is "
          "the strongest evidence the QE backbone is measuring translation quality.")

    MIN_CORR_N = 30
    if has_mqm:
        xs, ys = [], []
        for s in qe:
            if s in mqm:
                for i in set(qe[s]) & set(mqm[s]):
                    xs.append(qe[s][i])
                    ys.append(mqm[s][i])
        if len(xs) >= MIN_CORR_N:
            A(f"- **Segment-level XCOMET-QE ↔ GEMBA-MQM** (n={len(xs)}): "
              f"Pearson {_pearson(xs, ys)}, Spearman {_spearman(xs, ys)}. "
              "Higher = the two independent metrics agree on which segments are good.")
        elif xs:
            A(f"- **Segment-level XCOMET-QE ↔ GEMBA-MQM**: only n={len(xs)} segments have "
              f"both scores — too few to correlate (need ≥{MIN_CORR_N}).")

        common = [s for s in qe_mean if s in mqm and qe_mean[s] is not None]
        if len(common) >= 2:
            qe_v = [qe_mean[s] for s in common]
            mqm_v = [_mean(list(mqm[s].values())) for s in common]
            A(f"- **System-level QE-mean ↔ MQM-mean** (n={len(common)} systems): "
              f"Pearson {_pearson(qe_v, mqm_v)}.")

        jc = _judge_consistency(_read_jsonl(scores_dir / f"gemini_mqm_{bench_tag}.jsonl"))
        if jc:
            A("- **GEMBA-MQM judge self-consistency** — mean per-segment std across the "
              "judge runs (lower = more stable; MQM scale, 0 = flawless):")
            for s in sorted(jc, key=lambda k: jc[k]):
                A(f"    - `{s}`: ±{jc[s]}")
            if all(v == 0.0 for v in jc.values()):
                A("    - _⚠ Every std is exactly 0.0 — the judge ran at temperature 0, so "
                  "the N runs are identical decodes and this check measures nothing._")
    A()

    _not_run(A, scores_dir, bench_tag, refsets, has_mqm)

    # ---- caveats ----
    A("## Caveats\n")
    if has_mqm:
        A("- **Gemini self-judge bias** — Gemini judges its own translations; the "
          "ranking is anchored on XCOMET-QE + structural gates (Gemini-independent). "
          "Treat Gemini's own GEMBA-MQM row with caution.")
    A("- **XCOMET-QE is a relative signal** for Uzbek — calibration is weaker than for "
      "high-resource languages. Read the ranking as an ordering, not as absolute "
      "adequacy, and read it alongside the structural gates.")
    A("- **Reference sets are formal-domain** (wiki/news/talks), not call-agent "
      "dialogue. They validate the metric and give a general EN→UZ head-to-head; they "
      "do not decide the in-domain question.")
    A("- **Test-set contamination**: FLORES/NTREX are old and public, so LLMs may have "
      "seen them in pretraining (which inflates LLM scores on the refsets relative to "
      "NLLB). The in-domain benchmark is novel and contamination-free — another reason "
      "it is primary.")
    A("- **No human evaluation**: the ranking is automatic-metrics-only. A ~50-segment "
      "native-speaker MQM spot-check of the top systems would confirm it — the one "
      "check no code substitutes for, on a low-resource language.")
    return "\n".join(L)


def _empty_robustness(A, qe, qe_mean, paired_mean, structural, ranked, n_bench) -> None:
    """State, with numbers, how empty outputs move the ranking.

    An empty translation cannot be scored, so it silently leaves its system's mean —
    which flatters exactly the systems that failed. Rather than warn and move on, show
    the ranking under the opposite assumption (empty = 0.0, a total adequacy failure)
    and say whether the order survives it.
    """
    empties = {s: structural.get(s, {}).get("empty", 0) for s in ranked}

    # scored + empty must exhaust the benchmark. If it doesn't, the score file and the
    # structural file were built from different candidate files (e.g. one predates a
    # translate_all retry) and the columns below describe different runs.
    if n_bench:
        skew = {s: len(qe[s]) + empties.get(s, 0) - n_bench for s in ranked
                if s in qe and len(qe[s]) + empties.get(s, 0) != n_bench}
        if skew:
            A("_⚠ **Inconsistent inputs.** For "
              + ", ".join(f"`{s}` (scored {len(qe[s])} + empty {empties[s]} ≠ {n_bench})"
                          for s in skew)
              + ", the XCOMET score file and `structural.jsonl` disagree about how many "
                "segments exist. They were generated from different candidate files. "
                "Re-run `structural_checks` and `comet_qe` against the same "
                "`data/eval/candidates/` before publishing._\n")

    if not any(empties.values()) or not n_bench:
        return
    offenders = ", ".join(f"`{s}` {n}" for s, n in sorted(empties.items(),
                                                          key=lambda kv: -kv[1]) if n)
    zero_mean = {s: sum(qe[s].values()) / n_bench for s in ranked if s in qe}
    zero_order = sorted(zero_mean, key=lambda s: -zero_mean[s])
    A(f"_**Empty-output robustness.** {offenders} returned no translation for at least "
      f"one segment. XCOMET cannot score an empty string, so those segments drop out of "
      f"`XCOMET-QE (all)`. The `XCOMET-QE (paired)` column removes the bias by scoring "
      f"every system on the same {len(_shared_ids(qe))} segments._\n")
    if zero_order != ranked:
        moved = [f"`{s}` {ranked.index(s) + 1}→{zero_order.index(s) + 1}"
                 for s in ranked if ranked.index(s) != zero_order.index(s)]
        A(f"_Under the stricter assumption that an empty output scores 0.0 (a total "
          f"adequacy failure rather than a missing datum), the order changes: "
          f"{', '.join(moved)}. The paired ranking above is therefore not robust to how "
          f"empties are treated — prefer the system with both a high score and zero "
          f"empties._\n")
    else:
        A("_Scoring empties as 0.0 (a total adequacy failure rather than a missing "
          "datum) leaves the ranking unchanged, so the order is robust to how empties "
          "are treated._\n")


def _not_run(A, scores_dir: Path, bench_tag: str, refsets: list[str], has_mqm: bool) -> None:
    """Name every planned component whose artifacts are absent.

    A leaderboard that renders a never-run metric as a column of em-dashes reads as
    'the metric found nothing'. Say plainly that it did not run.
    """
    missing = []
    if not has_mqm:
        missing.append(
            "**GEMBA-MQM (Gemini judge)** — no `gemini_mqm_{tag}.jsonl`. The judge, the "
            "MQM win-rates, and the judge self-consistency check are absent from this "
            "release. Reproduce with `gemini_judge`.".format(tag=bench_tag))
    if not any(_read_jsonl(scores_dir / f"chrf_{rs}.jsonl") for rs in refsets):
        missing.append("**chrF++ / spBLEU** — no `chrf_<refset>.jsonl`. The string-metric "
                       "cross-check did not run. Reproduce with `chrf_eval`.")
    for rs in refsets:
        if not _read_jsonl(scores_dir / f"xcomet_ref_{rs}.jsonl"):
            missing.append(f"**{rs} refset** — no `xcomet_ref_{rs}.jsonl`; that reference "
                           f"set was not scored.")
    if not missing:
        return
    A("## Not run in this release\n")
    A("Planned in the methodology, absent from these numbers:\n")
    for m in missing:
        A(f"- {m}")
    A()


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else ("—" if x is None else str(x))


def _per_category(A, scores_dir: Path, bench_path: Path, bench_tag: str):
    # requires the benchmark file for category labels
    if not bench_path.exists():
        A("_(benchmark file missing; skipping per-category)_\n")
        return
    cat_of = {json.loads(l)["id"]: json.loads(l).get("category", "all")
              for l in bench_path.open()}
    rows = _read_jsonl(scores_dir / f"xcomet_qe_{bench_tag}.jsonl")
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["system"], cat_of.get(r["id"], "all"))].append(r["score"])
    cats = sorted({c for _, c in agg})
    systems = sorted({s for s, _ in agg})
    A("| system | " + " | ".join(cats) + " |")
    A("|---|" + "|".join(["---"] * len(cats)) + "|")
    for s in systems:
        cells = []
        for c in cats:
            vals = agg.get((s, c))
            cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
        A(f"| {s} | " + " | ".join(cells) + " |")
    A()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-dir", default="data/eval/scores")
    ap.add_argument("--bench", default="data/eval/uz_mt_benchmark.jsonl")
    ap.add_argument("--bench-tag", default="uz_mt_benchmark")
    ap.add_argument("--refsets", default="ntrex,flores,xwmt")
    ap.add_argument("--out", default="LEADERBOARD.md")
    ap.add_argument("--boot-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    report = build_report(
        Path(args.scores_dir), Path(args.bench), args.bench_tag,
        [r.strip() for r in args.refsets.split(",") if r.strip()],
        args.boot_iters, args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out} ({len(report)} chars)")


if __name__ == "__main__":
    main()
