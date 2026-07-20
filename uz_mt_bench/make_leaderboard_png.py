"""Render LEADERBOARD.md's in-domain table to leaderboard.png.

The PNG shipped in the README was previously made by hand, so it silently went
stale whenever a system was added. This regenerates it from the markdown that
`aggregate.py` writes, so the image can never disagree with the numbers.

    python -m uz_mt_bench.make_leaderboard_png
    python -m uz_mt_bench.make_leaderboard_png --out /tmp/preview.png

Requires a Chrome/Chromium binary for the headless screenshot (no Python
plotting dependency — the layout is HTML/CSS, matching the original design).
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Pretty names for the leaderboard; anything unlisted falls back to its key.
DISPLAY = {
    "gemma4-26b": "Gemma 4 26B",
    "gemma4-12b": "Gemma 4 12B",
    "gemma4-31b-cloud": "Gemma 4 31B (cloud)",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "translategemma-27b": "TranslateGemma 27B",
    "translategemma-12b": "TranslateGemma 12B",
    "nllb-3.3b": "NLLB 3.3B",
    "nllb-1.3b": "NLLB 1.3B",
    "neuronai-uzbek": "NeuronAI Uzbek",
    "tilmoch": "Tilmoch",
}
BASELINE = "nllb-1.3b"
WIDTH, SCALE = 1200, 2          # 2400px wide at 2x DPR, as the original
ROW_H, CHROME_H = 46, 316       # per-row height + fixed header/footer/padding


def parse_table(md_path: Path) -> list[dict]:
    """Pull the in-domain leaderboard rows out of the generated markdown."""
    text = md_path.read_text()
    section = text.split("## In-domain leaderboard")[1].split("\n### ")[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| system") or "---" in line:
            continue
        c = [x.strip() for x in line.split("|")[1:-1]]
        if len(c) < 9 or c[0] not in DISPLAY:
            continue
        rows.append(
            {
                "key": c[0],
                "struct": float(c[3]),
                "entity": float(c[4]),
                "paired": float(c[5]),
                "p": c[8],
            }
        )
    if not rows:
        raise SystemExit(f"no leaderboard rows parsed from {md_path}")
    return rows


def _beats(row: dict) -> str:
    """The 'beats baseline' cell: significance verdict, not the raw p-value."""
    if row["key"] == BASELINE:
        return '<span class="dash">—</span>'
    if row["p"] == "—":
        return '<span class="dash">—</span>'
    if row["p"].endswith("*"):
        return '<span class="yes">p&lt;0.05 ✓</span>'
    return '<span class="no">no</span>'


def build_html(rows: list[dict], n_seg: int, n_paired: int) -> str:
    body = []
    for i, r in enumerate(rows, 1):
        badge = ""
        if i == 1:
            badge = '<span class="badge leader">LEADER</span>'
        elif r["key"] == BASELINE:
            badge = '<span class="badge base">BASELINE</span>'
        fill = "fill lead" if i == 1 else "fill"
        body.append(f"""
      <tr>
        <td class="rank">{i}</td>
        <td class="sys">{html.escape(DISPLAY[r['key']])}{badge}</td>
        <td class="barcell"><div class="track"><div class="{fill}"
             style="width:{r['paired'] * 100:.1f}%"></div></div></td>
        <td class="score">{r['paired']:.3f}</td>
        <td class="pct">{r['struct'] * 100:.1f}%</td>
        <td class="pct">{r['entity'] * 100:.1f}%</td>
        <td class="beats">{_beats(r)}</td>
      </tr>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ width: {WIDTH}px; background: #f6f6f4; padding: 34px;
         font-family: -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
         -webkit-font-smoothing: antialiased; }}
  .card {{ background: #fff; border: 1px solid #e6e6e2; border-radius: 18px;
           padding: 44px 44px 34px; }}
  h1 {{ font-size: 32px; font-weight: 800; letter-spacing: -.5px; color: #111; }}
  .sub {{ margin-top: 10px; font-size: 14.5px; color: #6b7280; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 26px; }}
  th {{ font-size: 10.5px; font-weight: 700; letter-spacing: .09em; color: #9aa0a6;
        text-transform: uppercase; text-align: left; padding: 0 0 12px; }}
  th.r, td.score, td.pct, td.beats {{ text-align: right; }}
  tr {{ border-top: 1px solid #eeeeec; }}
  td {{ padding: 15px 0; font-size: 15px; vertical-align: middle; }}
  td.rank {{ width: 34px; color: #9aa0a6; font-weight: 700; }}
  td.sys {{ font-weight: 700; color: #111; white-space: nowrap; padding-right: 18px; }}
  td.barcell {{ width: 42%; padding-right: 26px; }}
  .track {{ background: #ececea; border-radius: 6px; height: 13px; width: 100%; }}
  .fill {{ background: #2b7cd3; border-radius: 6px; height: 13px; }}
  .fill.lead {{ background: #12365f; }}
  td.score {{ font-weight: 700; color: #111; width: 88px; }}
  td.pct {{ color: #4b5563; width: 118px; }}
  td.beats {{ width: 140px; font-weight: 700; }}
  .yes {{ color: #111; }} .no {{ color: #9aa0a6; font-weight: 500; }}
  .dash {{ color: #c4c7cb; }}
  .badge {{ font-size: 10px; font-weight: 700; letter-spacing: .07em; padding: 3px 9px;
            border-radius: 999px; margin-left: 11px; vertical-align: 2px; }}
  .leader {{ color: #1a7f37; border: 1.5px solid #1a7f37; }}
  .base {{ color: #8b9096; border: 1.5px solid #d6d8da; }}
  .foot {{ margin-top: 22px; font-size: 12.5px; line-height: 1.65; color: #9aa0a6; }}
</style></head><body>
  <div class="card">
    <h1>🇺🇿 English → Uzbek Translation Benchmark</h1>
    <div class="sub">XCOMET-QE (paired) on {n_seg} in-domain call-agent segments —
      every system scored on the same {n_paired} shared segments. Baseline: NLLB-1.3B.</div>
    <table>
      <tr>
        <th>#</th><th>System</th><th>XCOMET-QE (paired)</th><th class="r">Score</th>
        <th class="r">Struct pass</th><th class="r">Entity keep</th>
        <th class="r">Beats baseline</th>
      </tr>{''.join(body)}
    </table>
    <div class="foot">Reference-free XCOMET-QE ranking, cross-checked against
      human-referenced FLORES &amp; NTREX sets with XCOMET-ref (Pearson r = 0.995) and
      chrF++ (r = 0.97). Struct pass = formatting preserved; entity keep =
      names/numbers preserved. Tilmoch was scored in-domain only, so it is absent
      from the reference-set cross-check.</div>
  </div>
</body></html>"""


def find_chrome() -> str:
    for b in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if path := shutil.which(b):
            return path
    raise SystemExit("no Chrome/Chromium binary found for headless rendering")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaderboard", default="LEADERBOARD.md")
    ap.add_argument("--out", default="leaderboard.png")
    args = ap.parse_args()

    md = Path(args.leaderboard)
    rows = parse_table(md)
    text = md.read_text()
    n_seg = int(re.search(r"(\d+) customer-support dialogue segments", text).group(1))
    n_paired = int(re.search(r"mean over the (\d+) segments", text).group(1))

    out = Path(args.out).resolve()
    height = ROW_H * len(rows) + CHROME_H          # fit the card, no dead space
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "board.html"
        page.write_text(build_html(rows, n_seg, n_paired))
        proc = subprocess.run(
            [
                find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                f"--screenshot={out}",
                f"--window-size={WIDTH},{height}",
                f"--force-device-scale-factor={SCALE}",
                "--hide-scrollbars",
                page.as_uri(),
            ],
            capture_output=True, text=True, timeout=120,
        )
    # Chrome exits 0 even when a bad flag aborts the capture, so trust the file,
    # not the return code — otherwise a stale PNG silently survives a "success".
    if not out.exists():
        raise SystemExit(
            f"chrome wrote no screenshot (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )
    print(f"wrote {out} ({len(rows)} systems, {WIDTH * SCALE}x{height * SCALE})")


if __name__ == "__main__":
    main()
