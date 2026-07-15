# EN→UZ Translation-Quality Benchmark

**Which machine-translation system should you use to turn English voice-call-agent
data into Uzbek (Latin)?**

Uzbek is low-resource. The usual answer — "run BLEU against FLORES" — is a poor
guide here: FLORES is formal wiki prose, it is NLLB's own home benchmark, and it
has been public long enough that every large LLM has probably trained on it. None
of that tells you how a system handles a *caller interrupting an agent mid-sentence*,
or whether it preserves the flight number `TR2704` and the time `9:35` verbatim.

So this benchmark asks the question directly. It scores **9 systems** on **600
novel, in-domain call-agent segments** with a reference-free neural metric, hard
structural gates, and an out-of-domain human-reference cross-check.

📊 **[LEADERBOARD.md](LEADERBOARD.md)** — the results
📐 **[METHODOLOGY.md](METHODOLOGY.md)** — why these metrics, and what they can't tell you
⚖️ **[ATTRIBUTION.md](ATTRIBUTION.md)** — data sources, licenses, and what you may do with the outputs

---

## What it measures

Two questions, deliberately kept apart.

**1. In-domain quality (the decision).** 600 English turns sampled from real
task-oriented-dialogue corpora, stratified across 12 source corpora, 91 domains,
and 5 turn categories (`caller_turn`, `agent_reply`, `kb_passage`, `tool_spoken`,
`edge`). There are no human Uzbek references for these — authoring them with an
LLM would bias the evaluation toward LLM systems — so scoring is **reference-free**:

| signal | what it catches |
|---|---|
| **XCOMET-QE** (`Unbabel/XCOMET-XL`, 3.5B, QE mode) | adequacy / fluency, no reference needed |
| **structural gates** | wrong script, English pass-through, dropped entities, degeneration, empty output |
| **GEMBA-MQM** (Gemini judge, MQM error spans) | qualitative failure categories |

A system that scores well but silently drops `TR2704` is disqualified for a
call agent regardless of how fluent it sounds — which is why the gates are
reported next to the score, not folded into it.

**2. Out-of-domain sanity check (metric validation).** The same 9 systems on
**NTREX-128** and **FLORES-200**, scored by XCOMET-XL in *reference* mode against
professional human translations. This exists to answer "is the reference-free
metric measuring anything real?" — and it does: reference-free and reference-based
scoring agree on the system ranking at Pearson **0.995**.

## Systems under test

| system | family | serving |
|---|---|---|
| `nllb-1.3b` | NLLB-200 (enc-dec MT) | local — **baseline** |
| `nllb-3.3b` | NLLB-200 (enc-dec MT) | local |
| `gemma4-12b` / `gemma4-26b` | Gemma 4 (general LLM) | local Ollama |
| `gemma4-31b-cloud` | Gemma 4 (general LLM) | Ollama Cloud |
| `translategemma-12b` / `translategemma-27b` | TranslateGemma (MT-tuned) | local Ollama |
| `gemini-3.5-flash` | Gemini (LLM) | Google API |
| `neuronai-uzbek` | Uzbek-specialised Qwen3-4B FT | local HF |

Every LLM system gets the **same fixed translation prompt** at temperature 0;
NLLB uses beam 4. See [`uz_mt_bench/systems.py`](uz_mt_bench/systems.py).

---

## ⚠ Status of the current numbers

The leaderboard in this repo was produced by a scoring run that ran **without
long-segment chunking**. XCOMET's encoder truncates at 512 subword tokens, so the
56 segments longer than ~80 words were cut off mid-input, which floors their score
for *every* system and compresses the gaps between systems. The `kb_passage`
column is the visible casualty.

Chunking is now the default in `comet_qe.py`. **Re-run the QE scoring step and
regenerate the leaderboard before citing these numbers**:

```bash
python -m uz_mt_bench.comet_qe      # ~20 min on an A100; chunking is on by default
python -m uz_mt_bench.aggregate     # rewrites LEADERBOARD.md
```

The ⚠ banner in `LEADERBOARD.md` disappears once the scores carry `n_chunks`.
`aggregate.py` derives that banner from the score file itself, so it cannot go
stale.

Two further gaps, both stated in the leaderboard's **Not run in this release**
section: the **GEMBA-MQM judge** never completed (its output was a handful of
malformed-JSON rows and is not shipped), and **chrF++** and the **turkic_xwmt**
refset were never run. The harness for all three ships here; the results do not.

---

## Reproducing

Everything below runs from this directory. The 600-segment benchmark, all 9
systems' translations, and the score files are **included**, so you can re-derive
the leaderboard without a GPU or an API key:

```bash
python -m uz_mt_bench.aggregate && cat LEADERBOARD.md
```

Re-running the structural gates is also free (CPU, no model):

```bash
python -m uz_mt_bench.structural_checks
```

### Full pipeline, from scratch

**1. Install.** Needs a CUDA `torch` already present (install it separately for
your CUDA version). Do *not* install `torchvision` — this harness is text-only and
a mismatched `torchvision` breaks the `transformers` import chain.

```bash
pip install -r requirements.txt
huggingface-cli login          # XCOMET-XL and FLORES-200 are gated repos
```

**2. Keys and models.**

```bash
export GEMINI_API_KEY=...       # gemini-3.5-flash (system + judge)
export OLLAMA_API_KEY=...       # gemma4:31b-cloud only

ollama serve &
ollama pull gemma4:12b && ollama pull gemma4:26b
ollama pull translategemma:12b && ollama pull translategemma:27b
# gemma4:31b-cloud is served by Ollama Cloud; neuronai-uzbek auto-downloads from HF

python -m uz_mt_bench.preflight   # verifies every system answers before you spend
```

**3. Build the data.** (Both are already in `data/eval/` — rebuild only if you
want to change the sampling. `build_benchmark` needs the upstream SFT corpus,
which is *not* shipped; see [ATTRIBUTION.md](ATTRIBUTION.md).)

```bash
python -m uz_mt_bench.build_benchmark    # 600 in-domain + 60 smoke
python -m uz_mt_bench.fetch_refsets      # NTREX + FLORES + turkic_xwmt
```

**4. Smoke-test the wiring** on 60 segments and 2 systems before spending money:

```bash
python -m uz_mt_bench.translate_all \
    --input data/eval/uz_mt_benchmark.smoke.jsonl --systems nllb-1.3b,gemini-3.5-flash
python -m uz_mt_bench.structural_checks \
    --candidates data/eval/candidates/uz_mt_benchmark.smoke
```

**5. Translate** all 9 systems (resumable — re-run after an OOM or rate-limit and
it skips finished rows):

```bash
python -m uz_mt_bench.translate_all --systems all
```

**6. Score in-domain**, then **cross-check on the human refsets**:

```bash
python -m uz_mt_bench.structural_checks     # hard gates      (CPU)
python -m uz_mt_bench.comet_qe              # XCOMET-QE       (GPU)
python -m uz_mt_bench.gemini_judge          # GEMBA-MQM ×3    (API)

for RS in ntrex flores xwmt; do
  python -m uz_mt_bench.translate_all --input data/eval/refsets/$RS.jsonl
  python -m uz_mt_bench.comet_qe \
      --bench data/eval/refsets/$RS.jsonl --candidates data/eval/candidates/$RS --reference
  python -m uz_mt_bench.chrf_eval --refset $RS
done
```

**7. Build the leaderboard.**

```bash
python -m uz_mt_bench.aggregate
```

`aggregate.py` reports only what it finds. A metric whose score file is absent is
named under *Not run in this release* rather than rendered as a column of
em-dashes — a blank cell reads as "the metric found nothing", and that is a
different claim.

### Hardware

XCOMET-XL is 3.5B params and wants **~16-20 GB of VRAM** in fp32; the reference
runs used a single A100 40GB, one phase at a time. NLLB-3.3B ≈ 7 GB,
`neuronai-uzbek` ≈ 9 GB, and the local Ollama models 16-18 GB (loaded one at a
time). A 12 GB consumer card is not enough for the QE step.

Cost drivers are the Gemini calls (1 translation + 3 judge passes × 600 segments)
and the Ollama-Cloud 31B model.

---

## Layout

```
uz_mt_bench/
  build_benchmark.py     # sample + stratify the 600 in-domain segments
  fetch_refsets.py       # download NTREX / FLORES / turkic_xwmt
  systems.py             # the 9 systems behind one translate() interface
  preflight.py           # check every system responds before a paid run
  translate_all.py       # generate candidates          -> data/eval/candidates/
  structural_checks.py   # hard gates                   -> scores/structural.jsonl
  comet_qe.py            # XCOMET-XL, QE + reference    -> scores/xcomet_{qe,ref}_*.jsonl
  gemini_judge.py        # GEMBA-MQM judge              -> scores/gemini_mqm_*.jsonl
  chrf_eval.py           # chrF++ / spBLEU on refsets   -> scores/chrf_*.jsonl
  dropped_entities.py    # diagnostic: which entities each system loses
  aggregate.py           # everything -> LEADERBOARD.md
  entity_mask.py         # vendored: entity masking for NLLB
  nllb_translator.py     # vendored: production NLLB wrapper

data/eval/
  uz_mt_benchmark.jsonl        # the 600 segments (+ .smoke.jsonl, 60)
  candidates/<set>/<system>.jsonl
  scores/*.jsonl
  refsets/                     # not shipped — fetch_refsets.py downloads them
```

Benchmark record:

```json
{
  "id": "sgd-000123-t4",
  "section": "sgd", "domain": "Services_3", "category": "tool_spoken",
  "en_text": "I can reserve a seat on TR2704 departing at 9:35.",
  "context_prev": "Where are you departing from?",
  "preserve": ["TR2704", "9:35"],
  "char_len": 49, "word_len": 9
}
```

`preserve[]` drives the entity-preservation gate. `context_prev` is given to LLM
translators as prior dialogue context but is never itself translated or scored.

## Licensing

Code is **Apache-2.0**. The benchmark segments are derived from corpora under
CC-BY-SA-4.0, CC-BY-4.0, CDLA-Sharing-1.0 and MIT, so the collection is released
under **CC-BY-SA-4.0**, the most restrictive of its inputs. The `section` field on
every record identifies its upstream source.

**The shipped NLLB translations are CC-BY-NC-4.0** (non-commercial), inherited
from the NLLB-200 model weights. See [ATTRIBUTION.md](ATTRIBUTION.md) before
reusing any of the candidate outputs.

## Citing

If this benchmark or its methodology is useful to you, please cite it and the
metrics it stands on — XCOMET ([Guerreiro et al., TACL 2024](https://doi.org/10.1162/tacl_a_00683))
and GEMBA-MQM ([Kocmi & Federmann, WMT 2023](https://aclanthology.org/2023.wmt-1.64/)).
