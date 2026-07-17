# EN→UZ Translation-Quality Benchmark — Methodology

**Goal:** measure how well each candidate system translates English voice-call-agent
data into **Uzbek (Latin)**, in order to pick a translator for a production dataset
and quantify the gap against the incumbent (NLLB-1.3B) — using an evaluation design
grounded in current low-resource MT-evaluation research.

Results: [LEADERBOARD.md](LEADERBOARD.md). Running it: [README.md](README.md).

---

## 1. Research foundation (why these methods)

- **No single metric — ensemble a neural metric, an LLM judge, and a string metric.**
  The WMT24/25 metrics shared tasks put **XCOMET** and **MetricX** at the top and
  recommend combining the three for a comprehensive picture; no single score is
  trusted alone.
  ([WMT25 ranking](https://arxiv.org/html/2508.14909v1),
  [xCOMET, TACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00683/124263/),
  [MetricX-24](https://arxiv.org/pdf/2410.03983))
- **XCOMET-XL is the current large neural metric.** One checkpoint does **both**
  reference-free QE *and* reference-based scoring, and emits **error spans**,
  superseding CometKiwi. 3.5B params; XXL is 10.7B.
  ([Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL))
- **The LLM judge should follow GEMBA-MQM.** A fixed few-shot prompt extracting
  **MQM error spans with severities** (critical/major/minor), language-agnostic,
  96.5% system-level accuracy at WMT23. A single judgment is noisy, so **aggregate
  several**. ([GEMBA-MQM](https://aclanthology.org/2023.wmt-1.64/))
- **chrF++ is the Uzbek/Turkic string-metric standard.** Character-n-gram F-score
  suits morphologically rich languages; used by Uzbek MT papers on FLORES-200.
  ([Filling the Gap for Uzbek](https://arxiv.org/pdf/2508.14586))
- **Human EN↔UZ gold sets exist.** FLORES-200 `uzn_Latn` devtest (1,012
  professionally-translated sentences), NTREX-128 (1,997, Microsoft), turkic_xwmt
  (~1-2k, native-validated). General/formal domain, but independent human anchors
  for calibrating the metrics.
- **LLM-generated references bias evaluation toward LLM systems.** Reference-free QE
  is preferred for low-resource pairs where references are scarce.
  ([Reference-less QE survey](https://www.mdpi.com/2078-2489/16/10/916),
  [MQM-APE](https://arxiv.org/pdf/2409.14335))

**Design consequence.** The **reference-free backbone** — XCOMET-QE + GEMBA-MQM +
structural gates — decides the in-domain ranking, needs no golden set, and covers
all 9 systems. Reference-based scoring runs only on **external human refsets**, as
a cross-check on the metric rather than as the decision.

---

## 2. Systems under test

Nine systems behind one `translate(text, context=None) -> str` interface.

| # | System id | Family | Serving | Notes |
|---|---|---|---|---|
| 1 | `nllb-1.3b` | NLLB-200 enc-dec MT | local | incumbent production translator; entity-masking + sentence-split; **baseline** |
| 2 | `nllb-3.3b` | NLLB-200 enc-dec MT | local | `facebook/nllb-200-3.3B`, same wrapper |
| 3 | `gemma4:12b` | Gemma 4 (general LLM) | local Ollama | quantized |
| 4 | `gemma4:26b` | Gemma 4 (general LLM) | local Ollama | quantized |
| 5 | `gemma4:31b-cloud` | Gemma 4 (general LLM) | Ollama Cloud | quant fixed by hosting |
| 6 | `translategemma:12b` | TranslateGemma (MT-tuned) | local Ollama | quantized |
| 7 | `translategemma:27b` | TranslateGemma (MT-tuned) | local Ollama | quantized |
| 8 | `gemini-3.5-flash` | Gemini (LLM) | Google API | minimal thinking; **also the judge** — see [§6.3](#63-conflict-of-interest-gemini-is-both-a-system-and-the-judge) |
| 9 | `neuronai-uzbek` | Uzbek-specialized (Qwen3-4B FT) | local HF | `NeuronUz/NeuronAI-Uzbek`; chat template, greedy + repetition guards |

### 2.1 Fixed translation prompt (all LLM systems)

```
You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


{TEXT}
```

Instantiated as `English (en) → Uzbek (uz)`. Identical across every LLM system, so
prompt engineering is not a confound. NLLB uses its FLORES codes
(`eng_Latn → uzn_Latn`) through the existing wrapper.

**Determinism:** temperature 0 for LLMs, `num_beams=4` for NLLB. All outputs pass
through `normalize_uz()` (apostrophe standardization) before scoring.
`context_prev` is supplied to LLM translators as prior dialogue context; only the
target turn is requested and scored.

---

## 3. Metrics

| Metric | Type | Reference? | Runs on | Role |
|---|---|---|---|---|
| Structural checks | rule-based | no | all 600 | hard gate (script, entity/placeholder preservation, degeneration, empties) |
| **XCOMET-QE** (`Unbabel/XCOMET-XL`) | neural QE | **no** | all 600 | **primary unbiased adequacy score** |
| **GEMBA-MQM** (Gemini judge) | LLM error-span | no | all 600 | MQM error score + qualitative failure tags |
| XCOMET (reference mode) | neural | yes | human refsets | reference-based cross-check |
| chrF++ (sacrebleu) | string | yes | human refsets | morphology-friendly Uzbek standard |

XCOMET-QE + GEMBA-MQM + the structural gates form the **decision backbone**
(unbiased, all systems, in-domain). The refset metrics add a human-anchored
cross-check that validates the backbone.

---

## 4. Phase 0 — build the data

### 4.1 In-domain benchmark

Turn-level English segments drawn from the project's SFT corpus, stratified by
**source corpus** (12) and **category** (`caller_turn`, `agent_reply`,
`kb_passage`, `tool_spoken`, `edge`). **600 full / 60 smoke.** Deterministic seed,
deduplicated, fragments under 3 words dropped (except `edge`).

`preserve[]` is computed by `entity_mask.mask()` and holds the tokens a translation
must reproduce verbatim — flight codes, phone numbers, URLs, `{{placeholders}}`.
Localizable numeric amounts (money, quantities) are deliberately **excluded** from
`preserve[]`: a translator is allowed to render `$25,000` in Uzbek convention.

### 4.2 External human reference sets

**No LLM-authored golden set.** Research shows LLM references bias evaluation
toward LLM systems, and here Gemini is already both a system and the judge, so
authoring the references too would be triply circular. Instead, three existing
human sets, all independent of every system under test:

| Set | Domain | Uzbek size | Note |
|---|---|---|---|
| FLORES-200 `uzn_Latn` devtest | wiki/news | 1,012 | NLLB's home benchmark — mildly favors NLLB; keep, don't over-weight |
| NTREX-128 (uz) | news (WMT19) | 1,997 | Microsoft, professional, SacreBLEU-compatible |
| turkic_xwmt en↔uz | talks + news | ~1-2k | native-speaker validated; "talks" is closest to spoken register |

Using all three neutralizes any single set's bias.

### 4.3 In-domain scoring is reference-free

On the 600 customer-support dialogue segments there are no human references, and none are
authored. The in-domain ranking — the decision that actually picks a translator —
rests entirely on the reference-free backbone. This is the research-recommended
approach for low-resource pairs where trustworthy references are scarce.

**Two layers:**
- **In-domain** (the real target) → reference-free → decides the ranking.
- **Out-of-domain** (FLORES/NTREX/turkic_xwmt) → reference-based, human → validates
  the metric and gives an unbiased general EN→UZ head-to-head.

---

## 5. Phase 2 — structural gates

`structural_checks.py`, 100% of candidates, no GPU. Per-system rates for:

- **Language/script ID** — Uzbek **Latin**, not English pass-through, not Cyrillic drift.
- **Copy-through** (`uz == en`), **entity preservation** (every `preserve[]` token
  verbatim), **length-ratio outliers**, **degeneration** (empty output, n-gram loops).

A high-COMET system that drops `WEBSITEURL` or a flight number is disqualified for
the call-agent use case regardless of fluency, so the gates are reported beside the
score rather than folded into it.

Language ID uses `lingua` when installed and falls back to a stopword heuristic.
The shipped `structural.jsonl` reproduces exactly under the fallback.

---

## 6. Phase 3 — neural and LLM-judge scoring

### 6.1 XCOMET-QE (primary, reference-free)

`Unbabel/XCOMET-XL` (3.5B) in QE mode: score each `(en_text, uz_text)` pair 0-1
with no reference, across all 9 systems × 600 segments. GPU, batched.

**Long-segment chunking.** XCOMET's XLM-R encoder truncates at 512 subword tokens.
Anything past the cut is invisible to the metric, so a long-but-correct translation
scores near zero — in the first full run, segments over 200 English words averaged
QE 0.295 *for every system*, and those shared floor scores compressed the gaps
between systems. Sources longer than `--max-words` (default 80) are therefore split
into sentence-aligned chunks, scored separately, and recombined as a
source-word-weighted mean. 56 of the 600 segments exceed the threshold.
`--no-chunk` reproduces the old truncating behaviour.

### 6.2 GEMBA-MQM judge (Gemini)

Fixed few-shot prompt; the judge returns a list of MQM error spans, each with
`severity ∈ {critical, major, minor}` and `category ∈ {mistranslation, omission,
addition, grammar, wrong-script, entity-dropped, fluency}`. Score is the MQM
weighting (critical/major −5, minor −1), plus `adequacy`/`fluency` on 1-5 as a
secondary readout. `context_prev` is shown to the judge.

Following GEMBA-MQM v2, **N=3 independent passes are averaged** — cheap on Flash,
and it cuts judge noise. The passes sample at temperature 0.3; at temperature 0 the
N runs are identical decodes and averaging them measures nothing.

### 6.3 Conflict of interest: Gemini is both a system and the judge

`gemini-3.5-flash` translates **and** judges, which invites self-preference bias.
Mitigations: the ranking is anchored on **XCOMET-QE + structural gates**, both
Gemini-independent; Gemini's own GEMBA-MQM row is flagged wherever it appears.

---

## 7. Phase 5 — aggregation and statistics

`aggregate.py` → `LEADERBOARD.md`.

- **Per-system:** XCOMET-QE mean with **95% bootstrap CIs**, structural pass-rate
  and entity preservation, GEMBA-MQM (when run), and reference-mode XCOMET / chrF++
  on the refsets.
- **Paired comparison vs the baseline:** a one-sided **paired-bootstrap p-value**
  (`p(base)`) plus explicit **win/loss/tie counts** at ±0.01. Two systems can sit
  0.004 apart on the mean and still disagree on hundreds of segments; the means
  alone hide that.
- **Empty-output handling.** An empty translation cannot be scored, so it silently
  leaves its system's mean — flattering exactly the systems that failed. The
  leaderboard therefore reports a **paired mean** over the segments *every* system
  scored, an explicit `empty` count, and a statement of whether the ranking survives
  the stricter assumption that an empty output scores 0.0.
- **Shared-segment restriction on the refsets.** NLLB was run on all 1,997 NTREX
  segments while the paid LLMs were capped, so refset means are computed only over
  segments every system translated.
- **Metric agreement:** system-level correlation between the reference-free QE mean
  and the reference-based XCOMET mean. Agreement is the evidence that the QE
  backbone measures translation quality rather than noise.
- **Missing components are named, not blanked.** A metric whose score file is absent
  is listed under *Not run in this release*.

---

## 8. Known caveats

- **Gemini self-judge bias** — anchored out via XCOMET-QE and the structural gates.
- **Reference sets are formal-domain** (wiki/news/talks), not customer-support dialogue.
  Reference-based scores are a general EN→UZ signal plus metric validation; the
  in-domain ranking stays on reference-free QE over the 600 segments.
- **FLORES favors NLLB** (its home benchmark) — mitigated by also using NTREX and
  turkic_xwmt.
- **QE calibration for Uzbek is weaker** than for high-resource languages. Treat
  XCOMET scores as a **relative** ranking, cross-checked by the gates, the judge,
  and the metric-agreement correlation.
- **Test-set contamination** — FLORES and NTREX are old and public, so LLMs may have
  seen them in pretraining, which inflates their refset scores relative to NLLB. The
  in-domain benchmark is novel and contamination-free, another reason it is primary.
- **Quantization is not controlled.** The Ollama models are quantized and
  `gemma4:31b-cloud`'s quantization is fixed by the host, so a size comparison
  across those rows is confounded.
- **No human evaluation.** The ranking is automatic-metrics-only. A ~50-segment
  native-speaker MQM spot-check of the top systems remains the one validation no
  code substitutes for; it is deliberately parked, not done.
