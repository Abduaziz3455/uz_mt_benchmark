# Attribution & Licensing

This benchmark redistributes English text derived from twelve public dialogue
corpora, and machine translations of that text produced by nine models. The two
carry **different** licenses, and the model outputs are the more restrictive of the
pair. Read both sections before reusing anything here.

---

## 1. Code

`uz_mt_bench/` is licensed **Apache-2.0**. See [LICENSE](LICENSE).

`uz_mt_bench/entity_mask.py` and `uz_mt_bench/nllb_translator.py` are vendored
unchanged from the parent `icall_dataset` project so this directory runs
standalone.

---

## 2. The benchmark segments (`data/eval/uz_mt_benchmark.jsonl`)

600 English turns, 50 from each of 12 source corpora. Every record carries a
`section` field naming its upstream source, so you can filter by license.

| `section` | Source | License | Attribution |
|---|---|---|---|
| `sgd` | [Schema-Guided Dialogue (DSTC8)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | **CC-BY-SA-4.0** | Schema-Guided Dialogue Dataset, Google Research |
| `taskmaster` | [Taskmaster-1/2/3](https://github.com/google-research-datasets/Taskmaster) | CC-BY-4.0 | Taskmaster Dataset, Google Research |
| `multiwoz` | [MultiWOZ 2.2](https://huggingface.co/datasets/tuetschek/multi_woz_v22) | MIT | MultiWOZ 2.2, Budzianowski et al. |
| `star` | [STAR](https://github.com/RasaHQ/STAR) | MIT | STAR schema-guided dialogues, Rasa |
| `abcd` | [ABCD](https://github.com/asappresearch/abcd) | MIT | Action-Based Conversations Dataset, ASAPP Research |
| `soda` | [SODA](https://huggingface.co/datasets/allenai/soda) | CC-BY-4.0 | SODA, Kim et al., AllenAI |
| `bitext_cs` | [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | **CDLA-Sharing-1.0** | Bitext Customer Support LLM dataset |
| `bitext_banking` | [Bitext Retail Banking](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) | **CDLA-Sharing-1.0** | Bitext Retail Banking LLM dataset |
| `bitext_insurance` | [Bitext Insurance](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) | **CDLA-Sharing-1.0** | Bitext Insurance LLM dataset |
| `bitext_media` | [Bitext Media/Streaming](https://huggingface.co/datasets/bitext/Bitext-media-llm-chatbot-training-dataset) | **CDLA-Sharing-1.0** | Bitext Media LLM dataset |
| `talkmap_banking` | [talkmap banking corpus](https://huggingface.co/datasets/talkmap/banking-conversation-corpus) | MIT | talkmap banking conversation corpus |
| `talkmap_telecom` | [talkmap telecom corpus](https://huggingface.co/datasets/talkmap/telecom-conversation-corpus) | MIT | talkmap telecom conversation corpus |

### Collection license: CC-BY-SA-4.0

The inputs mix permissive (MIT), attribution (CC-BY-4.0) and **share-alike**
(CC-BY-SA-4.0, CDLA-Sharing-1.0) terms. The collection is released under
**CC-BY-SA-4.0**, the most restrictive of its constituents. If you redistribute
these segments or a derivative of them, you must do so under compatible terms and
preserve the attributions above.

**Note on SODA:** SODA is GPT-distilled. If your licensing posture forbids
GPT-derived data, exclude `section == "soda"` (50 of 600 segments).

`build_benchmark.py` regenerates the benchmark from the parent project's SFT
corpus, which is **not** shipped here. The 600 sampled segments are.

---

## 3. The candidate translations (`data/eval/candidates/`)

Machine-translated Uzbek output from nine systems. **Each system's output inherits
that model's license**, which is not the license of the source text.

| System | Model | Output license / terms |
|---|---|---|
| `nllb-1.3b`, `nllb-3.3b` | [NLLB-200](https://huggingface.co/facebook/nllb-200-3.3B) | **CC-BY-NC-4.0 — non-commercial** |
| `gemma4-12b`, `gemma4-26b`, `gemma4-31b-cloud` | Gemma 4 | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| `translategemma-12b`, `translategemma-27b` | TranslateGemma | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| `gemini-3.5-flash` | Gemini | [Google APIs Terms of Service](https://developers.google.com/terms) |
| `neuronai-uzbek` | [NeuronUz/NeuronAI-Uzbek](https://huggingface.co/NeuronUz/NeuronAI-Uzbek) | see the model card |

> **The NLLB rows are non-commercial.** NLLB-200's weights are CC-BY-NC-4.0 and its
> output inherits that restriction. Any corpus translated with `nllb-1.3b` or
> `nllb-3.3b` — including the `data/eval/candidates/*/nllb-*.jsonl` files here —
> cannot be used commercially. This is precisely the constraint that motivated the
> benchmark: finding out whether a commercially-usable translator matches NLLB's
> quality.

The score files in `data/eval/scores/` contain only `{id, system, score}` tuples
and no translated text.

---

## 4. Reference sets — not redistributed

FLORES-200, NTREX-128 and turkic_xwmt are **not** included in this repository.
`fetch_refsets.py` downloads them from their upstream sources on demand:

| Set | Source | License |
|---|---|---|
| FLORES-200 | [facebookresearch/flores](https://github.com/facebookresearch/flores/tree/main/flores200) | CC-BY-SA-4.0 (gated on HF) |
| NTREX-128 | [MicrosoftTranslator/NTREX](https://github.com/MicrosoftTranslator/NTREX) | CC-BY-SA-4.0 |
| turkic_xwmt | [HF `turkic_xwmt`](https://huggingface.co/datasets/turkic_xwmt) | CC-BY-4.0 |

`data/eval/candidates/{ntrex,flores}/` contains only the systems' Uzbek output and
segment ids — no source sentences and no human references. Re-running the
reference-mode scoring requires fetching the refsets first.

---

## 5. Metrics

| Component | Source | License |
|---|---|---|
| XCOMET-XL | [Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) | Apache-2.0 (gated repo — accept the license on HF) |
| `unbabel-comet` | [Unbabel/COMET](https://github.com/Unbabel/COMET) | Apache-2.0 |
| sacrebleu (chrF++) | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | Apache-2.0 |
| GEMBA-MQM prompt | [Kocmi & Federmann, WMT 2023](https://aclanthology.org/2023.wmt-1.64/) | method, reimplemented here |

---

## 6. Summary

- **Reuse the code** → Apache-2.0, no conditions beyond notice.
- **Reuse the English benchmark segments** → CC-BY-SA-4.0, attribute the twelve
  sources above, share alike.
- **Reuse the NLLB Uzbek translations** → non-commercial only (CC-BY-NC-4.0).
- **Reuse the other systems' translations** → under each model provider's terms.
- **Reuse the scores** → they are measurements, shipped under the code's Apache-2.0.
