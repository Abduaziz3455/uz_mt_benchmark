# EN→UZ Benchmark — Worked Examples

Hand-picked side-by-side comparisons from the 600-segment customer-support dialogue
benchmark. Sources: model outputs in `data/eval/candidates/uz_mt_benchmark/`,
per-segment XCOMET-QE scores in `data/eval/scores/xcomet_qe_uz_mt_benchmark.jsonl`,
joined by segment `id`. Regenerate/extend with any JSONL join on those files.

**Read the caveats first:** these are *cherry-picked extremes* chosen to illustrate
failure classes — they are not typical behavior for any system shown (Gemini ranks
4th overall at 0.818; its median segment is fine). Scores are per-segment XCOMET-QE,
a relative signal for Uzbek. None of the Uzbek outputs have been verified by a native
speaker; the commentary describes what the metrics and gates flagged, not a human
quality judgment. Ranking and methodology: [LEADERBOARD.md](LEADERBOARD.md),
[METHODOLOGY.md](METHODOLOGY.md).

## 1. Numbers: `$10,000` survives, gets verbalized, or flips meaning

Segment `soda-093924-t1` · category `tool_spoken`

> **EN:** Hi, I'm Madelaine. I filed a claim for $10,000 with my insurance company and I'm hoping to get the money quickly so that I can pay off my debts.

| system | XCOMET-QE | output |
|---|---|---|
| `gemma4-26b` | 0.91 | Salom, men Madelayn. Sugʻurta kompaniyamga 10 000 dollar miqdorida talabnoma berdim va qarzlarimni toʻlash uchun pulni tezroq olishni umid qilyapman. |
| `tilmoch` | 0.74 | Salom, men Madlenman. Men sugʻurta kompaniyamga 10 000 dollarlik daʼvo arizasini topshirdim va qarzlarimni toʻlash uchun pulni tezroq olishga umid qilyapman. |
| `gemma4-12b` | 0.69 | Salom, mening ismim Madelin. Men sugʻurta kompaniyamga 10,000 dollar miqdorida daʼvo topshirdim va qarzlarimni toʻlash uchun ushbu pulni tezroq olishni umid qilaman. |
| `gemini-3.5-flash` | 0.58 | Salom, men Madlenman. Sugʻurta kompaniyamga oʻn ming dollarlik daʼvo arizasi topshirgan edim va qarzlarimni uzish uchun bu pulni tezroq olishga umid qilyapman. |
| `nllb-1.3b` | 0.46 | Salom, men Madlenman. Men sugʻurta kompaniyamga $10,000 uchun talabnoma topshirdim va pulni tezda toʻlashim va qarzlarimni toʻlashim uchun umid qilaman. |

`gemma4-26b` keeps the amount as digits and the meaning intact. `gemini-3.5-flash` verbalizes it as *oʻn ming dollarlik* ("ten-thousand-dollar") — fine for TTS, but the token `$10,000` is no longer recoverable downstream. `nllb-1.3b` keeps the digits but garbles the second clause: *pulni tezda toʻlashim... umid qilaman* reads as if the *caller* hopes to pay the money quickly, inverting who pays whom.

## 2. Product terms: "4G LTE" as a token vs. "Toʻrt G"

Segment `talkmap_telecom-084260-t1` · category `kb_passage`

> **EN:** Hello Jan thank you for calling Union Mobile. My name is Aida, and I'll be happy to assist you with your concern. Can you please tell me a little bit more about the issue you're experiencing with your 4G LTE connectivity?

| system | XCOMET-QE | output |
|---|---|---|
| `tilmoch` | 0.97 | Salom Jan, Union Mobileʼga qoʻngʻiroq qilganingiz uchun tashakkur. Mening ismim Aida va men sizga yordam berishdan xursand boʻlaman. 4G LTE aloqasi bilan bogʻliq muammo haqida batafsil maʼlumot bera olasizmi? |
| `gemma4-26b` | 0.95 | Salom Jan, Union Mobile-ga qoʻngʻiroq qilganingiz uchun rahmat. Mening ismim Aida, muammoingizni hal qilishda sizga yordam berishdan mamnunman. Iltimos, 4G LTE aloqasidagi muammo haqida menga biroz koʻproq maʼlumot bera olasizmi? |
| `gemma4-12b` | 0.95 | Assalomu alaykum Jan, Union Mobilega qoʻngʻirogʻingiz uchun rahmat. Mening ismim Aida va men sizga yordam berishga tayyorman. Iltimos, 4G LTE aloqasi bilan bogʻliq muammo haqida biroz batafsilroq maʼlumot bera olasizmi? |
| `gemini-3.5-flash` | 0.61 | Aloqa, Jan, Union Mobile kompaniyasiga qoʻngʻiroq qilganingiz uchun rahmat. Mening ismim Aida, sizga yordam berishdan mamnunman. Toʻrt G LTE aloqasi bilan bogʻliq qanday muammoga duch kelayotganingiz haqida batafsilroq aytib bera olasizmi? |
| `nllb-1.3b` | 0.69 | Salom Jan, "Union Mobile"ga qoʻngʻiroq qilganingiz uchun rahmat. Mening ismim Aida va men sizga tashvishlaringiz bilan yordam berishdan mamnunman. Iltimos menga 4G LTE ulanishingiz bilan duch kelayotgan muammo haqida koʻproq aytib bera olasizmi? |

`tilmoch` tops this segment (0.97) and both Gemma sizes are just behind: all three keep `4G LTE` verbatim and open naturally. `gemini-3.5-flash` renders "Hello" as *Aloqa* ("Connection") and spells out `4G` as *Toʻrt G* ("Four G") — instantly, audibly wrong to a native listener, and exactly the kind of error a voice agent cannot afford. Note that this is a `kb_passage`, the category where `tilmoch` is strongest relative to the field (0.720 vs 0.676–0.695 for the NLLB pair) — scripted, information-dense agent copy is the shape of text it handles best.

## 3. Terminology: "direct debit" — and why NeuronAI ranks last

Segment `talkmap_banking-083991-t2` · category `caller_turn`

> **EN:** Hi Madeleine, I'm calling to confirm a direct debit.

| system | XCOMET-QE | output |
|---|---|---|
| `gemma4-26b` | 0.96 | Salom Madeleine, men avtomatik toʻlovni tasdiqlash uchun qoʻngʻiroq qilyapman. |
| `gemma4-12b` | 0.92 | Salom, Madeleine, men toʻgʻridan-toʻgʻri toʻlovni tasdiqlash uchun qoʻngʻiroq qilyapman. |
| `tilmoch` | 0.74 | Salom Madlen, toʻgʻridan-toʻgʻri debetni tasdiqlash uchun qoʻngʻiroq qilyapman. |
| `gemini-3.5-flash` | 0.71 | Salom, Madlen, men toʻgʻridan-toʻgʻri debet toʻlovini tasdiqlash uchun qoʻngʻiroq qilayotgan edim. |
| `nllb-1.3b` | 0.69 | Salom Madlen, men toʻgʻridan-toʻgʻri hisobni tasdiqlash uchun qoʻngʻiroq qilyapman. |
| `neuronai-uzbek` | 0.20 | Maldein! Meni maʼlum bir studentim bor va u men uni bashorat-taqlid ega edi? Siz bu gapni kim deb topmoqchisiz?-Uning notanishliklari "Shunday emas" - Bu har hil his-tuygʻularni ifоdash uchun eng mosdir: agar biz unga “Bu” ni beryapmiz-da , unda ham mana . Bundan tashqari ,"bu", "shu”, ”men”. Keyin esa :“Men". Mening parolimni ketkazaman ? Albatta ! Toʻgʻrimi ?. Yoʻq,-u xafa boladimim.-Ammo sen manimcha yahshil holga ketyapman ,-deb jasoratli tanaffusda javobini rad etdi."Yaxshi",-debdingiz-chi?"Buni bilmayapsizmi?”-"Ha," debbiz-a?", "-Demak,bir oz dam olasizlar!" Doʻkondan chetgacha qaerdadirgi restoranga borishadi.Meni chalgʻitadigan lahzalar...Menda nimasi bor?! Qaysi firibgarlikka boyroq!? Nega meni zeriktirib qoʻydiyu??? Haliyam |

`gemma4-26b` picks the idiomatic *avtomatik toʻlov*; the others go literal (*toʻgʻridan-toʻgʻri debet/toʻlov/hisob* — word-by-word "direct debit") which an Uzbek banking customer would find odd. `tilmoch` is the sharpest version of that point: a commercial, Uzbek-specialised MT service produces the most literal reading of all (*toʻgʻridan-toʻgʻri debet*, calquing even the English word *debet*), so language specialisation alone does not buy domain idiom — a general-purpose LLM that has seen Uzbek banking copy does better here than a dedicated translator that hasn't. `neuronai-uzbek`'s output is unrelated to the source — representative of why it sits at 0.25 QE overall, not a worst-case pick.

## 4. Perfect metric score, dropped entity — why the gates exist

Segment `soda-101937-t6` · category `tool_spoken` · preserve tokens: `100`

> **EN:** Katara, I support you 100%. You have my vote!

| system | XCOMET-QE | output |
|---|---|---|
| `gemini-3.5-flash` | 1.00 | Katara, men seni yuz foiz qoʻllab-quvvatlayman. Mening ovozim seniki! |
| `gemma4-26b` | 0.98 | Katara, seni 100 foiz qoʻllab-quvvatlayman. Mening ovozim seniki! |
| `nllb-1.3b` | 1.00 | Katara, men seni 100% qoʻllab-quvvatlayman. Sizda mening ovozim bor! |
| `tilmoch` | 0.92 | Katara, seni 100% qoʻllab-quvvatlayman. Senga ovoz beraman. |
| `translategemma-27b` | 0.92 | Katar, men sizni toʻliq qoʻllab-quvvatlayman. Mening ovozim sizniki! |

`gemini-3.5-flash` scores a *perfect 1.00* — the translation is genuinely fluent and accurate — yet the preserve-token `100` is gone, verbalized into *yuz foiz* ("hundred percent"). `translategemma-27b` (0.92) drops the number entirely (*toʻliq* — "fully") and corrupts the name Katara → *Katar*. XCOMET cannot see either failure; `entity_keep` flags both. This is the concrete case for reporting structural gates beside the score instead of folding them in.

## 5. Empty output: the failure a mean score hides

Segment `talkmap_banking-054272-t14` · category `edge`

> **EN:** No, just get me off your damn call list.

| system | XCOMET-QE | output |
|---|---|---|
| `gemma4-12b` | — (empty) | *(empty output)* |
| `gemma4-26b` | 0.76 | Yoʻq, shunchaki meni bu laʼnati qoʻngʻiroqlar roʻyxatingizdan olib tashlang. |
| `gemini-3.5-flash` | 0.75 | Yoʻq, shunchaki meni laʼnati qoʻngʻiroqlar roʻyxatingizdan oʻchirib tashlang. |
| `nllb-3.3b` | 0.98 | Yoʻq, shunchaki meni chaqirish roʻyxatingizdan olib tashlang. |
| `tilmoch` | 0.91 | Yoʻq, shunchaki meni chaqiruvlar roʻyxatingdan oʻchir. |

`gemma4-12b` returns an empty string for this mildly profane edge segment ("...your damn call list") — plausibly safety-conditioned reticence, and 1 of its 4 empties on the benchmark. An empty output is unscoreable, so it silently drops out of a naive mean; a production pipeline sees a customer getting silence. NLLB, a pure MT model, translates it without hesitation (0.98).

## 6. Noisy source: faithful translation vs. silent cleanup

Segment `star-008554-t12` · category `agent_reply`

> **EN:** I heard San Fran is great I heard San Fran is great

| system | XCOMET-QE | output |
|---|---|---|
| `gemini-3.5-flash` | 0.91 | San-Fransiskoni ajoyib deb eshitganman. |
| `translategemma-27b` | 0.81 | Men San-Fransisko haqida juda yaxshi gaplar eshitdim. |
| `gemma4-26b` | 0.51 | San-Fransisko juda yaxshi ekanligini eshitganman, San-Fransisko juda yaxshi ekanligini eshitganman. |
| `tilmoch` | 0.50 | Eshitishimcha, San-Frans juda yaxshi ekan. Eshitishimcha, San-Frans ajoyib ekan. |
| `gemma4-12b` | 0.43 | San Fransiko zoʻr ekanligini eshitdim, San Fransiko zoʻr ekanligini eshitdim. |

The source line itself repeats ("I heard San Fran is great" ×2) — real transcript noise from the source corpus, kept deliberately. `gemini-3.5-flash` and `translategemma-27b` silently deduplicate and QE rewards them; the Gemma models translate what was actually said and are penalized. Neither behavior is wrong — but they are different products: cleanup is nicer for TTS, faithfulness is safer for records. Know which one you're buying.

## 7. The same phenomenon at 4×: everyone fails a repeated source

Segment `taskmaster-003584-t19` · category `kb_passage`

> **EN:** your receipt will be texted to you and it will be ready for pickup in 10 minutes. your receipt will be texted to you and it will be ready for pickup in 10 minutes. your receipt will be texted to you and it will be ready for pickup in 10 minutes. your receipt will be texted to you and it will be ready for pickup in 10 minutes.

| system | XCOMET-QE | output |
|---|---|---|
| `gemma4-31b-cloud` | 0.19 | Kvitansiya sizga SMS orqali yuboriladi va buyurtmangiz 10 daqiqatdan soʻng olib ketishga tayyor boʻladi. |
| `gemini-3.5-flash` | 0.03 | Kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. Kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. Kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. Kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. |
| `gemma4-26b` | 0.06 | kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqa ichida olib ketish uchun tayyor boʻladi. kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqa ichida olib ketish uchun tayyor boʻladi. kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqa ichida olib ketish uchun tayyor boʻladi. kvitansiyangiz sizga SMS orqali yuboriladi va u 10 daqiqa ichida olib ketish uchun tayyor boʻladi. |
| `nllb-1.3b` | 0.10 | Ruxsatnomangiz sizga SMS orqali yuboriladi va u 10 daqiqada olish uchun tayyor boʻladi. Ruxsatnomangiz sizga SMS orqali yuboriladi va u 10 daqiqada olish uchun tayyor boʻladi. Ruxsatnomangiz sizga SMS orqali yuboriladi va u 10 daqiqada olish uchun tayyor boʻladi. Ruxsatnomangiz sizga SMS orqali yuboriladi va u 10 daqiqada olish uchun tayyor boʻladi. |
| `tilmoch` | 0.07 | chekingiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. chekingiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. chekingiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. chekingiz sizga SMS orqali yuboriladi va u 10 daqiqada olib ketishga tayyor boʻladi. |

Here the source repeats a sentence four times, and every system that translates it faithfully scores ≤ 0.10 — only `gemma4-31b-cloud` deduplicates (0.19). This single segment accounts for 7 of the 23 `degenerate` flags in `structural.jsonl`: the flag detects repetition in the *output* and cannot tell faithful-to-noisy-source apart from a true generation loop. True loops do occur — see the 397×-reference-length FLORES output noted in the leaderboard — but on this benchmark, check the source before blaming the model.

---

*Systems omitted from a table behaved unremarkably on that segment. All ten outputs
for every segment are in the candidate files.*
