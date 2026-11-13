# Findings for the report

Everything below is on FLORES+ devtest 300 to 1011 (712 sentences) for COMET, chrF, BLEU, LAAL and compute, and on the 100-sentence RQ2 sample (`rq2_ids.json`) for guessing. Intervals and p-values are from the paired bootstrap (Koehn 2004; 10,000 rounds, seed 2026, `pipeline/bootstrap.py`, output in `bootstrap.txt`). Full table: `report_table.md` / `.csv`. Figures: `figures/`. Examples for the report (chosen 10 Nov): devtest 564, 708, 648 and 860 in `examples.md`.

## RQ1: quality, lag and compute

- **Full sentences.** Gemma 0.892 COMET against Hy-MT2 0.888 (+0.004, p < 0.0001). chrF and BLEU separate the arms much more (68.0 against 65.2 chrF, 43.2 against 37.6 BLEU). Curly quotes explain only 0.8 BLEU of that. Gemma also has more near-exact matches with the reference (8 against 3 exact, 19 against 7 at sentence BLEU ≥ 80). That is a weak hint of wording memorised from FLORES (whose English is the original), not evidence. Mention as a caveat.
- **At the same setting Gemma commits earlier:** LAAL 0.1 to 0.5 words lower in every streaming setting (p < 0.0001), with COMET level. So the arms are compared at matched lag (each arm's COMET interpolated along its policy curve at whole-word LAAL).
- **Matched lag.** Gemma is ahead at low lag: wait-k at 3 words +0.013 (p < 0.0001), LA at 3 and 4 words +0.003 (p 0.046, 0.013). From about 5 words up the arms are level under wait-k and LA. Under Grammar, Gemma is ahead by +0.005 at 6 and 7 words (p < 0.0001).
- **Streaming costs quality.** Every run is significantly below its full-sentence baseline. The best (LA s3 with the verb rule) loses 0.004 (Hy-MT2) and 0.008 (Gemma). Wait-1 loses 0.068 and 0.056.
- **Policies.** LA dominates wait-k at the same lag. Grammar sits slightly below the LA curve in quality (Hy-MT2 d0 0.874 at 4.99 words against LA s2 0.878 at 4.49) but needs about 3 model calls per sentence instead of 7.5 to 21.5.
- **Compute.** Gemma needs 25 to 35% more compute per German word than Hy-MT2 under every policy (LA s3: 161 against 121 ms; full sentence 56 against 40 ms). LA s3 costs about 3 times a full-sentence translation, wait-k about 4 to 6 times, Grammar only 1.2 times (47 to 69 ms). Everything is far inside real time (400 ms per word at 2.5 words/s). Memory: 3.1 against 1.4 GiB.
- **Grammar delay.** d0 to d1: +0.002 to 0.003 COMET for 1.3 words of lag. d1 to d2: +0.0007 for 0.9 words. d1 is the sensible operating point.

## RQ2: guessing

- **Raw replies.** Hy-MT2 guesses in 24 to 38% of mid-sentence replies, Gemma in 8 to 16% (16 to 22 points lower under every wait-k and LA setting, p < 0.0001). Under Grammar both guess less (Hy-MT2 8 to 12%, Gemma 4 to 5%), because the model is only called at clause boundaries.
- **Committed guesses** (per 100 German words, what reaches the listener). At matched lag Gemma commits significantly fewer in every family, often about half: wait-k at 3 words 3.6 against 6.9, LA at 3 words 2.2 against 4.9 (both p < 0.0001). Grammar commits every reply whole, so nothing filters its guesses, but it guesses so rarely that its committed rate matches LA with the verb rule.
- **Wrong committed guesses (M3).** About a third of committed guesses turn out wrong against the complete German (Hy-MT2 30 to 45%, Gemma 20 to 40%). At matched lag Gemma commits significantly fewer wrong guesses at low lag: wait-k at 3 words 1.12 against 2.90 per 100 words, LA at 3 words 0.66 against 1.60, LA + verb rule at 4 words 0.36 against 0.97 (all p ≤ 0.001). From about 5 words up under LA the difference is no longer significant: too few wrong guesses in 100 sentences.
- **The verb rule.** At the same stride it buys COMET (+0.001 to 0.008) by waiting (+0.3 to 0.6 words). At matched lag it adds no COMET (all within ±0.0015), so on quality it is equivalent to waiting longer. But it cuts committed guesses by 30 to 45% at 5 to 6 words (p 0.004 and below). Wrong guesses fall by about half at those lags too, in the same direction at all six points, but only one point is clearly significant (Gemma at 5 words, p 0.025). Wording: the rule clearly cuts guessing, probably wrong guessing too, but the sample cannot confirm the latter.

## Method notes and limitations

- **Judges.** RQ2 guess detection is Haiku, checked against ChatGPT (web version, 6 Nov) on 50 steps: 46/50, κ 0.84, with the 4 disagreements settled by hand (Haiku slightly undercounts). M3 is Haiku with structured output (what a listener understands / what the German says / label), no examples in the prompt, low effort. Each distinct guess is judged once and its label shared across runs. Checked against ChatGPT (web version, 10 Nov) on 40 blind guesses (20 RIGHT, 20 WRONG by Haiku): 35/40, κ 0.75. Settled by hand, Haiku agrees on 39/40; in 4 of the 5 disagreements ChatGPT judged wording rather than meaning, and Haiku's one error is a WRONG that should be RIGHT, so if anything it slightly overcounts wrong guesses (`m3_validation/`).
- **M3 judges a guess when it is committed,** without the words that follow. A guess the continuation repairs ("what certain people are | motivated to do") still counts as wrong; 3 of the 15 candidate examples were like that, so M3 somewhat overstates lasting misleading guesses. It affects both arms alike, and the listener did hear the wrong reading for a moment.
- **M3 is against the German and the human reference,** not the model's own full-sentence translation. A wrong guess counts even if the model would have got it wrong with the whole sentence too, which measures what the listener hears.
- **All guessing counts as unwanted** (spoken output: a committed word cannot be revised). M3 shows how much of it is actually wrong.
- **Matched lag** interpolates linearly between 3 points per curve.
- **Multiple tests.** About 100 comparisons. Everything at p < 0.001 survives a Bonferroni correction; results around 0.05 are weak evidence.
- **Wait-k has no closing-quote hold** (LA and Grammar do). It affects 2 to 7 of 712 sentences per run.
- **Thermal behaviour** (`throttle_check.txt`). The fanless M1 Air slows 7 to 20% in decode speed during the first run of a session; runs on a warm machine are flat. Compute for wait-1 and the full-sentence baselines is a few percent high; no comparison changes.
- **Resumed sentences** are timed on their final attempt only.
- **Stream input** (unsegmented speech) was dropped: sentences arrive pre-segmented. Limitation and future work.
- **COMET** penalises German word order more than wrong content (`comet_probe/`: −0.088 against −0.023 for a wrong entity or number). This motivates the RQ2 guess measures alongside COMET.
- **Greedy decoding** (temperature 0, top-k 1) for both arms: deterministic, and a guess is the model's own most likely continuation.

## Future work

- Grammar: local agreement over its chunk replies, lookahead rules, ASCII quotes, a wider delay sweep.
- Stream input without sentence segmentation.
- M3 split into "streaming caused" and "model gets it wrong anyway" (a second pass against the model's full-sentence translation).
