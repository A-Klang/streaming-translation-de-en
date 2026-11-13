# Changelog

Changes to the scripts after the 23 September version: the pilot scripts until 28 September, the pipeline in `pipeline/` from 30 September. Newest last. Backups named below live in `archived/backups/`; retired result folders in `archived/results/`. `pilot_en-src_backup.py` is pilot.py before any of these changes, `pilot_v2_backup.py` is pilot.py before the llama.cpp backend, and `walk_vN_backup.py` is version N of walk.py.

## 27 September 2026

### pilot.py: source language flag (backup: pilot_en-src_backup.py)

Why: the project switched direction to German to English, so Axel can read and judge the output himself. The script had English hardcoded as the source.

- Added `--src` (default `eng_Latn`, so old commands work unchanged). `--langs` is now explicitly the target.
- Added English to the language table, and an Opus-MT model table per direction (de-en, zh-en added).
- NLLB now takes its source language from `--src` instead of always English.
- Gemma's prompt names the source language ("Translate the following German text into English"). HY-MT's prompt has no source language, per its model card. HY-MT uses its Chinese prompt when Chinese is on either side.
- Output files for non-English sources are named `<src>-<tgt>`, e.g. `results/hymt15/deu_Latn-eng_Latn.jsonl`. English-source runs keep the old names, so the 27 Sep English to German results are untouched.
- The prefix test splits Chinese sources by character (no spaces), other sources by space.

### pilot.py: crash fix

- The new helper `src_words` clashed with a local variable of the same name in `run_arm` (UnboundLocalError on the first run). Renamed the helper to `split_src`.

### walk.py: new script, word by word streaming walk (v1)

Why: to see real streaming on one sentence instead of the single half sentence snapshot in the prefix test.

- Feeds the source one word at a time. At each step the model gets the source so far plus the English already shown as the forced start of its answer.
- What gets shown is decided by local agreement: a word is shown once two consecutive steps agree on it. At the last source word the full output is shown.

### walk.py v2: table output

Why: the stacked three line output was hard to read.

- Terminal output is a table (Step, New German word, Model's full translation at this step, Added to screen) that fits the terminal width.
- Also writes the table as markdown to `results/walk/<arm>/<src>-<tgt>_<i>.md`.

### walk.py v3: punctuation aware agreement

Why: in the first run, words were compared with punctuation attached. `“We”` and `“We` did not match, which delayed "We" one step, and `old.”` did match, which showed a sentence end and closing quote far too early and turned "he added" into "He added that."

- Words are compared without surrounding punctuation.
- A word that ends a sentence or quote is shown without that ending, and nothing after it is shown until the source sentence ends.
- `--rule plain` gives the v1 behaviour back. Markdown files now carry the rule in their name (`..._0_plain.md`, `..._0_clean.md`).

### walk.py v4: "On screen so far" column

- Added a fifth column with the full text shown so far at each step.

### walk.py v5: sentence splits and closing quotes

Why: v3 stalled from step 12 to 17 on sentence 0. The model split the German sentence into two English sentences, and v3 refused to show anything after the first full stop, even when consecutive steps agreed on the second sentence.

- A sentence end is shown whole if both steps also agree on at least 2 words after it (`MIN_AFTER`), which means the model has really split the sentence. Otherwise the v3 behaviour applies.
- Closing quotation marks are never shown before the source has closed its own quote.
- Continuations starting with a hyphen or closing bracket are joined without a space (fixes "4 -month-old").

### pilot.py: quantized arms through llama.cpp

Why: Gemma 4 E2B does not fit in the Air's 8 GB in bf16, and Elja access is not working yet. Quantized through llama.cpp is also how both models will run on the Pi. To keep the arms comparable, HY-MT gets a quantized arm too.

- New arm kind `server`: pilot.py starts llama.cpp's `llama-server` (Metal on the M1), which downloads the GGUF on first use, and talks to it over HTTP on port 8089.
- New arms `gemma-e2b-q4` (unsloth/gemma-4-E2B-it-GGUF, Q4_K_M, 3.1 GB) and `hymt15-q4` (tencent/HY-MT1.5-1.8B-GGUF, Q4_K_M, 1.1 GB, Tencent's own quantization).
- Same prompts and the same prefix forcing as the bf16 arms: the server applies the chat template stored in the GGUF, with thinking disabled, the forced words are appended as plain text, and greedy decoding (temperature 0) completes it. A leading BOS from the template is removed because the server adds its own.
- Memory for these arms is the server process's resident memory, not MPS memory, so it is not directly comparable with the bf16 numbers.
- The server log is written to `results/<arm>/llama-server.log`.
- walk.py accepts the quantized arms too (`--arm gemma-e2b-q4`).

### pilot.py: Hy-MT2 quantized arm (backup: pilot_v3_backup.py)

Why: Hy-MT2-1.8B (Tencent, May 2026) reports about 3 XCOMET points over HY-MT1.5-1.8B on FLORES-200 and ships an official Q4_K_M GGUF of the same size (1.13 GB). Tested before the full runs so a switch of Arm B costs nothing.

- New arm `hymt2-q4` (tencent/Hy-MT2-1.8B-GGUF, Q4_K_M), same server path, greedy decoding and prefix forcing as `hymt15-q4`. Prompt is the existing `HYMT2_PROMPT` from the model card, no system prompt.
- walk.py picks it up automatically (`--arm hymt2-q4`).

### pilot.py: incomplete-input prompt arms (backup: pilot_v4_backup.py)

Why: in streaming the input is incomplete at almost every step, so telling the model is the honest description of the task. The same instruction goes to both arms, so it is not per-model prompt tuning. Also a check of Hy-MT2's claimed better instruction following.

- `PARTIAL_NOTE`: "The text may stop mid-sentence. Translate only what is there, and do not complete the sentence." Inserted after the first sentence of each arm's existing prompt, identical wording for both.
- New arms `gemma-e2b-q4-partial` and `hymt2-q4-partial`: same GGUF files and settings as `gemma-e2b-q4` and `hymt2-q4`, only the prompt differs. Results land in their own folders, so the model-card runs stay untouched.
- `hymt2-q4` role changed from B-candidate to B: Hy-MT2 replaces HY-MT1.5 as Arm B (decided 27 Sep after its pilot run).

### Result: incomplete-input prompt not adopted (27 Sep, no code change)

Gemma follows the instruction (half sentences 19/20 faithful instead of 18) but loses 0.008 COMET on full sentences; Hy-MT2 ignores it. Template overhead roughly doubles for both. Default stays `gemma-e2b-q4` (plain prompt) and `hymt2-q4` (model-card prompt). The `-partial` arms stay in the script so walk.py can compare both Gemma prompts under streaming.

### Cleanup: results/archived (27 Sep)

Why: keep `results/` to the two current arms.

- Moved to `results/archived/`: `hymt15`, `hymt15-q4` (predecessor), `nllb`, `opus` (former fallbacks), `gemma-e2b-q4-partial`, `hymt2-q4-partial` (incomplete-input prompt check), `walk` (the HY-MT1.5 walks on sentence 0) and the top-level `smoke` folder.
- `results/` now holds `gemma-e2b-q4`, `hymt2-q4` and `summary.json`. The summary keeps the archived arms' scores.
- `pilot.py --score-only` only looks one level down (`results/*/*.jsonl`), so archived runs are not rescored. walk.py recreates `results/walk/` on its next run.

### Cleanup: whole folder (27 Sep)

- Top level now holds only what is in use: `pilot.py`, `walk.py`, `show_prefix.py`, the two requirements files, this changelog, `results/` and the two venvs.
- All `*_backup.py` files moved to `archived/backups/`.
- The retired result folders moved from `results/archived/` to `archived/results/`, so the archive is in one place and `results/` only has current arms.

### walk.py: default arm (backup: archived/backups/walk_v6_backup.py)

- Default `--arm` changed from `hymt15` (unquantized HY-MT1.5, retired) to `hymt2-q4`, the current Arm B. Usage example in the docstring updated to match.

### Repository prep (27 Sep)

- Added `README.md` (overview, layout, setup, commands, method notes) and `.gitignore` (both venvs, `__pycache__`, `.DS_Store`) ahead of putting the folder on GitHub.

## 28 September 2026

### tagger_prefix.py: new script, spaCy on German prefixes

Why: the planned chunk policy decides on German grammar while the sentence is still incomplete, so the tagger has to be reliable on fragments.

- Cuts every devtest sentence after each word, tags the fragment with spaCy (de_core_news_lg) and compares each new word's tags with its tags in the full sentence. A small state machine labels clause events (subordinate clause opens, finite verb, clause closes).
- Result (n=100): clause events are reliable on arrival except particles and clause openings; with a one-word delay (`--lag 1`) every event is at 0.97 to 1.0 precision and recall. Output in `tagger_results/`.

### agree_test.py: new script, wrong guesses under local agreement

Why: to measure how often two agreeing steps still commit a guess about German that has not arrived yet.

- Runs walk.py's local agreement over many sentences and logs every step, so each commit can be checked against the German seen at that moment. Reruns skip finished sentences.
- `--probs N` also logs token probabilities, `--verb-rule` applies verb_gate.py, `--ids judged_ids.json` restricts the run to the 100 hand-judged sentences.
- Result (devtest 100 to 299, 100 judged): Hy-MT2 committed 31 wrong guesses, Gemma 10. 37 of 41 were verbs guessed before the German verb arrived, the rest invented numbers. A confidence threshold did not help (the models are confidently wrong).

### verb_gate.py: new script, the verb rule

Why: almost all wrong guesses were verbs, and German puts the verb that carries the meaning late.

- An English verb or number may only be committed if the German that has arrived contains at least as many. German auxiliaries count only once their clause has closed without a full verb.
- Result: wrong guesses Hy-MT2 31 → 8, Gemma 10 → 6, for about 0.35 to 0.47 words of extra LAAL.

## 30 September to 5 October 2026

### pipeline/: new, own streaming pipeline (v1)

Why: a clean, readable pipeline written for the study, instead of building on the dense pilot scripts.

- `data.py` loads FLORES+ by sentence id, `backend.py` sends one raw `/completion` request to a hand-started llama-server (greedy, prefix forcing, timings and token counts), `policies.py` has wait-k and local agreement, `simulate.py` runs a policy over sentences on a 2.5 words/s virtual clock with compute-aware delays, `score.py` gives chrF, BLEU, optional COMET, LAAL, CA-LAAL, tokens and compute per source word. All commands in `commands.md`.
- Hy-MT2 only. The doubled BOS warning is fixed by leaving BOS out of the template; the server adds it.

## 14 October 2026

### pipeline: Gemma arm and server check

Why: RQ1 needs both arms in the same pipeline.

- `backend.py`: a template per arm (Gemma's copied from the server's `/apply-template` with thinking off), `TEMPLATES` and `MODELS` tables, `arm` passed to `build_prompt` and `translate`.
- Policies take the arm; `simulate.py` has `--arm`, puts it in the output filename, and stops if the server has a different model loaded (checked through `/props`).

### pipeline: word-glue fix

Why: "4" was committed, the next reply continued with "-month-old", and the output read "4 -month-old".

- A reply that starts without a space continues the last committed word. Also attaches closing quotes correctly. Checked for both arms.

### pipeline: clean rule for local agreement

Why: exact word comparison made "it" and "it," disagree, and sentence ends and closing quotes could be committed early (the same problems walk.py v3 to v5 fixed).

- Words are compared without surrounding punctuation and case; the newer reply's version is committed.
- A sentence end is committed only with at least two agreed words after it (also holds "Dr." until the name follows).
- No closing quote is committed while the German quote („...“) is still open.
- Simplification against walk.py: a held word waits whole instead of being committed without its ending.

### pipeline: verb rule (verb_rule.py)

Why: port of verb_gate.py into the pipeline, so local agreement can run with and without it.

- `german_counts` counts German verbs and numbers (auxiliaries only once their clause closes, German ordinals as numbers). `hold_verbs` cuts the agreed words before the first English verb or number the German has not delivered.
- spaCy tokens are mapped to pipeline words by counting the spaces before each token.
- `LocalAgreement(verb_rule=...)` and `--verb-rule` in simulate.py, off by default.

### pipeline: wait-k catch-up

Why: wait-k asked for exactly one word per step, so a step that returned nothing left it permanently one word further behind.

- Each step asks for the words missing from the schedule (`len(seen) - k + 1` minus what is committed).

### pipeline: prompt caching

Why: a real system would cache, so measuring without it reports latency nobody would see.

- `CACHE_PROMPT` in backend.py (on). simulate.py records cached tokens next to computed ones; score.py prints both per source word.
- Outputs identical with and without caching (10 sentences). Gemma needs `--swa-full`; with it, compute fell from 440 ms (no caching) to 260 ms per German word.

### pipeline: stride for local agreement

Why: local agreement needs a latency knob to give a curve instead of one point per model. Polák et al. (2022) found the chunk size to be the main knob, and a larger agreement count only moves along the same curve at more compute.

- `LocalAgreement(stride=s)` calls the model only every s German words and at the end of the sentence. `--stride` in simulate.py (default 1); output files are named `<arm>_la-s<stride>[-verb].jsonl`.
- Stride 3 on sentence 0: 6 calls instead of 18, same final sentence, first commit at word 6.

### Documentation

- README: status, layout with the pipeline, pipeline commands, method notes on caching, `--swa-full`, words as the unit and the data split, spaCy model downloads.
- requirements.txt: added spacy and requests.

### pipeline: per-step records

Why: timings are now used only to compare configurations on one device (M1, Metal), not as absolute speeds, so compute has to be reportable per call and per token type.

- `run_sentence` keeps a `steps` list, one record per model call: `t` (German words seen), computed, cached and output tokens, prompt and decode ms, and `n_new` (English words committed). Sentence totals unchanged, so score.py is untouched.
- Sentence 0, Hy-MT2, LA + verb: 18 records, sums match the totals, 6 of 18 calls commit nothing.

### leak_check: German words in the committed English

Why: check whether German words leaking into the output come from how the models are called or from the models themselves.

- `leak_check/leak_check.py` flags committed words that are in the German source but not in the English reference, and compares each flag with the same arm's full-sentence run (`--policy waitk --k 999`). Runs in `leak_check/results/pipeline/`, all flags in `leak_check/leaks.txt`.
- Devtest 0 to 99, both arms, wait-3 and LA + verb: Hy-MT2 copies the opening German word in 4 sentences (8, 37, 56, 86) under both policies, all translated correctly with the full sentence; Gemma none. The other flags are heuristic false positives.

## 24 October 2026

### pipeline: raw replies in the per-step records

Why: RQ2 is about what the models write beyond the German, and most of that never gets committed. Saving every reply means the guesses can be evaluated later without rerunning the models.

- Each record in `steps` now also has `text`, the model's raw continuation after the forced English. The English that was forced at a step is not saved separately: it is the committed words with a delay below that step's `t`.
- Checked on devtest 0, Hy-MT2, LA: 18 of 18 records have the text. Step 6 already shows an uncommitted guess ("-month-old babies" before "Mäuse" has arrived).

### Memory measure and Gemma without its image encoder

Why: RQ1b reports memory, and the pilot's resident memory of the server process misses the Metal buffers.

- Memory per arm is read from llama-server's load log (start with `-v`): weights as the GGUF file size, plus the KV cache and compute buffers. The log's CPU and GPU mapped weight buffers overlap (both views of the same memory-mapped file, together larger than the file), so the file size is used instead of their sum. These buffers are allocated once at load, so the total is the peak. Reported as one total, since the M1 has unified memory.
- Study flags (`-c 4096`, Metal): Hy-MT2 1081 MiB weights + 256 MiB KV + 60 MiB compute ≈ 1.40 GiB; Gemma 2963 + 72 + 162 MiB ≈ 3.12 GiB. Gemma's KV cache is a quarter of Hy-MT2's because only 15 of its layers keep their own.
- `-hf` had also downloaded Gemma's image encoder (`mmproj-BF16.gguf`), and the server loaded it on every start: about 1.1 GiB of GPU memory, unused for text. Gemma's server command now has `--no-mmproj` (commands.md). Translations and token counts of earlier runs are unaffected.

## 3 November 2026

### pipeline: one folder per setting, no overwrites

Why: every run replaced the last file with the same settings, so a quick test or demo run could wipe out a real result (the 30-sentence Hy-MT2 LA file the judge runs use, for example).

- simulate.py writes to `results/pipeline/<arm>_<policy>/<arm>_<policy>_<YYYYMMDD-HHMMSS>.jsonl` and opens the file in `x` mode, so an existing file is never overwritten. The settings stay in the filename, so judge.py and align.py, which name their output after the input file, need no change.
- Existing files in `results/pipeline/` and `leak_check/results/pipeline/` moved into these folders, timestamped with their last-modified time. Older names without a stride (`la`, `la-verb`) keep their own folders.
- leak_check.py reads the newest file in each setting's folder. Output checked identical to before the move.


### pipeline: sentence-end hold for wait-k

Why: Hy-MT2 ends cut-off German with "…" or "." (it does so with no forcing at all). Wait-k committed that ending, and with "…" in the forced prefix the model then ended every later reply immediately. On devtest 5, wait-3 committed "after…" and produced nothing for five German words. Local agreement already had a sentence-end hold, wait-k didn't, so the comparison wasn't fair.

- `WaitK.hold_sentence_ends`: before the German has finished, an English sentence end is committed only once the German has delivered one more than the English already has (the same counting idea as the verb rule). Otherwise the ending is stripped and the word kept ("workshop." -> "workshop"); a reply that is only "…" commits nothing. Counting rather than always stripping keeps real sentence breaks (devtest 4) and abbreviations ("Dr.").
- Checked on devtest 0 to 9 with Hy-MT2 wait-3: only sentence 5 changes. It now keeps pace and guesses instead of stalling: "...after he left his garage workshop to use the doorbell, but couldn't hear it."
- Earlier wait-k runs (including the 18 Oct leak check runs) are from before this change. Hy-MT2 wait-3 had "…" in the committed output in 8 of devtest 0 to 99, against 1 for every other run.

### simulate.py: model replies in the live table

Why: local agreement commits what two consecutive replies agree on, and the live table only showed the result. With the replies visible you can see the policy work, for example on devtest 0 where the model has "they" right at two steps, then switches to "we" and that gets committed.

- `--replies` (with `--live`) prints the model's raw reply under each row that made a call, dimmed. Display only, saved results are unchanged.


## 4 November 2026

### pipeline: unattended overnight runs

Why: the report runs (2 arms, 9 settings, devtest 300 to 1011) take roughly 14 hours and run overnight, so a single failed request or a server crash must not end a night's work.

- backend.py: `translate` retries a failed request up to 3 times (after 5, 15 and 45 s) before raising.
- simulate.py: each result line is flushed as soon as it is written, so a hard stop loses at most the sentence in progress. `--resume FILE` continues an interrupted run in the same file: it skips the sentences already there, drops a line cut off by a hard stop, and refuses a file from another setting. simulate.py now prints `writing <file>` at the start.
- pipeline/night.sh: new. For one arm, starts its llama-server, waits for `/health`, runs the 9 settings (wait-k 1, 3, 5; local agreement at stride 1, 2, 3, each with and without the verb rule) on devtest 300 to 1011, then stops the server. A setting that fails is resumed in its own file, up to 3 times, with the server restarted first if it is down. Logs in `logs/night_<arm>_<time>/`; the server log is gitignored.
- Checked with Hy-MT2 on devtest 0 to 3: all 9 settings ran; killing the server during wait-1 led to a restart and a resume, and the file ended with sentences 0 to 3 exactly once. Test files deleted afterwards.

### pipeline: demo-day changes (4 Nov)

- policies.py: local agreement strips a trailing "…" from each reply before the sentence is finished (Hy-MT2 marks cut-off input with it; it is not a real sentence end), the counterpart of the wait-k sentence-end hold of 3 Nov. Active in all report runs of 5 and 6 Nov.
- simulate.py: with `--live`, each sentence first shows the model's full-sentence translation for the audience to compare against. Display only, not counted in any statistic.

## 6 November 2026

### comet_probe: what COMET penalises

Why: RQ2 is about wrong meaning reaching the committed output. If COMET barely moves for that kind of error, a COMET difference can't stand in for an error count, and RQ2 needs the judge or annotation.

- `comet_probe/comet_probe.py` scores five versions of six devtest sentences (300, 303, 310, 330, 350, 360) against the FLORES reference: correct, German word order, wrong verb, negation, wrong entity or number. It also scores seven real outputs from the 5 and 6 Nov runs. Output in `comet_probe/comet_probe_out.txt`. Run from the repository folder with `.venv-comet`.
- Mean difference to the correct version: German word order −0.088, wrong verb −0.054, negation −0.067, wrong entity or number −0.023 ("72 accidents" for 27: −0.004; "much larger prey" for smaller: 0.000). COMET penalises awkward but correct word order more than a wrong meaning.
- Real outputs point the same way: Gemma LA s3 with "otters" for ocelots scores 0.814, above its garbled but roughly faithful wait-1 output (0.800); Hy-MT2 wait-1 with a German leak ("Die Kommission … Martellys") scores 0.880.
- Caveats: six sentences, one reference, variants written by hand, and the word-order versions are harsher than the models' real literal output. Illustration only, not a measurement to quote.

### pipeline: room for a whole sentence in wait-k's last call

Why: the full-sentence baseline (`--policy waitk --k 999`) translates each sentence in one call at the end, but that call was capped at 64 tokens, too few for the longest devtest sentences (up to 55 German words).

- `WaitK` now gives its last call of a sentence `FINAL_TOKENS = 256`. Other calls are unchanged.
- No call in the 18 report runs of 5 and 6 Nov reached 64 tokens, so with greedy decoding their output would be the same under the new cap; they stay valid.

### pipeline: Grammar safe for unattended runs

Why: two things in `GrammarChunk` would break or distort an overnight run. It committed every reply whole, so Hy-MT2's "…" or "." at the end of a cut-off chunk went into the output (the wait-k stall of 3 Nov). And a reply that hit the token limit raised `RuntimeError`, which night.sh would resume into the same crash three times and then give up on the whole setting.

- Wait-k's sentence-end hold is now a shared function, `hold_unearned_ends`, and Grammar applies it to every reply before the German has finished: an English sentence end is committed only once the German has delivered one more than the English has. Wait-k behaves as before (devtest 300 to 309, wait-3: same output and delays as the report run).
- A reply cut off by the token limit before the German has finished commits nothing and leaves the chunk open, so the next boundary translates it again with more German. At the end of the sentence the reply is kept, as in wait-k and local agreement.
- simulate.py: each step record now has `stop`, the model's stop type (`eos` or `limit`), so cut-off replies can be counted for every policy.
- Checked with Hy-MT2 on devtest 0 to 19: 82 calls, no limit hits, no "…" in the committed output; many replies ended with "." or ".”" mid-sentence and all were held. Test file deleted afterwards.

### pipeline: Grammar gets the same output rules as the other policies

Why: Grammar should differ from wait-k and local agreement only in when it commits, not in what it is allowed to commit.

- Closing-quote hold: while the German quote is still open, Grammar drops closing quote marks from the reply (`added,”` -> `added,`), the condition local agreement uses. Local agreement cuts before such a word; Grammar commits a whole chunk at once, so it keeps the words. Like local agreement it only sees German quote marks („ “), not ASCII quotes in the source (devtest 45).
- Wait-k has no closing-quote hold. In the report runs it committed an early closing quote in 2 to 7 of 712 sentences per run (local agreement: 0). Small enough not to rerun; to be mentioned in the report.
- spaCy now tags the whole German prefix and the boundary rules look only at the tokens not yet translated, instead of tagging the untranslated fragment on its own (the September tagger test measured tags on prefixes). Offline check on devtest 0 to 99: identical chunks in 99 of 100 sentences (259 vs 260 boundaries), so the change barely matters for these rules, which decide mostly on punctuation and the newest word. The one difference (devtest 51, "dass niemand garantieren | kann,") shows the real weak point: the newest word's tag, which the September test found unreliable on arrival without a one-word delay.
- Checked with Hy-MT2 on devtest 0 to 49: 197 calls, no limit hits; 11 replies closed a quote before the German did and all were held. Test file deleted afterwards.

### pipeline: Grammar delay (one word of lookahead before judging a word)

Why: Grammar decides on the newest German word, whose tag is often only settled by the next word (*an*, *auf*, *mit*, *zu* look like separable particles at the end of a prefix and turn out to be prepositions). On devtest 0 to 99 the newest word's tag differs from its full-sentence tag for 8.5% of words, 0.8% one word later and 0.2% two words later, and 27 of Grammar's 260 boundaries (10%) disappear with full-sentence tags ("Rechtsstreitigkeit mit | der …", "beriet er Carter in | der Außenpolitik").

- `GrammarChunk(delay=d)`, `--delay` in simulate.py (default 0, the original policy). With delay d the rules judge the word d places behind the newest, with the whole prefix tagged; the lookahead words are context for the tagger only, and the model gets the chunk up to the boundary as before. Results are saved under `<arm>_grammar-d<delay>`.
- Checked with Hy-MT2 on devtest 0 to 49: delay 0 chrF 63.8, LAAL 4.94 words, 197 calls; delay 1 chrF 63.9, LAAL 6.42 words, 186 calls (fewer wrong cuts, so longer chunks, so more lag than the one word). Sentence level clearly better where delay 0 cut wrongly: devtest 1 keeps the indirect speech instead of inventing a quote, 9 and 18 keep their prepositional phrases whole. Test files deleted afterwards.

### RQ2: judge prompt, sample and judge.py fixes

Why: RQ2 counts guesses in the raw replies (which model guesses more) and in the committed output (how much each policy and rule lets through), with an LLM judge, on a fixed sample so the runs can be compared sentence by sentence.

- Judge prompt: early full stops are no longer guesses (Axel's call of 31 Oct; about 240 of the 298 steps flagged on 31 Oct were only that). The prompt now lives in the repo as `pipeline/rq2_prompt.txt`; judge.py still runs `claude -p` from `~/judge_test` for isolation. The old version is kept there as `rq2_prompt_2026-10-29.txt`.
- `rq2_ids.json`: 100 sentences drawn from devtest 300 to 1011 (`random.Random(2026)`), mean length 21.5 German words, the same as all 712. judge.py has `--ids` to judge only those.
- judge.py found the claude binary by a hardcoded extension version (2.1.314), which an extension update had removed; it now picks the newest installed version.
- Checked with Haiku on devtest 302, Hy-MT2 LA s1: 24 of 24 steps labelled in 16 s, early full stops not flagged, 7 guesses. Six look right ("They were" for "Es wurden", "six cases", "four explosions", "were injured" before "Verbrennungen"); "this led to…" before "ausgelöst" was missed. Test output deleted.

### RQ2: hand-labelling sheet for the judge check

Why: the judge's labels need a human check before the report can lean on them. Michael won't annotate, so Axel labels a small sample.

- `rq2_validation/make_sheet.py` draws 50 steps from the LA s1 runs of both arms on 10 of the RQ2 sentences (`rq2_validation/ids10.json`): 25 the judge (Haiku) flagged as guesses and 25 it didn't, split across the arms and shuffled, since only about 10 to 30% of steps are guesses and a random draw would hold too few. `sheet.txt` shows the judge's instructions and, per step, the German so far, the fixed English and the new reply; the judge's labels and the arm are only in `key.json`.
- judge.py: the label parser also accepts `<step12> NONE`, a form Haiku sometimes copies from the prompt (devtest 998, Gemma: 0 of 20 steps parsed). Labels of the 20 judged sentences re-read from the saved raw answers; all steps labelled.
- Haiku on those 10 sentences, LA s1: Hy-MT2 72 guesses in 221 steps, Gemma 19 in 221. Before the hand check.
- Judge check done: Haiku and ChatGPT (web) agree on 46 of 50 steps (kappa 0.84); Axel settled the four disagreements, three for ChatGPT. Haiku wrong on 3 of 50, leaning towards missed guesses. Details in `rq2_validation/adjudication.md`; `compare.py` computes the agreement.

### RQ2: counts from the judge's labels (rq2_count.py)

Why: turns the judge's per-step labels into the RQ2 numbers per run: how often the model guesses, how much of it reaches the output, and how much each policy lets through.

- `pipeline/rq2_count.py`, one line per judged run: raw (share of replies with a guess), committed guesses per 100 German words, share of sentences with at least one committed guess, passed (committed / raw guesses), and how many quoted guesses could not be found in their reply.
- A guess counts as committed if its quoted words start within the step's first `n_new` words. The last step of each sentence is left out, since the whole German has arrived by then and anything unsupported is a mistranslation, not a guess (devtest 790: "services." at the final step).
- Judge run started 6 Nov: Haiku on the 100 RQ2 sentences of all 18 report runs (27,674 steps, 1800 calls), log in `logs/judge_run.log`.
- First pass finished with two gaps: Hy-MT2 wait-5 was never judged (the shell loop skipped the file list's last line, which had no newline), and 36 sentences had steps without a label, mostly because Haiku wrote yet another form (`<step> 12 NONE`). The parser now skips any leading `<…>` tag or "STEP" before the number and accepts `12: NONE`; all saved answers were re-read, and the 10 sentences still missing a label for some step were dropped and judged again with Hy-MT2 wait-5.
- judge.py checks each answer itself: if any step has no label, it asks again, up to 3 tries, and marks a sentence that stays incomplete. Completeness is checked per step number (`t`), not by label count, since steps above stride 1 are numbered by German word.
- Result, Haiku on the 100 RQ2 sentences of all 18 report runs, in `results/judge/rq2_counts.txt`: Hy-MT2 guesses in 28% of its LA s1 replies, Gemma in 10%; committed, Hy-MT2 has about twice as many guesses per 100 German words at every setting. The verb rule cuts committed guesses by 31 to 75%, and local agreement with the verb rule commits 1.7 to 3 times fewer guesses than wait-k at equal or lower lag.

## 7 November 2026

### Grammar sweep (night.sh, results)

Why: the Grammar policy goes into the report as it is, with the lookahead delay as its knob (agreed with Michael).

- `night.sh` takes an optional second argument: `grammar` runs `--policy grammar --delay 0, 1, 2` instead of the 9 main settings. Log folders now carry the group name.
- Ran both arms on devtest 300 to 1011 (6 runs, 712 sentences each, about 14 min per setting for Hy-MT2 and 19 for Gemma). COMET in `results/pipeline/scores_grammar.txt`: Hy-MT2 0.874 / 0.877 / 0.878 at LAAL 4.99 / 6.29 / 7.18, Gemma 0.880 / 0.882 / 0.883 at LAAL 5.07 / 6.38 / 7.27; compute 47 to 50 ms per word (Hy-MT2) and 67 to 69 ms (Gemma).
- Haiku judged the same 100 sentences (`results/judge/*grammar*_haiku.jsonl`). `rq2_counts.txt` now covers all 24 runs (counted from the timestamped judge files only). Committed guesses per 100 German words: Hy-MT2 1.35 / 0.84 / 0.75, Gemma 0.56 / 0.37 / 0.47. Every Grammar reply is committed whole, so "passed" is 100% by construction.

### Paired bootstrap (bootstrap.py, score.py, rq2_count.py)

Why: the differences between settings are often a few thousandths of COMET, so each claim needs a significance test (Koehn 2004).

- `score.py --comet` also writes `<run>.comet.json` next to each run, with the sentence scores by idx. Sentence LAAL moved into `sentence_laal()`, with unchanged numbers. All 26 report runs (24 streaming, 2 full-sentence) rescored.
- `rq2_count.py`: the counting is now per sentence (`per_sentence()`), so the bootstrap can reuse it. The output is identical.
- New `pipeline/bootstrap.py`: 10,000 rounds, seed 2026. One draw of the 712 sentences (COMET, LAAL) and one of the 100 RQ2 sentences (committed guesses) per round, shared by every run, so every comparison is paired. It gives a 95% interval per run, the arm difference at the same setting and at matched lag (linear interpolation along each policy curve at whole-word LAAL), verb rule on against off at the same setting and at matched lag, the Grammar delays, and each run against its full-sentence baseline. Guesses are matched on LAAL over the same 100 sentences. Output in `results/bootstrap.txt`.
- Findings: Gemma is ahead on full sentences (+0.004 COMET) and at low lag (wait-k at 3 words +0.013), level from about 5 words up, and ahead under Grammar (+0.005). It commits 0.1 to 0.5 words earlier and guesses significantly less at matched lag in almost every family. At matched lag the verb rule adds no COMET (all within ±0.0015), but it cuts committed guesses by 30 to 45% at 5 to 6 words (p < 0.004).

### M3, raw-rate test, throttling check, report table, figures, examples

Why: everything the report needs, run on the final results.

- `pipeline/m3_judge.py` and `m3_prompt.txt`: every committed guess (1149, 1044 distinct) is judged once by Haiku against the complete German and the FLORES reference, RIGHT or WRONG for a listener. Structured output through `--json-schema`, with what the listener understands and what the German says written down before the label; no examples in the prompt (they biased the judge towards the example answers); `--effort low`; up to 15 guesses per call. `judge.call_judge` takes a prompt file and extra CLI arguments. Labels in `results/judge/m3_haiku.jsonl`, items in `m3_items.json`.
- `rq2_count.py` counts wrong committed guesses (per 100 German words and as a share); `bootstrap.py` tests them, and the raw guess rate, at the same setting and (wrong guesses) at matched lag. It also writes `results/summary.json`.
- `m3_validation/`: a blind 40-guess sheet (20 RIGHT, 20 WRONG by Haiku; sentences 302, 309, 325 left out) to paste into ChatGPT, and `compare.py` for κ and the disagreements. Not run yet.
- `results/throttle_check.txt`: decode speed slows 7 to 20% over the first run of a session, flat afterwards.
- `pipeline/report_table.py`: one table of every run with intervals (`results/report_table.md`, `.csv`).
- `pipeline/figures.py`: four figures in `results/figures/` (PDF and PNG). matplotlib added to requirements.txt.
- `pipeline/examples.py`: candidate examples of wrong committed guesses (`results/examples.md`).
- `results/findings.md`: the findings and method notes for the report.

## 10 November 2026

### M3 judge check

Why: the M3 labels (Haiku) needed the same check against a second judge as RQ2.

- `m3_validation/chatgpt_labels.txt`: ChatGPT's labels on the 40-guess sheet. compare.py: 35/40, kappa 0.75.
- `m3_validation/settled.md`: the 5 disagreements settled by Axel; Haiku agrees with the settled labels on 39/40.
- `results/findings.md`: the method note says so.
