# Judge check, 6 Nov 2026

The 50 steps in `sheet.txt` were labelled by the judge (Claude Haiku 5.5, `key.json`) and, independently, by ChatGPT (web version, temporary chat, model name not shown; `chatgpt_labels.txt`, from `paste_to_chatgpt.txt`). `python rq2_validation/compare.py rq2_validation/chatgpt_labels.txt`:

- Agreement 46 of 50 (92%), Cohen's kappa 0.84. Both guess 23, both none 23, only Haiku 2, only ChatGPT 2. Hy-MT2 steps 25 of 25, Gemma 21 of 25.
- For comparison: Haiku, Sonnet and Opus agreed on 91% of 595 steps on 31 Oct (kappa 0.87 to 0.90).

The four disagreements, settled by Axel with the full-sentence reference:

| Step | Haiku | ChatGPT | Decision |
|---|---|---|---|
| 11, "leitet damit einen Wandel" -> "initiating a change" | GUESS | NONE | Haiku: the particle *ein* of *einleiten* has not arrived; ChatGPT cited it anyway |
| 22, "man ... tragen will" -> "they want to wear" | GUESS | NONE | ChatGPT: "they" translates the generic *man*, which has arrived |
| 40, "keine gebügelten Socken" -> "have ironed socks" | NONE | GUESS | ChatGPT: no verb has arrived (*tragen will* comes later; reference "wearing") |
| 49, "besonders bei Schulabsolventen" -> "true for" | NONE | GUESS | ChatGPT: the predicate *beliebt* has not arrived (reference "popular") |

Result: Haiku wrong on 3 of 50 steps (1 false guess, 2 missed guesses, both invented predicates), ChatGPT on 1. Haiku, if anything, undercounts guesses. Haiku is used for the RQ2 runs.
