"""One table of every report run: quality, lag, compute and guessing, with the bootstrap's
95% intervals for COMET, LAAL and the RQ2 counts (results/summary.json, from bootstrap.py).
Writes results/report_table.csv and results/report_table.md. From the repository folder:
    python pipeline/report_table.py
"""
import csv
import json
from pathlib import Path
from statistics import mean

from sacrebleu.metrics import BLEU, CHRF

from bootstrap import ARMS, FULL, SETTINGS, run_path
from score import sentence_laal

NAMES = {"hymt2-q4": "Hy-MT2", "gemma-e2b-q4": "Gemma"}
summary = json.loads(Path("results/summary.json").read_text())

rows = []
for arm in ARMS:
    for s in SETTINGS + [FULL]:
        runs = [json.loads(line) for line in open(run_path(arm, s))]
        refs = [[r["ref"] for r in runs]]
        hyps = [r["hyp"] for r in runs]
        words = sum(r["n_src"] for r in runs)
        b = summary[f"{arm}_{s}"]
        row = dict(arm=NAMES[arm], setting="full sentence" if s == FULL else s,
                   comet=b["comet"]["point"], comet_lo=b["comet"]["lo"], comet_hi=b["comet"]["hi"],
                   chrf=CHRF().corpus_score(hyps, refs).score, bleu=BLEU().corpus_score(hyps, refs).score,
                   laal=b["laal"]["point"], laal_lo=b["laal"]["lo"], laal_hi=b["laal"]["hi"],
                   ca_laal_s=mean(sentence_laal(r)[1] for r in runs if r["delays"]),
                   ms_per_word=sum(r["compute_ms"] for r in runs) / words,
                   calls_per_sentence=mean(r["calls"] for r in runs),
                   prompt_tokens_per_word=sum(r["prompt_tokens"] - r["cached_tokens"] for r in runs) / words)
        for m in ("raw", "guesses", "wrong"):
            if m in b:
                row.update({m: b[m]["point"], f"{m}_lo": b[m]["lo"], f"{m}_hi": b[m]["hi"]})
        rows.append(row)

fields = list(dict.fromkeys(k for r in rows for k in r))
with open("results/report_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    w.writerows(rows)

ci = lambda r, m, d: f"{r[m]:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]" if m in r else "–"
lines = ["| Arm | Setting | COMET | chrF | BLEU | LAAL (words) | CA-LAAL (s) | ms/word | calls/sent. | raw guess % | committed /100w | wrong /100w |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['arm']} | {r['setting']} | {ci(r, 'comet', 3)} | {r['chrf']:.1f} | {r['bleu']:.1f} | "
                 f"{ci(r, 'laal', 2)} | {r['ca_laal_s']:.2f} | {r['ms_per_word']:.0f} | {r['calls_per_sentence']:.1f} | "
                 f"{ci(r, 'raw', 1)} | {ci(r, 'guesses', 2)} | {ci(r, 'wrong', 2)} |")
Path("results/report_table.md").write_text(
    "COMET, chrF, BLEU, LAAL, CA-LAAL and compute on devtest 300 to 1011 (712 sentences); the guess columns on the "
    "100-sentence RQ2 sample. Brackets: 95% bootstrap intervals (10,000 rounds).\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
