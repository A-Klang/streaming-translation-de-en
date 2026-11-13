"""Hand-labelling sheet to check the RQ2 judge: 50 steps from the LA s1 runs of both arms,
half of them flagged as guesses by the judge and half not, shuffled. The judge's labels go
to a separate key file, so the sheet can be filled in blind. Run from the repository folder."""
import json
import random
from pathlib import Path

RUNS = {
    "hymt2-q4": ("results/pipeline/hymt2-q4_la-s1/hymt2-q4_la-s1_20261105-012425.jsonl",
                 "results/judge/hymt2-q4_la-s1_20261105-012425_haiku.jsonl"),
    "gemma-e2b-q4": ("results/pipeline/gemma-e2b-q4_la-s1/gemma-e2b-q4_la-s1_20261106-032529.jsonl",
                     "results/judge/gemma-e2b-q4_la-s1_20261106-032529_haiku.jsonl"),
}
PER_CELL = {("hymt2-q4", True): 13, ("hymt2-q4", False): 12,       # (arm, judge flagged) -> steps drawn
            ("gemma-e2b-q4", True): 12, ("gemma-e2b-q4", False): 13}
OUT = Path("rq2_validation")
rng = random.Random(2026)

# every judged step, with what the judge saw
pool = {cell: [] for cell in PER_CELL}
for arm, (run_file, judge_file) in RUNS.items():
    runs = {r["idx"]: r for r in map(json.loads, open(run_file))}
    for j in map(json.loads, open(judge_file)):
        r = runs[j["idx"]]
        src, fixed = r["src"].split(), ""
        for s in r["steps"]:
            label = j["labels"].get(str(s["t"]))
            pool[(arm, label is not None)].append(dict(
                arm=arm, idx=r["idx"], t=s["t"], german=" ".join(src[:s["t"]]),
                fixed=fixed, new=s["text"].strip(), judge=label))
            fixed = s["committed"]

items = [x for cell, n in PER_CELL.items() for x in rng.sample(pool[cell], n)]
rng.shuffle(items)

rules = open("pipeline/rq2_prompt.txt").read().split("For each step you get:")[0].strip()
lines = ["RQ2 judge check: label each step the way the judge was asked to.",
         "Write NONE, or GUESS followed by the guessed words in quotes, after LABEL:",
         "", "The judge's instructions:", "", rules, "", "=" * 70]
for i, x in enumerate(items, 1):
    lines += ["", f"### {i:02d}", f"GERMAN: {x['german']}", f"FIXED:  {x['fixed']}",
              f"NEW:    {x['new']}", "LABEL: "]
(OUT / "sheet.txt").write_text("\n".join(lines) + "\n")
(OUT / "key.json").write_text(json.dumps(
    [dict(item=i, **{k: x[k] for k in ("arm", "idx", "t", "judge")}) for i, x in enumerate(items, 1)],
    ensure_ascii=False, indent=1))
print(len(items), "steps written to", OUT / "sheet.txt")
