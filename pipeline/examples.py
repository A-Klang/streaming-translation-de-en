"""Candidate examples for the report, from the M3 labels (results/judge/m3_*): committed guesses
that turned out wrong, (a) under LA s1 for each arm, and (b) where LA s1 committed a wrong guess
in a sentence where LA s1 with the verb rule committed none. Writes results/examples.md.
From the repository folder:  python pipeline/examples.py"""
import json
import random
from pathlib import Path

from bootstrap import run_path

items = json.loads(Path("results/judge/m3_items.json").read_text())
labels = {j["idx"]: j for j in map(json.loads, open("results/judge/m3_haiku.jsonl"))}
rng = random.Random(1)
wrong = {}  # run -> list of (idx, item, answer)
for idx, its in items.items():
    j = labels[int(idx)]
    for x, lab, ans in zip(its, j["labels"], j["answers"]):
        if lab == "WRONG":
            for occ in x["runs"]:
                wrong.setdefault(occ["run"], []).append((int(idx), x, ans))

def final(run, idx):
    arm, setting = run.split("_", 1)
    return next(r for r in map(json.loads, open(run_path(arm, setting))) if r["idx"] == idx)

out = ["# Candidate examples (M3 labels, not yet hand-checked)", ""]
def block(run, idx, x, ans):
    r = final(run, idx)
    seen = len(x["seen"].split())
    src = r["src"].split()
    return [f"**{run}, devtest {idx}**", "",
            f"- German: {' '.join(src[:seen])} ‖ {' '.join(src[seen:])}",
            f"- committed: {x['before']} {x['new']}".rstrip(),
            f"- judge: listener hears *{ans['listener_understands']}*; German says *{ans['german_says']}*",
            f"- final output: {r['hyp']}", f"- reference: {r['ref']}", ""]

for arm in ("hymt2-q4", "gemma-e2b-q4"):
    run = f"{arm}_la-s1"
    out += [f"## Wrong committed guesses, {run}", ""]
    for idx, x, ans in rng.sample(wrong[run], 4):
        out += block(run, idx, x, ans)

out += ["## Verb rule: LA s1 commits a wrong guess, LA s1 + verb rule commits no wrong guess in that sentence", ""]
for arm in ("hymt2-q4", "gemma-e2b-q4"):
    with_rule = {i for i, _, _ in wrong.get(f"{arm}_la-s1-verb", [])}
    cands = [w for w in wrong[f"{arm}_la-s1"] if w[0] not in with_rule]
    for idx, x, ans in rng.sample(cands, min(3, len(cands))):
        out += block(f"{arm}_la-s1", idx, x, ans)
        out += [f"- with the verb rule: {final(f'{arm}_la-s1-verb', idx)['hyp']}", ""]

Path("results/examples.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
