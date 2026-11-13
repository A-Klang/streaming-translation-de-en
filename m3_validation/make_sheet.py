"""Check of the M3 judge: 40 committed guesses, 20 that Haiku labelled RIGHT and 20 WRONG,
shuffled, for a second labeller (ChatGPT) with the same instructions. Sentences 302, 309 and
325 are left out: they were looked at while the prompt was written. The judge's labels go to
key.json, so the sheet is blind. Run from the repository folder."""
import json
import random
from pathlib import Path

OUT = Path("m3_validation")
SEEN = {302, 309, 325}
rng = random.Random(2026)

items = json.loads(Path("results/judge/m3_items.json").read_text())
labels = {json.loads(l)["idx"]: json.loads(l)["labels"] for l in open("results/judge/m3_haiku.jsonl")}
rows = {r["idx"]: r for r in map(json.loads, open(next(Path("results/pipeline/hymt2-q4_waitk999").glob("*.jsonl"))))}
pool = {"RIGHT": [], "WRONG": []}
for idx, its in items.items():
    if int(idx) in SEEN:
        continue
    for k, (x, lab) in enumerate(zip(its, labels[int(idx)])):
        pool[lab].append(dict(idx=int(idx), k=k, judge=lab, **x))
picked = rng.sample(pool["RIGHT"], 20) + rng.sample(pool["WRONG"], 20)
rng.shuffle(picked)

rules = Path("pipeline/m3_prompt.txt").read_text().strip()
lines = [f"Label each of the {len(picked)} numbered guesses below according to these instructions.", "",
         "Answer with exactly one line per guess, in order, and nothing else:",
         "01 RIGHT | <what a listener understands> / <what the German says>",
         "02 WRONG | <what a listener understands> / <what the German says>", "",
         rules, "", "=" * 70]
for i, x in enumerate(picked, 1):
    r = rows[x["idx"]]
    lines += ["", f"### {i:02d}", f"GERMAN: {r['src']}", f"REFERENCE: {r['ref']}",
              f"SEEN: {x['seen']}", f"BEFORE: {x['before']}", f"NEW: {x['new']}"]
(OUT / "paste_to_chatgpt.txt").write_text("\n".join(lines) + "\n")
(OUT / "key.json").write_text(json.dumps(
    [dict(item=i, idx=x["idx"], k=x["k"], judge=x["judge"], new=x["new"]) for i, x in enumerate(picked, 1)],
    ensure_ascii=False, indent=1))
print(len(picked), "guesses written to", OUT / "paste_to_chatgpt.txt")
