"""Compare Haiku's M3 labels on the 40 sheet guesses with a second labeller's, and list the
disagreements with both readings. Usage, from the repository folder:
    python m3_validation/compare.py m3_validation/chatgpt_labels.txt"""
import json
import re
import sys

key = json.load(open("m3_validation/key.json"))
other = {}
for line in open(sys.argv[1]):
    m = re.match(r"\s*(\d+)[.:)]?\s+(RIGHT|WRONG)\b\s*(?:\|\s*(.*))?", line, re.IGNORECASE)
    if m:
        other[int(m.group(1))] = (m.group(2).upper(), m.group(3) or "")
assert len(other) == len(key), f"{len(other)} labels for {len(key)} guesses"

a = [k["judge"] == "WRONG" for k in key]
b = [other[k["item"]][0] == "WRONG" for k in key]
n = len(key)
agree = sum(x == y for x, y in zip(a, b))
p_o = agree / n
p_e = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
print(f"agreement {agree}/{n} ({100 * p_o:.0f}%), Cohen's kappa {(p_o - p_e) / (1 - p_e):.2f}")
print(f"only Haiku says WRONG {sum(x and not y for x, y in zip(a, b))}, only the other says WRONG {sum(y and not x for x, y in zip(a, b))}")
answers = {j["idx"]: j["answers"] for j in map(json.loads, open("results/judge/m3_haiku.jsonl"))}
for k, x, y in zip(key, a, b):
    if x != y:
        h = answers[k["idx"]][k["k"]]
        print(f"\n{k['item']:02d} (devtest {k['idx']}) NEW: {k['new']}")
        print(f"   Haiku {k['judge']}: {h['listener_understands']} / {h['german_says']}")
        print(f"   other {other[k['item']][0]}: {other[k['item']][1]}")
