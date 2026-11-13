"""Compare the judge's (Haiku) labels on the 50 sheet steps with a second labeller's.
Usage, from the repository folder: python rq2_validation/compare.py rq2_validation/chatgpt_labels.txt"""
import json
import re
import sys

key = json.load(open("rq2_validation/key.json"))
other = {}
for line in open(sys.argv[1]):
    m = re.match(r'\s*(\d+)\s+(NONE|GUESS)(?:\s+"([^"]*)")?\s*(?:\|\s*(.*))?', line)
    if m:
        other[int(m.group(1))] = (m.group(3) if m.group(2) == "GUESS" else None, m.group(4) or "")
assert len(other) == len(key), f"{len(other)} labels for {len(key)} steps"

a = [k["judge"] is not None for k in key]          # judge says guess
b = [other[k["item"]][0] is not None for k in key]  # other labeller says guess
n = len(key)
both, neither = sum(x and y for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))
p_o = (both + neither) / n
p_e = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
print(f"agreement {both + neither}/{n} ({100 * p_o:.0f}%), Cohen's kappa {(p_o - p_e) / (1 - p_e):.2f}")
print(f"both guess {both}, both none {neither}, only judge {sum(x and not y for x, y in zip(a, b))}, "
      f"only other {sum(y and not x for x, y in zip(a, b))}")
for arm in ("hymt2-q4", "gemma-e2b-q4"):
    ix = [i for i, k in enumerate(key) if k["arm"] == arm]
    print(f"  {arm}: agreement {sum(a[i] == b[i] for i in ix)}/{len(ix)}")
