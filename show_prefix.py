#!/usr/bin/env python3
"""Print a prefix forcing test file for reading by eye.

Usage: python show_prefix.py results/hymt15/deu_Latn_prefix.jsonl
"""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
for r in rows:
    flags = []
    if r.get("empty"):
        flags.append("EMPTY")
    if r.get("restart"):
        flags.append("RESTARTED")
    print(f"--- {r['i']} {' '.join(flags)}")
    print("  HALF SRC :", r["half_src"])
    print("  FORCED   :", r["forced"])
    print("  CONTINUED:", r["continuation"])
    print("  FULL HYP :", r["full_hyp"])
n = len(rows)
print(f"\n{n} cases, {sum(r.get('empty', False) for r in rows)} empty, "
      f"{sum(r.get('restart', False) for r in rows)} flagged as restarts. "
      "Count the sensible continuations yourself; the flags only catch the obvious failures.")
