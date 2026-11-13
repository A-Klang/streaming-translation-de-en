"""Flag German words that leak into the committed English.

A word is flagged if it appears in the German source but not in the English
reference (names and numbers appear in both, so they drop out). Each flag in a
streaming run is compared with the same arm's full-sentence run (waitk999):
  streaming only -> caused by streaming (how we call the model)
  both           -> model weakness, also there with the whole sentence

Run from leak_check/ after the simulate.py runs:
    python leak_check.py
Writes leaks.txt (every flag, for checking by hand) and prints a summary.
"""
import json
from pathlib import Path

RUNS = Path("results/pipeline")
ARMS = ["hymt2-q4", "gemma-e2b-q4"]
POLICIES = ["waitk3", "la-s1-verb"]
FULL = "waitk999"
PUNCT = '„“”"‘’\'«».,!?;:…()–-'


def words(text: str) -> list[str]:
    return [w.strip(PUNCT) for w in text.split() if w.strip(PUNCT)]


def leaks(row: dict) -> list[str]:
    # committed words that are in the German source but not in the English reference
    src = {w.lower() for w in words(row["src"])}
    ref = {w.lower() for w in words(row["ref"])}
    return [w for w in words(row["hyp"])
            if w.lower() in src and w.lower() not in ref and not w.isdigit()]


def load(arm: str, name: str) -> dict[int, dict]:
    # newest run in the setting's folder (timestamped names sort by time)
    runs = sorted((RUNS / f"{arm}_{name}").glob("*.jsonl"))
    if not runs:
        return {}
    return {r["idx"]: r for r in map(json.loads, open(runs[-1]))}


if __name__ == "__main__":
    lines = []
    for arm in ARMS:
        full = load(arm, FULL)
        if not full:
            print(f"{arm}: no {FULL} run yet, skipped")
            continue
        full_leaks = {i: leaks(r) for i, r in full.items()}
        n_full = sum(1 for v in full_leaks.values() if v)
        print(f"\n{arm}  ({len(full)} sentences)")
        print(f"  {FULL:<11} sentences with a flag: {n_full}")
        lines.append(f"===== {arm} {FULL} (full sentence) =====")
        for i, ws in full_leaks.items():
            if ws:
                lines += [f"[{i}] {ws}", f"   DE  {full[i]['src']}", f"   EN  {full[i]['hyp']}", ""]

        for pol in POLICIES:
            rows = load(arm, pol)
            if not rows:
                print(f"  {pol:<11} no run yet")
                continue
            stream_only = both = 0
            lines.append(f"===== {arm} {pol} =====")
            for i, r in rows.items():
                ws = leaks(r)
                if not ws:
                    continue
                full_set = {w.lower() for w in full_leaks.get(i, [])}
                new = [w for w in ws if w.lower() not in full_set]
                if new:
                    stream_only += 1
                else:
                    both += 1
                tag = "STREAMING" if new else "ALSO FULL"
                lines += [f"[{i}] {tag} {ws}",
                          f"   DE    {r['src']}",
                          f"   EN    {r['hyp']}",
                          f"   FULL  {full[i]['hyp'] if i in full else '-'}",
                          f"   REF   {r['ref']}", ""]
            print(f"  {pol:<11} sentences flagged: streaming only {stream_only}, also in full {both}")

    Path("leaks.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nAll flags in leaks.txt")
