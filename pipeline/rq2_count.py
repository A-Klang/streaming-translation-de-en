"""RQ2 counts from the judge's labels, one line per run.

raw        share of model calls whose reply contains a guess
committed  steps whose guess reached the committed output, per 100 German words
sentences  share of sentences with at least one committed guess
passed     share of guessing replies whose guess was committed (committed / raw guesses)
wrong      (M3, once m3_judge.py has run) committed guesses that turned out wrong, per 100 German words,
           and as a share of committed guesses

The last step of each sentence is left out: the whole German has arrived by then.
A guess counts as committed if the quoted words start within the step's committed words
(the first n_new words of the reply). Usage, from the repository folder:
    python pipeline/rq2_count.py results/judge/*_haiku.jsonl
"""
import json
import re
import sys
from pathlib import Path

PUNCT = '„“”"‘’\'«».,!?;:…()'


def norm(words: list[str]) -> list[str]:
    return [w.strip(PUNCT).lower() for w in words]


def guess_start(reply: str, guess: str) -> int | None:
    # word position in the reply where the quoted guess begins, None if it can't be found
    r, g = norm(reply.split()), [w for w in norm(guess.split()) if w]
    if not g:
        return None
    for i in range(len(r) - len(g) + 1):
        if [w for w in r[i:i + len(g)]] == g:
            return i
    # the judge sometimes quotes part of a word or a slightly different span: fall back to the first word
    for i, w in enumerate(r):
        if w and (w == g[0] or w.startswith(g[0]) or g[0].startswith(w)):
            return i
    return None


def run_file(judge_file: Path) -> Path:
    # results/judge/<arm>_<setting>_<time>_haiku.jsonl -> results/pipeline/<arm>_<setting>/<arm>_<setting>_<time>.jsonl
    stem = re.sub(r"_[a-z]+$", "", judge_file.stem)
    setting = re.sub(r"_\d{8}-\d{6}$", "", stem)
    return Path("results/pipeline") / setting / f"{stem}.jsonl"


M3_ITEMS, M3_LABELS = Path("results/judge/m3_items.json"), Path("results/judge/m3_haiku.jsonl")


def m3_labels() -> dict[tuple, str]:
    # (run, idx, t) -> RIGHT or WRONG for each committed guess (m3_judge.py), empty before M3 is run
    if not M3_LABELS.exists():
        return {}
    items = json.loads(M3_ITEMS.read_text())
    out = {}
    for j in map(json.loads, open(M3_LABELS)):
        for item, label in zip(items[str(j["idx"])], j["labels"]):
            for occ in item["runs"]:
                out[occ["run"], j["idx"], occ["t"]] = label
    return out


def per_sentence(judge_file: Path, m3: dict | None = None) -> dict[int, dict]:
    # counts for each judged sentence: German words, steps, guessing replies, committed guesses,
    # and (with M3 labels) committed guesses that turned out wrong
    runs = {r["idx"]: r for r in map(json.loads, open(run_file(judge_file)))}
    run = run_file(judge_file).parent.name
    m3 = m3_labels() if m3 is None else m3
    out = {}
    for j in map(json.loads, open(judge_file)):
        r = runs[j["idx"]]
        c = dict(words=r["n_src"], steps=0, raw=0, committed=0, unfound=0, wrong=0, unlabelled=0)
        for s in r["steps"]:
            if s["t"] == r["n_src"]:
                continue  # the whole German has arrived: nothing left to guess, only to mistranslate
            c["steps"] += 1
            g = j["labels"].get(str(s["t"]))
            if g is None:
                continue
            c["raw"] += 1
            start = guess_start(s["text"], g)
            if start is None:
                c["unfound"] += 1
            elif start < s["n_new"]:
                c["committed"] += 1
                if m3:
                    label = m3.get((run, j["idx"], s["t"]))
                    c["wrong"] += label == "WRONG"
                    c["unlabelled"] += label is None
        out[j["idx"]] = c
    return out


def count(judge_file: Path, m3: dict | None = None) -> dict:
    sent = per_sentence(judge_file, m3).values()
    total = {k: sum(c[k] for c in sent) for k in ("words", "steps", "raw", "committed", "unfound", "wrong", "unlabelled")}
    return dict(run=run_file(judge_file).parent.name, sentences=len(sent), steps=total["steps"],
                raw=total["raw"] / total["steps"], committed=100 * total["committed"] / total["words"],
                sentences_hit=sum(c["committed"] > 0 for c in sent) / len(sent),
                passed=total["committed"] / total["raw"] if total["raw"] else 0.0, unfound=total["unfound"],
                wrong=100 * total["wrong"] / total["words"],
                wrong_share=total["wrong"] / total["committed"] if total["committed"] else 0.0,
                unlabelled=total["unlabelled"])


if __name__ == "__main__":
    m3 = m3_labels()
    head = f"{'run':26} {'sent':>4} {'steps':>5} {'raw':>6} {'comm/100w':>9} {'sent≥1':>7} {'passed':>7} {'unfound':>7}"
    print(head + (f" {'wrong/100w':>10} {'wrong':>6}" if m3 else ""))
    for f in sorted(sys.argv[1:]):
        c = count(Path(f), m3)
        print(f"{c['run']:26} {c['sentences']:4} {c['steps']:5} {100 * c['raw']:5.1f}% {c['committed']:9.2f} "
              f"{100 * c['sentences_hit']:6.0f}% {100 * c['passed']:6.1f}% {c['unfound']:7}"
              + (f" {c['wrong']:10.2f} {100 * c['wrong_share']:5.0f}%" + (f" ({c['unlabelled']} unlabelled)" if c["unlabelled"] else "")
                 if m3 else ""))
