import argparse
import json
from pathlib import Path

## One aligner line per model reply: full German ||| English so far plus the new reply

def step_lines(r: dict) -> list[str]:
    german = r["src"]
    fixed = ""
    lines = []
    for s in r["steps"]:
        english = fixed + s["text"]
        seen = " ".join(german.split()[:s["t"]])
        lines.append(f"{seen} ||| {english.strip()}")
        fixed = s["committed"]

    return lines

## Read the aligner's pairs: which words in NEW come from German that hasn't arrived?

def label_steps(r: dict, pair_lines: list[str]) -> list[dict]:
    fixed = ""
    labels = []
    for s, pair_line in zip(r["steps"], pair_lines):
        english = (fixed + s["text"]).split()
        n_fixed = len(fixed.split())
        if fixed and not s["text"][:1].isspace(): # a glued fragment makes the last word partly new
            n_fixed -= 1
        pairs = [tuple(map(int, p.split("-"))) for p in pair_line.split()]
        ahead, unaligned = [], []
        for j in range(n_fixed, len(english)):
            sources = [i for i, k in pairs if k == j]
            if not sources:
                unaligned.append(english[j])
            elif min(sources) >= s["t"]: # German words 0 to t-1 have arrived
                ahead.append(english[j])
        labels.append(dict(idx=r["idx"], t=s["t"], ahead=ahead, unaligned=unaligned))
        fixed = s["committed"]
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--pairs") # the aligner's output; without it, write the aligner's input
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.file)][:args.n]
    stem = Path(args.file).stem
    out_dir = Path("results/align")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairs:
        lines = [line for r in rows for line in step_lines(r)]
        out = out_dir / f"{stem}.txt"
        out.write_text("\n".join(lines) + "\n")
        print(len(lines), "lines saved to", out)
    else:
        pair_lines = open(args.pairs).read().split("\n")
        out = out_dir / f"{stem}_labels.jsonl"
        pos = 0
        n_steps = n_ahead = n_unaligned = 0
        with open(out, "w") as f:
            for r in rows:
                labels = label_steps(r, pair_lines[pos:pos + len(r["steps"])])
                pos += len(r["steps"])
                for l in labels:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
                    n_steps += 1
                    n_ahead += bool(l["ahead"])
                    n_unaligned += bool(l["unaligned"])
        print(f"{n_steps} steps, {n_ahead} with words ahead, {n_unaligned} with unaligned words")
        print("saved", out)
