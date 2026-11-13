"""M3: were the committed guesses right? Every guess the RQ2 judge found in a committed
step (results/judge/*_haiku.jsonl, all report runs) is judged once against the complete
German and the FLORES reference: RIGHT or WRONG for a listener (prompt: m3_prompt.txt).
The same guess in the same context is judged once and shares its label across runs.

Writes results/judge/m3_items.json (every distinct guess, with the runs it occurs in) and
results/judge/m3_<model>.jsonl (labels per sentence). Usage, from the repository folder:
    python pipeline/m3_judge.py --n 3        # the first 3 sentences, to look at
    python pipeline/m3_judge.py              # all 100
"""
import argparse
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from judge import call_judge
from rq2_count import guess_start, run_file, norm

PROMPT_FILE = str(Path(__file__).resolve().parent / "m3_prompt.txt")
# the answer format: for each guess, what a listener would understand and what the German says,
# then the label. Writing both down first is the comparison the label rests on.
SCHEMA = json.dumps({"type": "object", "required": ["guesses"], "properties": {"guesses": {
    "type": "array", "items": {"type": "object", "required": ["n", "listener_understands", "german_says", "label"],
                               "properties": {"n": {"type": "integer"},
                                              "listener_understands": {"type": "string"},
                                              "german_says": {"type": "string"},
                                              "label": {"enum": ["RIGHT", "WRONG"]}}}}}})
EXTRA = ("--json-schema", SCHEMA, "--effort", "low")


def collect() -> dict[int, list[dict]]:
    # sentence idx -> its distinct committed guesses
    items = {}
    for jf in sorted(glob.glob("results/judge/*_2026*_haiku.jsonl")):
        run = run_file(Path(jf)).parent.name
        runs = {r["idx"]: r for r in map(json.loads, open(run_file(Path(jf))))}
        for j in map(json.loads, open(jf)):
            r = runs[j["idx"]]
            src, before = r["src"].split(), ""
            for s in r["steps"]:
                g = j["labels"].get(str(s["t"]))
                start = guess_start(s["text"], g) if g is not None else None
                if s["t"] < r["n_src"] and start is not None and start < s["n_new"]:
                    words = s["text"].split()
                    end = min(start + max(1, len([w for w in norm(g.split()) if w])), s["n_new"])
                    new = " ".join(words[:start] + ["[" + " ".join(words[start:end]) + "]"] + words[end:s["n_new"]])
                    key = (s["t"], before, new)
                    sent = items.setdefault(r["idx"], {})
                    item = sent.setdefault(key, dict(seen=" ".join(src[:s["t"]]), before=before, new=new, runs=[]))
                    item["runs"].append(dict(run=run, t=s["t"]))
                before = s["committed"]
    return {idx: list(v.values()) for idx, v in items.items()}


def build_prompt(src: str, ref: str, items: list[dict]) -> str:
    blocks = [f"GERMAN: {src}\nREFERENCE: {ref}"]
    for i, x in enumerate(items, 1):
        blocks.append(f"GUESS {i}\nSEEN: {x['seen']}\nBEFORE: {x['before']}\nNEW: {x['new']}")
    return "\n\n".join(blocks)


def parse(reply: dict) -> dict[int, dict]:
    # guess number -> the judge's filled-in fields
    return {g["n"]: g for g in (reply.get("structured_output") or {}).get("guesses", [])}


CHUNK = 15  # guesses per call; a sentence can have up to 50


def judge(src: str, ref: str, items: list[dict], model: str, tries: int = 3):
    # labels numbered 1.. over the whole sentence, the chunks' answers joined
    labels, tokens = {}, 0
    for c in range(0, len(items), CHUNK):
        chunk = items[c:c + CHUNK]
        for _ in range(tries):
            reply = call_judge(build_prompt(src, ref, chunk), model, PROMPT_FILE, EXTRA)
            got = parse(reply)
            if all(i in got for i in range(1, len(chunk) + 1)):
                break
        labels.update({c + i: v for i, v in got.items() if i <= len(chunk)})
        tokens += reply["usage"]["output_tokens"]
    return dict(usage=dict(output_tokens=tokens)), labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, help="only the first n sentences")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    items = collect()
    Path("results/judge/m3_items.json").write_text(json.dumps(
        {str(k): v for k, v in sorted(items.items())}, ensure_ascii=False, indent=1))
    # any run file has the German and the reference
    first = glob.glob("results/pipeline/hymt2-q4_waitk999/*.jsonl")[0]
    rows = {r["idx"]: r for r in map(json.loads, open(first))}
    ids = sorted(items)[:args.n] if args.n else sorted(items)

    out = Path(f"results/judge/m3_{args.model}.jsonl")
    done = {json.loads(line)["idx"] for line in open(out)} if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print(sum(len(items[i]) for i in ids), "distinct guesses in", len(ids), "sentences,", len(todo), "to judge")

    with open(out, "a") as f, ThreadPoolExecutor(args.jobs) as pool:
        calls = {pool.submit(judge, rows[i]["src"], rows[i]["ref"], items[i], args.model): i for i in todo}
        for call in as_completed(calls):
            i = calls[call]
            reply, labels = call.result()
            n = len(items[i])
            f.write(json.dumps(dict(idx=i, model=args.model,
                                    labels=[labels[k]["label"] if k in labels else None for k in range(1, n + 1)],
                                    answers=[labels.get(k) for k in range(1, n + 1)],
                                    output_tokens=reply["usage"]["output_tokens"]),
                               ensure_ascii=False) + "\n")
            f.flush()
            missing = sum(labels.get(k) is None for k in range(1, len(items[i]) + 1))
            print(i, f"{len(items[i]) - missing}/{len(items[i])} labelled, {sum(v['label'] == 'WRONG' for v in labels.values())} wrong"
                  + (" (INCOMPLETE after retries)" if missing else ""))
    print("saved", out)
