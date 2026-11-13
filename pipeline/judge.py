import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

JUDGE_DIR = Path.home() / "judge_test"
PROMPT_FILE = str(Path(__file__).resolve().parent / "rq2_prompt.txt") # in the repo; the judge still runs from JUDGE_DIR, which keeps it isolated
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
NO_TOOLS = "Bash Read Write Edit Glob Grep WebFetch WebSearch Agent Skill"
# the claude binary of the newest installed VS Code extension (the version in the path changes with every update)
CLAUDE = str(sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"),
                    key=lambda p: [int(x) for x in re.findall(r"\d+", p.parts[-4])])[-1])


def build_prompt(r: dict) -> str:
    src_words = r["src"].split()
    fixed = ""
    blocks = []
    for s in r["steps"]:
        german = " ".join(src_words[:s["t"]])
        blocks.append(f"STEP {s['t']}\nGERMAN: {german}\nFIXED: {fixed}\nNEW: {s['text'].strip()}")
        fixed = s["committed"]
    return "\n\n".join(blocks)

## Call the judge through claude -p, isolated from CLAUDE.md, memory and tools
def call_judge(prompt: str, model: str, prompt_file: str = PROMPT_FILE, extra: tuple = ()) -> dict:
    # extra: more CLI arguments (M3 passes --json-schema and --effort)
    env = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    cmd = [CLAUDE, "-p", "--safe-mode", "--model", MODELS[model],
           "--system-prompt-file", prompt_file,
           "--disallowedTools", NO_TOOLS,
           "--no-session-persistence",
           "--output-format", "json", *extra]
    done = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=JUDGE_DIR, env=env)
    if done.returncode != 0:
        raise RuntimeError(done.stderr or done.stdout)
    return json.loads(done.stdout)

## Read the judge's answer: one line per step
def parse_labels(text: str) -> dict[int, str | None]:
    # step number -> the guessed words, or None if the step has no guess
    labels = {}
    for line in text.splitlines():
        # accepts "12 NONE", "STEP 12 NONE", "<step12> NONE", "<step> 12 NONE" and "12: NONE"
        # (the judge sometimes copies the answer template)
        m = re.fullmatch(r'(?:<[^>\d]*>\s*)?<?(?:STEP\s*)?(\d+)>?[:.)]?\s+(NONE|GUESS)(?:\s+"(.*)")?',
                         line.strip(), re.IGNORECASE)
        if m:
            labels[int(m.group(1))] = m.group(3) if m.group(2).upper() == "GUESS" else None
    return labels


def judge_sentence(r: dict, model: str, tries: int = 3) -> tuple[dict, dict]:
    # ask again if the answer leaves a step without a label (it happens now and then)
    for _ in range(tries):
        reply = call_judge(build_prompt(r), model)
        labels = parse_labels(reply["result"])
        if all(s["t"] in labels for s in r["steps"]):
            break
    return reply, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", choices=list(MODELS), default="haiku")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ids", help="JSON list of sentence indices to judge (overrides --n), e.g. rq2_ids.json")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.file)]
    if args.ids:
        ids = set(json.load(open(args.ids)))
        rows = [r for r in rows if r["idx"] in ids]
    else:
        rows = rows[:args.n]

    out = Path("results/judge") / f"{Path(args.file).stem}_{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(line)["idx"] for line in open(out)] if out.exists() else []
    todo = [r for r in rows if r["idx"] not in done]

    with open(out, "a") as f, ThreadPoolExecutor(args.jobs) as pool:
        # the calls run in parallel; only this loop writes to the file
        calls = {pool.submit(judge_sentence, r, args.model): r for r in todo}
        for call in as_completed(calls):
            r = calls[call]
            reply, labels = call.result()
            usage = reply["usage"]
            f.write(json.dumps(dict(idx=r["idx"], model=args.model, labels=labels, raw=reply["result"],
                                    n_steps=len(r["steps"]), input_tokens=usage["input_tokens"],
                                    cache_write_tokens=usage["cache_creation_input_tokens"],
                                    cache_read_tokens=usage["cache_read_input_tokens"],
                                    output_tokens=usage["output_tokens"],
                                    thinking_tokens=usage["output_tokens_details"]["thinking_tokens"],
                                    duration_ms=reply["duration_ms"]),
                               ensure_ascii=False) + "\n")
            f.flush()
            guesses = sum(g is not None for g in labels.values())
            missing = sum(s["t"] not in labels for s in r["steps"])
            print(r["idx"], f"{len(r['steps']) - missing}/{len(r['steps'])} steps labelled, {guesses} guesses"
                  + (" (INCOMPLETE after retries)" if missing else ""))
    print("saved", out)



