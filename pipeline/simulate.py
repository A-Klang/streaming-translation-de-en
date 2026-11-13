import argparse
import json
import sys
from datetime import datetime
from backend import MODELS, loaded_model, translate
from pathlib import Path
from data import load, Sentence
from policies import WaitK, LocalAgreement, GrammarChunk, join_words

WORDS_PER_SEC = 2.5

def run_sentence(policy, s: Sentence, live: bool = False, replies: bool = False) -> dict:
    policy.reset()
    committed = []
    delays = [] # German words read when each English word was committed
    ca_delays = [] # seconds on the virtual clock, compute included
    clock = 0.0
    calls = prompt_tokens = output_tokens = cached_tokens = 0
    compute_ms = 0.0
    steps = [] # one record per model call
    n = len(s.src_words)
    if live:
        print(f"\n[{s.idx}] {n} German words")
        # full-sentence translation for the audience to compare against, not counted in any stats
        full = translate(policy.arm, s.src_words, "", 256)
        print(f"full sentence: {' '.join(full.text.split())}\n", flush=True)

    for t in range(1, n+1):
        clock = max(clock, t / WORDS_PER_SEC) #wait for word t to arrive
        forced = join_words(committed) # the English the model is asked to continue at this step
        new, res = policy.step(s.src_words[:t], committed, t == n)
        if res is not None:
            calls += 1
            prompt_tokens += res.prompt_tokens
            output_tokens += res.output_tokens
            cached_tokens += res.cached_tokens
            ms = res.prompt_ms + res.decode_ms
            compute_ms += ms
            clock += ms / 1000 #model busy
        # a reply that starts without a space continues the last committed word ("4" + "-month-old")
        if new and committed and not res.text[:1].isspace():
            committed[-1] += new.pop(0)
        committed += new
        delays += [t] * len(new)
        ca_delays += [clock] * len(new)
        if res is not None:
            steps.append(dict(t=t, prompt_tokens=res.prompt_tokens, cached_tokens=res.cached_tokens,
                  output_tokens=res.output_tokens, prompt_ms=res.prompt_ms,
                  decode_ms=res.decode_ms, n_new=len(new), text=res.text, stop=res.stop_type,
                  committed=join_words(committed)))

        if live:
            old = join_words(committed[:len(committed) - len(new)])
            print(f"{t:3} {s.src_words[t - 1]:<15} | {old} \033[1;32m{join_words(new)}\033[0m", flush=True)
            if replies and res is not None: # forced English dimmed, the model's continuation in dim green
                space = " " if forced and res.text[:1].isspace() else ""
                print(f"{'':22}\033[2m{forced}{space}\033[32m{' '.join(res.text.split())}\033[0m", flush=True)

    return dict(idx=s.idx, src=" ".join(s.src_words), ref=s.ref, hyp=join_words(committed),
                n_src=n, delays=delays, ca_delays=ca_delays, calls=calls,
                prompt_tokens=prompt_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
                compute_ms=compute_ms, steps=steps)

        
        
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",choices=["waitk", "la", "grammar"], required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--replies", action="store_true") # with --live, also print the model's reply at each call
    ap.add_argument("--arm", choices=list(MODELS), required=True)
    ap.add_argument("--verb-rule", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--delay", type=int, default=0, help="grammar: words of lookahead before a word is judged")
    ap.add_argument("--resume", help="results file of an interrupted run with the same settings, to continue it")
    args = ap.parse_args()
    loaded = loaded_model()
    if loaded != MODELS[args.arm]:
        sys.exit(f"The server is running {loaded}, but --arm {args.arm} needs {MODELS[args.arm]}")

    if args.policy == "waitk":
        policy, name = WaitK(args.arm, args.k), f"waitk{args.k}"
    elif args.policy == "grammar":
        policy = GrammarChunk(args.arm, delay=args.delay)
        name = f"grammar-d{args.delay}"
    else:
        policy = LocalAgreement(args.arm, args.verb_rule, args.stride)
        name = f"la-s{args.stride}" + ("-verb" if args.verb_rule else "")
        


    # one folder per setting, a new timestamped file per run, never overwritten
    run_dir = Path("results/pipeline") / f"{args.arm}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume:
        # continue an interrupted run: keep its finished sentences, append the rest to the same file
        out = Path(args.resume)
        if out.parent.resolve() != run_dir.resolve():
            sys.exit(f"{out} is not a {args.arm}_{name} run (expected a file in {run_dir})")
        lines = out.read_text().splitlines()
        good = []
        for line in lines:
            try:
                done.add(json.loads(line)["idx"])
                good.append(line)
            except json.JSONDecodeError:
                break # a line cut off by a hard stop; drop it and redo that sentence
        if len(good) < len(lines):
            out.write_text("".join(l + "\n" for l in good))
        print(f"resuming {out}: {len(done)} sentences already done", flush=True)
        mode = "a"
    else:
        out = run_dir / f"{args.arm}_{name}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        mode = "x"
    print("writing", out, flush=True)
    with open(out, mode) as f:
        for s in load(args.n, args.start):
            if s.idx in done:
                continue
            r = run_sentence(policy, s, args.live, args.replies)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush() # a hard stop loses at most the sentence in progress
            print(r["idx"], r["hyp"], flush=True)
    print("saved", out)