#!/usr/bin/env python3
"""Slip-through test for the "two steps must agree" commit rule.

Runs walk.py's local agreement streaming (same agreement rule, same prefix
forcing) over many FLORES+ sentences and logs every step, so each committed
piece of English can later be checked against the German that had arrived
when it was committed. A slip-through is a commit that states content the
German at that step does not contain yet (an invented ending that both steps
agreed on).

One JSON line per sentence in results/agree_test/<arm>.jsonl. Reruns skip
sentences already done, so the run can be stopped and resumed.

  caffeinate -i python agree_test.py --arms hymt2-q4 gemma-e2b-q4 --start 100 --n 200
  caffeinate -i python agree_test.py --probs 5    # same run, token probabilities logged
  caffeinate -i python agree_test.py --verb-rule  # with the verb rule (verb_gate.py)
  caffeinate -i python agree_test.py --verb-rule --ids judged_ids.json   # only the 100 judged sentences
"""
import argparse
import json
import time
from pathlib import Path

import pilot as P
import walk as W


def norm_probs(cp):
    """llama-server token probabilities, old or new format, as
    [token, logprob, [[alt token, alt logprob], ...]] per generated token."""
    import math
    out = []
    for e in cp or []:
        if "top_logprobs" in e:          # newer servers: raw-logit log probabilities
            out.append([e.get("token", ""), e.get("logprob"),
                        [[t.get("token", ""), t.get("logprob")] for t in e["top_logprobs"]]])
        else:                            # older servers: probs list with tok_str/prob
            alts = [[t.get("tok_str", ""), math.log(max(t.get("prob", 0), 1e-12))]
                    for t in e.get("probs", [])]
            out.append([e.get("content", ""), alts[0][1] if alts else None, alts])
    return out


def generate_with_probs(tr, prompt, n_probs):
    """Same request as pilot.ServerTranslator._generate, plus token probabilities."""
    r = tr._post("/completion", {"prompt": prompt, "n_predict": tr.max_new_tokens,
                                 "temperature": 0.0, "top_k": 1, "cache_prompt": True,
                                 "n_probs": n_probs, "post_sampling_probs": False})
    return r["content"].strip(), norm_probs(r.get("completion_probabilities"))


def stream_sentence(tr, d, src, joiner=" ", n_probs=0, verb_rule=False):
    """Local agreement over one sentence. Returns the per-step log."""
    words = src.split()
    shown, prev_full, steps = "", None, []
    for k in range(1, len(words) + 1):
        prefix = joiner.join(words[:k])
        t0 = time.time()
        toks = None
        if n_probs:
            prompt = tr._prompt_text(prefix, d) + (shown if shown else "")
            text, toks = generate_with_probs(tr, prompt, n_probs)
            full = W.join_cont(shown, text).strip() if shown else text
        elif shown:
            cont = tr.continue_prefix(prefix, d, shown)
            full = W.join_cont(shown, cont).strip()
        else:
            full = tr.translate(prefix, d)[0]
        sec = time.time() - t0
        last = k == len(words)
        if last:
            new_shown = full
        elif prev_full is not None:
            agreed = W.agree_clean(prev_full, full,
                                   quote_closed=any(c in prefix for c in "“”»\""))
            new_shown = agreed if len(agreed.split()) > len(shown.split()) else shown
        else:
            new_shown = shown
        held = ""
        if verb_rule and not last:
            import verb_gate as VG
            new_shown, held = VG.allowed(prefix, shown, new_shown)
        added = new_shown[len(shown):].strip()
        step = dict(k=k, word=words[k - 1], full=full, added=added,
                    final_step=last, sec=round(sec, 3))
        if verb_rule:
            step["held"] = held             # agreed English the verb rule kept back
        if toks is not None:
            step["forced"] = shown          # English forced at this step
            step["toks"] = toks             # generated tokens with log probabilities
        steps.append(step)
        shown, prev_full = new_shown, full
    return steps, shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["hymt2-q4", "gemma-e2b-q4"])
    ap.add_argument("--start", type=int, default=100,
                    help="first FLORES+ devtest index (100 on keeps clear of the pilot sentences)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ids", help="JSON file with a list of FLORES+ indices to run instead of --start/--n")
    ap.add_argument("--verb-rule", action="store_true",
                    help="never commit English verbs or numbers ahead of the German ones "
                         "(verb_gate.py); results go to results/agree_test_verb/")
    ap.add_argument("--probs", type=int, default=0, metavar="N",
                    help="also log each generated token's log probability and the top N "
                         "alternatives (server arms only); results go to results/agree_test_probs/")
    a = ap.parse_args()

    import torch
    device = P.pick_device(torch)
    d = ("deu_Latn", "eng_Latn")
    ids = json.load(open(a.ids)) if a.ids else list(range(a.start, a.start + a.n))
    pairs = P.load_flores(*d, max(ids) + 1)
    out_dir = Path("results") / ("agree_test_verb" if a.verb_rule else
                                 "agree_test_probs" if a.probs else "agree_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in a.arms:
        out = out_dir / f"{arm}.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(l)["i"] for l in out.open(encoding="utf-8") if l.strip()}
        todo = [i for i in ids if i not in done]
        if not todo:
            print(f"{arm}: all {len(ids)} sentences already done")
            continue
        print(f"{arm}: {len(todo)} sentences to run ({len(done)} already done)")
        tr = P.make_chat(P.ARMS[arm], torch, device, a.max_new_tokens, out_dir / arm)
        if a.probs:
            props = tr._get("/props")
            meta = {k: props.get(k) for k in ("bos_token", "eos_token", "build_info") if k in props}
            (out_dir / f"{arm}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"  server: {meta}")
        t_arm = time.time()
        try:
            with out.open("a", encoding="utf-8") as f:
                for n_done, i in enumerate(todo, 1):
                    src, ref = pairs[i]
                    t0 = time.time()
                    steps, final = stream_sentence(tr, d, src, n_probs=a.probs, verb_rule=a.verb_rule)
                    if a.probs and n_done == 1:
                        t = steps[-1].get("toks") or []
                        if not t or t[0][1] is None:
                            raise SystemExit("server returned no token probabilities; stopping")
                        print("  probability check, first tokens of the last step:",
                              [(x[0], round(x[1], 3)) for x in t[:4]])
                    f.write(json.dumps(dict(i=i, arm=arm, src=src, ref=ref, final=final,
                                            steps=steps), ensure_ascii=False) + "\n")
                    f.flush()
                    el = time.time() - t_arm
                    eta = el / n_done * (len(todo) - n_done)
                    print(f"  [{i}] {len(steps)} words, {time.time() - t0:.1f} s "
                          f"({n_done}/{len(todo)}, about {eta / 60:.0f} min left)")
        finally:
            if hasattr(tr, "close"):
                tr.close()
    print(f"\ndone, logs in {out_dir}/")


if __name__ == "__main__":
    main()
