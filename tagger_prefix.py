"""Tagger-on-prefixes test for the semantic chunking gate.

Every FLORES+ German devtest sentence is cut after each whitespace word and
tagged as a fragment. The tags a word gets when it has just arrived are
compared with the tags it gets in the full sentence. The full-sentence tags
are a silver reference (the tagger on complete input), not gold.

On top of the tags a crude gate state machine labels gate events per token:
SUB_OPEN (dass/weil, relative pronoun), FIN_MAIN (first finite verb of the
main clause), FIN_SUB (finite verb that closes an open subordinate clause),
FIN (further finite verb), PART (participle), PARTICLE (separable particle),
INF (infinitive). Event precision on prefixes is what matters for the gate:
a false FIN_SUB or PARTICLE opens the gate too early.

Usage:
  python tagger_prefix.py --backend spacy --model de_core_news_lg
  python tagger_prefix.py --backend spacy --model de_dep_news_trf
  python tagger_prefix.py --backend stanza
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

FIN = {"VVFIN", "VAFIN", "VMFIN"}
INF = {"VVINF", "VAINF", "VMINF", "VVIZU"}
PART = {"VVPP", "VAPP", "VMPP"}
SUB_OPEN = {"KOUS", "PRELS", "PRELAT"}
EVENTS = ["SUB_OPEN", "FIN_MAIN", "FIN_SUB", "FIN", "PART", "PARTICLE", "INF"]


def gate_events(tags):
    """Label each tag with a gate event; state carried left to right."""
    depth, main_fin, out = 0, False, []
    for t in tags:
        if t in SUB_OPEN:
            depth += 1
            ev = "SUB_OPEN"
        elif t in FIN:
            if depth > 0:
                depth -= 1
                ev = "FIN_SUB"
            else:
                ev = "FIN" if main_fin else "FIN_MAIN"
                main_fin = True
        elif t in PART:
            ev = "PART"
        elif t == "PTKVZ":
            ev = "PARTICLE"
        elif t in INF:
            ev = "INF"
        else:
            ev = "-"
        out.append(ev)
    return out


def is_punct(tag):
    return tag.startswith("$")


# ---------------------------------------------------------------- backends
# Each backend maps a list of texts to a list of token lists [(idx, text, tag)].

def spacy_backend(model):
    import spacy
    nlp = spacy.load(model)
    off = [p for p in ("ner", "parser", "lemmatizer") if p in nlp.pipe_names]
    nlp.select_pipes(disable=off)

    def run(texts):
        return [[(t.idx, t.text, t.tag_) for t in doc]
                for doc in nlp.pipe(texts, batch_size=64)]
    return run, f"spacy-{model}"


def stanza_backend():
    import stanza
    nlp = stanza.Pipeline("de", processors="tokenize,mwt,pos", verbose=False)

    def run(texts):
        docs = nlp([stanza.Document([], text=t) for t in texts])
        out = []
        for doc in docs:
            toks = []
            for s in doc.sentences:
                for tok in s.tokens:
                    tag = "+".join(w.xpos or "?" for w in tok.words)
                    toks.append((tok.start_char, tok.text, tag))
            out.append(toks)
        return out
    return run, "stanza-de"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["spacy", "stanza"], default="spacy")
    ap.add_argument("--model", default="de_core_news_lg")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--walk", type=int, default=5,
                    help="sentences to print prefix by prefix for the eye check")
    ap.add_argument("--out", default="tagger_results")
    ap.add_argument("--lag", type=int, default=0,
                    help="judge the word that arrived LAG words ago (confirmation delay)")
    ap.add_argument("--file", help="one German sentence per line, instead of FLORES+")
    a = ap.parse_args()

    if a.file:
        lines = [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
        sents = lines[a.start:a.start + a.n]
    else:
        from datasets import load_dataset
        ds = load_dataset("openlanguagedata/flores_plus", "deu_Latn", split="devtest")
        sents = [ds[i]["text"] for i in range(a.start, a.start + a.n)]
    a.n = len(sents)

    run, name = spacy_backend(a.model) if a.backend == "spacy" else stanza_backend()
    if a.lag:
        name += f"_lag{a.lag}"
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # Build all prefixes, remembering where the newest word starts.
    # With --lag L the word under test is word k-L, judged once word k is in.
    jobs = []  # (sentence id, k, prefix text, [start, end) of judged word)
    for si, s in enumerate(sents):
        ms = list(re.finditer(r"\S+", s))
        for k, m in enumerate(ms, start=1):
            j = k - a.lag
            if j < 1:
                continue
            lo = ms[j - 1].start()
            hi = ms[j].start() if j < len(ms) else 10**9
            jobs.append((si, k, s[:m.end()], (lo, hi)))

    t0 = time.time()
    full = run(sents)
    pref = run([j[2] for j in jobs])
    secs = time.time() - t0

    full_tag = [{idx: (txt, tag) for idx, txt, tag in toks} for toks in full]
    full_ev = []
    for toks in full:
        evs = gate_events([t[2] for t in toks])
        full_ev.append({t[0]: e for t, e in zip(toks, evs)})

    c = Counter()
    tag_conf, ev_conf = Counter(), Counter()
    first_noun = Counter()
    disagree, walk = [], []

    for (si, k, text, (wstart, wend)), toks in zip(jobs, pref):
        evs = gate_events([t[2] for t in toks])
        for (idx, txt, tag), ev in zip(toks, evs):
            ref = full_tag[si].get(idx)
            if ref is None or ref[0] != txt:
                c["misaligned"] += 1
                continue
            rtag, rev = ref[1], full_ev[si][idx]
            if is_punct(rtag):
                continue
            arrival = wstart <= idx < wend
            c["tok_all"] += 1
            c["agree_all"] += tag == rtag
            if not arrival:
                continue
            c["tok_arr"] += 1
            c["agree_arr"] += tag == rtag
            if tag != rtag:
                tag_conf[(rtag, tag)] += 1
            if ev != "-" or rev != "-":
                ev_conf[(rev, ev)] += 1
                if ev != rev:
                    disagree.append(
                        f"[{si + a.start}] word {k}: {rev} -> {ev}   "
                        f"(full {rtag}, prefix {tag})\n"
                        f"  prefix: {text[:idx]}[{txt}]\n"
                        f"  full:   {sents[si]}\n")
            if k - a.lag == 1:
                first_noun[(rtag == "NN", tag == "NN")] += 1
        if si < a.walk:
            new = [(t[1], t[2], e) for t, e in zip(toks, evs) if wstart <= t[0] < wend]
            mark = "  ".join(f"{w}/{g}" + (f" <{e}>" if e != "-" else "")
                             for w, g, e in new)
            walk.append(f"[{si + a.start}] {k:2d} {mark:40s} | {text}")
        elif si == a.walk and k == 1 + a.lag:
            walk.append("")

    per_event = {}
    for e in EVENTS:
        tp = ev_conf[(e, e)]
        ref_n = sum(v for (r, p), v in ev_conf.items() if r == e)
        pre_n = sum(v for (r, p), v in ev_conf.items() if p == e)
        per_event[e] = {
            "ref": ref_n, "pred": pre_n,
            "precision": round(tp / pre_n, 3) if pre_n else None,
            "recall": round(tp / ref_n, 3) if ref_n else None,
        }

    summary = {
        "backend": name, "lag": a.lag, "sentences": a.n, "prefixes": len(jobs),
        "sec_total": round(secs, 1),
        "ms_per_prefix": round(1000 * secs / len(jobs), 1),
        "tag_agree_arrival": round(c["agree_arr"] / c["tok_arr"], 3),
        "tag_agree_all_tokens": round(c["agree_all"] / c["tok_all"], 3),
        "misaligned_tokens": c["misaligned"],
        "events": per_event,
        "event_disagreements": len(disagree),
        "first_word_NN": {"full NN, prefix NN": first_noun[(True, True)],
                          "full not NN, prefix NN": first_noun[(False, True)],
                          "full NN, prefix not NN": first_noun[(True, False)]},
        "top_tag_changes_full_to_prefix": [
            f"{r} -> {p}: {v}" for (r, p), v in tag_conf.most_common(15)],
    }
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{name}_disagree.txt").write_text("\n".join(disagree))
    (out / f"{name}_walk.txt").write_text("\n".join(walk))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/{name}_summary.json, _disagree.txt, _walk.txt")


if __name__ == "__main__":
    main()
