#!/usr/bin/env python3
"""Word-by-word streaming walk through one or more FLORES sentences.

At each step the model gets the source words seen so far, plus the English
already shown as the forced start of its answer, and continues. What gets
shown is decided by local agreement: a word is shown once two consecutive
steps agree on it. At the end of the sentence the final output is shown.

  python walk.py --i 0
  python walk.py --i 0 9 --arm hymt2-q4
"""
import argparse
import pilot as P


CLOSING = '.!?…”"’»\''
EDGE = '„“”"‘’\'«».,!?;:…()'


def norm(w):
    """A word without surrounding punctuation, for comparison."""
    return w.strip(EDGE).lower()


QUOTE_CLOSE = '”"»'
MIN_AFTER = 2   # agreed words needed after a sentence end before it may be shown


def agree_clean(a, b, quote_closed=False):
    """Words two hypotheses agree on, ignoring punctuation.

    A word that ends a sentence (. ! ? … or a closing quote) is only shown
    whole if both hypotheses also agree on at least MIN_AFTER words after it,
    i.e. the model has really split the sentence. Otherwise it is shown
    without its ending and nothing after it is shown yet. Closing quotation
    marks are never shown before the source has closed its quote.
    """
    xs, ys = a.split(), b.split()
    n = 0
    while n < min(len(xs), len(ys)):
        x, y = xs[n], ys[n]
        if norm(x) != norm(y) or (not norm(y) and x != y):
            break
        n += 1
    agreed = ys[:n]
    for j, w in enumerate(agreed):
        if w.rstrip(CLOSING) != w and (n - 1 - j) < MIN_AFTER:
            agreed = agreed[:j] + ([w.rstrip(CLOSING)] if w.rstrip(CLOSING) else [])
            break
    if not quote_closed:
        agreed = [w.rstrip(QUOTE_CLOSE) if w.rstrip(QUOTE_CLOSE) else w for w in agreed]
    return " ".join(agreed)


def join_cont(shown, cont):
    if not shown:
        return cont
    return shown + ("" if cont[:1] in ".,!?;:…”’\"-)" else " ") + cont


def lcp_words(a, b):
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return " ".join(a.split()[:n])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--i", type=int, nargs="+", default=[0])
    p.add_argument("--arm", default="hymt2-q4", choices=[a for a, v in P.ARMS.items() if v["kind"] in ("chat", "server")])
    p.add_argument("--src", default="deu_Latn")
    p.add_argument("--tgt", default="eng_Latn")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--rule", choices=["clean", "plain"], default="clean",
                   help="clean: compare without punctuation, never show a sentence end early; "
                        "plain: the first version, raw word comparison")
    a = p.parse_args()

    import torch
    device = P.pick_device(torch)
    from pathlib import Path
    tr = P.make_chat(P.ARMS[a.arm], torch, device, a.max_new_tokens, Path("results") / "walk" / a.arm)
    d = (a.src, a.tgt)
    pairs = P.load_flores(a.src, a.tgt, max(a.i) + 1)

    for i in a.i:
        src, ref = pairs[i]
        words = P.split_src(src, a.src)
        joiner = "" if a.src == "zho_Hans" else " "
        shown, prev_full, rows = "", None, []
        for k in range(1, len(words) + 1):
            prefix = joiner.join(words[:k])
            if shown:
                cont = tr.continue_prefix(prefix, d, shown)
                full = join_cont(shown, cont).strip()
            else:
                full = tr.translate(prefix, d)[0]
            if k == len(words):
                new_shown = full
            elif prev_full is not None:
                agreed = agree_clean(prev_full, full, quote_closed=any(c in prefix for c in "“”»\"")) if a.rule == "clean" else lcp_words(prev_full, full)
                new_shown = agreed if len(agreed.split()) > len(shown.split()) else shown
            else:
                new_shown = shown
            added = new_shown[len(shown):].strip()
            rows.append((k, words[k - 1], full, added, new_shown))
            shown, prev_full = new_shown, full
        print_table(i, src, ref, rows, shown)
        write_md(i, src, ref, rows, shown, a)
    if hasattr(tr, "close"):
        tr.close()


HEAD = ("Step", "New German word", "Model's full translation at this step", "Added to screen",
        "On screen so far")
WIDTHS = (4, 16, 58, 26)


def print_table(i, src, ref, rows, shown):
    import shutil, textwrap
    from itertools import zip_longest
    cols = shutil.get_terminal_size((120, 40)).columns
    rest = max(60, cols - 4 - 12 - 16)          # room for the three text columns
    WIDTHS = (4, 12, rest * 2 // 5, rest // 5, rest - rest * 2 // 5 - rest // 5)
    line = "+" + "+".join("-" * (w + 2) for w in WIDTHS) + "+"
    def emit(cells):
        wrapped = [textwrap.wrap(str(c), w) or [""] for c, w in zip(cells, WIDTHS)]
        for parts in zip_longest(*wrapped, fillvalue=""):
            print("| " + " | ".join(p.ljust(w) for p, w in zip(parts, WIDTHS)) + " |")
    print(f"\nSENTENCE {i}")
    print("German   : " + src)
    print("Reference: " + ref + "\n")
    print(line); emit(HEAD); print(line.replace("-", "="))
    for k, word, full, added, now in rows:
        emit((k, word, full, added or "(waiting)", now))
        print(line)
    print("\nFinal text on screen: " + shown + "\n")


def write_md(i, src, ref, rows, shown, a):
    from pathlib import Path
    out = Path("results") / "walk" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("|", "\\|")
    lines = [f"# Streaming walk, sentence {i} ({a.arm}, {a.src} to {a.tgt}, rule: {a.rule})", "",
             f"**German:** {src}", "", f"**Reference:** {ref}", "",
             "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    for k, word, full, added, now in rows:
        lines.append(f"| {k} | {esc(word)} | {esc(full)} | {esc(added) or '*(waiting)*'} | {esc(now)} |")
    lines += ["", f"**Final text on screen:** {shown}", ""]
    f = out / f"{a.src}-{a.tgt}_{i}_{a.rule}.md"
    f.write_text("\n".join(lines), encoding="utf-8")
    print(f"(saved {f})")

if __name__ == "__main__":
    main()
