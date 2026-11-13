import argparse
import json
from pathlib import Path
from statistics import mean
from sacrebleu.metrics import CHRF, BLEU

WORDS_PER_SEC = 2.5

def laal(delays: list[float], tau: int, src_len: float, n_hyp: int, n_ref: int) -> float:
    rate = src_len / max(n_hyp, n_ref)
    return sum(d - i * rate for i, d in enumerate(delays[:tau])) / tau


def sentence_laal(r: dict) -> tuple[float, float]:
    # LAAL in words and CA-LAAL in seconds for one sentence (needs at least one committed word)
    n_src = r["n_src"]
    tau = next((i + 1 for i, d in enumerate(r["delays"]) if d == n_src), len(r["delays"]))
    n_hyp, n_ref = len(r["delays"]), len(r["ref"].split())
    return (laal(r["delays"], tau, n_src, n_hyp, n_ref),
            laal(r["ca_delays"], tau, n_src / WORDS_PER_SEC, n_hyp, n_ref))


def comet(model, rows: list[dict], path: str) -> float:
    data = [dict(src=r["src"], mt=r["hyp"], ref=r["ref"]) for r in rows]
    out = model.predict(data, batch_size=8, gpus=0, num_workers=1, progress_bar=False)
    # sentence scores next to the run file, for the bootstrap: <run>.comet.json, {idx: score}
    Path(path).with_suffix(".comet.json").write_text(
        json.dumps({r["idx"]: s for r, s in zip(rows, out.scores)}))
    return out.system_score



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--comet", action="store_true")
    args = ap.parse_args()
    comet_model = None
    if args.comet:
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    for path in args.files:
        rows = [json.loads(line) for line in open(path)]
        hyps = [r["hyp"] for r in rows]
        refs = [[r["ref"] for r in rows]]
        chrf = CHRF().corpus_score(hyps, refs).score
        bleu = BLEU().corpus_score(hyps, refs).score

        both = [sentence_laal(r) for r in rows if r["delays"]]
        laals, ca_laals = [b[0] for b in both], [b[1] for b in both]

        src_words = sum(r["n_src"] for r in rows)
        prompt_tok = sum(r["prompt_tokens"] for r in rows) / src_words
        output_tok = sum(r["output_tokens"] for r in rows) / src_words
        cached_tok = sum(r.get("cached_tokens", 0) for r in rows) / src_words
        ms_per_word = sum(r["compute_ms"] for r in rows) / src_words

        print(path, f"({len(rows)} sentences)")
        print(f"  chrF {chrf:.1f}   BLEU {bleu:.1f}" + (f"   COMET {comet(comet_model, rows, path):.3f}" if args.comet else ""))
        print(f"  LAAL {mean(laals):.2f} words ({mean(laals) / WORDS_PER_SEC:.2f} s)   CA-LAAL {mean(ca_laals):.2f} s")
        print(f"  tokens per source word: prompt {prompt_tok:.1f} computed + {cached_tok:.1f} cached, output {output_tok:.1f}")

        print(f"  compute per source word {ms_per_word:.0f} ms (real time allows {1000 / WORDS_PER_SEC:.0f} ms)")
