"""Paired bootstrap (Koehn 2004) over the report runs.

Every round draws the test sentences with replacement, once, and scores every run on that
same draw, so a difference between two runs is always computed on the same sentences and
the difficulty they share cancels. The unit is the sentence, never the step: steps within
a sentence are correlated. Two draws: the 712 report sentences for COMET and LAAL, the 100
RQ2 sentences (rq2_ids.json) for committed guesses per 100 German words, and (once
m3_judge.py has run) the committed guesses that turned out wrong.

Prints a 95% interval for every run, then the paired comparisons: delta (second minus first),
95% interval of the delta, and a two-sided p (twice the share of rounds on the smaller side).
The arms are also compared at matched lag: each round, each arm's COMET is interpolated along
its curve (one policy family, ordered by LAAL) at whole-word lags both arms cover.

Needs the sentence-level COMET files from score.py --comet (<run>.comet.json).
Usage, from the repository folder:
    python pipeline/bootstrap.py > results/bootstrap.txt
Also writes results/summary.json (point and 95% interval per run and metric) for report_table.py
and figures.py.
"""
import json
from pathlib import Path

import numpy as np

from rq2_count import m3_labels, per_sentence
from score import sentence_laal

ROUNDS = 10000
SEED = 2026
ARMS = ["hymt2-q4", "gemma-e2b-q4"]
FAMILIES = {
    "wait-k": ["waitk1", "waitk3", "waitk5"],
    "LA": ["la-s1", "la-s2", "la-s3"],
    "LA + verb rule": ["la-s1-verb", "la-s2-verb", "la-s3-verb"],
    "Grammar": ["grammar-d0", "grammar-d1", "grammar-d2"],
}
SETTINGS = [s for f in FAMILIES.values() for s in f]
FULL = "waitk999"  # the full-sentence baseline
N_REPORT = 712
LAG_OF = dict(comet="laal", guesses="laal_rq2", wrong="laal_rq2")  # the lag each metric is matched on (same sentences)


def run_path(arm: str, setting: str) -> Path:
    # the newest complete run of this setting
    files = [f for f in sorted(Path(f"results/pipeline/{arm}_{setting}").glob("*.jsonl"))
             if sum(1 for _ in open(f)) == N_REPORT]
    return files[-1]


def load(arm: str, setting: str) -> dict:
    path = run_path(arm, setting)
    rows = {r["idx"]: r for r in map(json.loads, open(path))}
    comet = {int(k): v for k, v in json.loads(path.with_suffix(".comet.json").read_text()).items()}
    run = dict(comet=np.array([comet[i] for i in IDS]),
               # nan for a sentence with no committed word, left out of that round's mean
               laal=np.array([sentence_laal(rows[i])[0] if rows[i]["delays"] else np.nan for i in IDS]))
    judge = Path("results/judge") / f"{path.stem}_haiku.jsonl"
    if judge.exists():
        counts = per_sentence(judge, M3)
        run["guesses"] = np.array([counts[i]["committed"] for i in RQ2_IDS])
        if M3:
            run["wrong"] = np.array([counts[i]["wrong"] for i in RQ2_IDS])
        run["words"] = np.array([counts[i]["words"] for i in RQ2_IDS])
        run["raw_n"] = np.array([counts[i]["raw"] for i in RQ2_IDS])
        run["steps"] = np.array([counts[i]["steps"] for i in RQ2_IDS])
        pos = {i: k for k, i in enumerate(IDS)}
        run["laal_rq2"] = run["laal"][[pos[i] for i in RQ2_IDS]]
    return run


def stats(run: dict) -> dict:
    # every statistic, per round (arrays of length ROUNDS) and on the full sample ("point")
    out = dict(comet=run["comet"][I].mean(1), laal=np.nanmean(run["laal"][I], 1))
    point = dict(comet=run["comet"].mean(), laal=np.nanmean(run["laal"]))
    if "guesses" in run:
        out["guesses"] = 100 * run["guesses"][J].sum(1) / run["words"][J].sum(1)
        point["guesses"] = 100 * run["guesses"].sum() / run["words"].sum()
        # raw: share of mid-sentence model replies that contain a guess, in percent
        out["raw"] = 100 * run["raw_n"][J].sum(1) / run["steps"][J].sum(1)
        point["raw"] = 100 * run["raw_n"].sum() / run["steps"].sum()
        # lag on the RQ2 sentences, so guesses are matched against lag on the same draw
        out["laal_rq2"] = np.nanmean(run["laal_rq2"][J], 1)
        point["laal_rq2"] = np.nanmean(run["laal_rq2"])
    if "wrong" in run:
        out["wrong"] = 100 * run["wrong"][J].sum(1) / run["words"][J].sum(1)
        point["wrong"] = 100 * run["wrong"].sum() / run["words"].sum()
    return dict(rounds=out, point=point)


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(x, [2.5, 97.5]))


def p_two_sided(d: np.ndarray) -> float:
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def fmt_p(p: float) -> str:
    return f"<{1 / ROUNDS:.4f}" if p == 0 else f"{p:.4f}"


def compare(label: str, a: dict, b: dict, metrics=("comet", "laal", "raw", "guesses", "wrong")):
    for m in metrics:
        if m not in a["rounds"] or m not in b["rounds"]:
            continue
        d = b["rounds"][m] - a["rounds"][m]
        lo, hi = ci(d)
        delta = b["point"][m] - a["point"][m]
        digits = 4 if m == "comet" else 2
        print(f"  {label:34} {m:8} {delta:+.{digits}f}  [{lo:+.{digits}f}, {hi:+.{digits}f}]  p {fmt_p(p_two_sided(d))}")


def at_lag(arm: str, family: list[str], lags: list[int], metric: str = "comet") -> np.ndarray:
    # each round: this arm's metric interpolated at the given lags along the family's curve
    lk = LAG_OF[metric]
    lag = np.stack([S[arm, s]["rounds"][lk] for s in family], 1)
    com = np.stack([S[arm, s]["rounds"][metric] for s in family], 1)
    order = np.argsort(lag, 1)
    lag, com = np.take_along_axis(lag, order, 1), np.take_along_axis(com, order, 1)
    return np.array([np.interp(lags, l, c) for l, c in zip(lag, com)])


def point_at_lag(arm: str, family: list[str], lag: int, metric: str = "comet") -> float:
    # the same interpolation on the full sample
    lk = LAG_OF[metric]
    return np.interp(lag, *zip(*sorted((S[arm, s]["point"][lk], S[arm, s]["point"][metric]) for s in family)))


def matched(label: str, a: tuple[str, list[str]], b: tuple[str, list[str]], names: tuple[str, str],
            metric: str = "comet"):
    # metric of curve b minus curve a, at the whole-word lags both curves cover
    lk, digits = LAG_OF[metric], 4 if metric == "comet" else 2
    lo = max(min(S[arm, s]["point"][lk] for s in fam) for arm, fam in (a, b))
    hi = min(max(S[arm, s]["point"][lk] for s in fam) for arm, fam in (a, b))
    lags = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    ra, rb = at_lag(*a, lags, metric), at_lag(*b, lags, metric)
    for k, lag in enumerate(lags):
        d = rb[:, k] - ra[:, k]
        pa, pb = point_at_lag(*a, lag, metric), point_at_lag(*b, lag, metric)
        c_lo, c_hi = ci(d)
        print(f"  {label:16} at {lag} words   {names[0]} {pa:.{digits}f}  {names[1]} {pb:.{digits}f}  "
              f"delta {pb - pa:+.{digits}f}  [{c_lo:+.{digits}f}, {c_hi:+.{digits}f}]  p {fmt_p(p_two_sided(d))}")


if __name__ == "__main__":
    first = run_path(ARMS[0], SETTINGS[0])
    IDS = sorted(json.loads(line)["idx"] for line in open(first))
    RQ2_IDS = sorted(json.loads(Path("rq2_ids.json").read_text()))
    M3 = m3_labels()  # wrong committed guesses (M3), if m3_judge.py has run
    rng = np.random.default_rng(SEED)
    I = rng.integers(0, len(IDS), (ROUNDS, len(IDS)))  # one draw shared by every run
    J = rng.integers(0, len(RQ2_IDS), (ROUNDS, len(RQ2_IDS)))

    S = {(arm, s): stats(load(arm, s)) for arm in ARMS for s in SETTINGS + [FULL]}
    print(f"Paired bootstrap, {ROUNDS} rounds, seed {SEED}: {len(IDS)} sentences for COMET and LAAL, "
          f"{len(RQ2_IDS)} for committed guesses per 100 German words\n")

    print("95% intervals per run")
    for arm in ARMS:
        for s in SETTINGS + [FULL]:
            st = S[arm, s]
            cells = []
            for m, digits in (("comet", 3), ("laal", 2), ("raw", 1), ("guesses", 2), ("wrong", 2)):
                if m in st["point"]:
                    lo, hi = ci(st["rounds"][m])
                    cells.append(f"{m} {st['point'][m]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]")
            print(f"  {arm + ' ' + s:26} " + "   ".join(cells))
    print()

    # the same intervals for the table and figure scripts
    Path("results/summary.json").write_text(json.dumps({
        f"{arm}_{s}": {m: dict(point=float(S[arm, s]["point"][m]), lo=float(ci(S[arm, s]["rounds"][m])[0]),
                               hi=float(ci(S[arm, s]["rounds"][m])[1])) for m in S[arm, s]["point"]}
        for arm in ARMS for s in SETTINGS + [FULL]}, indent=1))

    print("Gemma minus Hy-MT2, same setting")
    for s in SETTINGS + [FULL]:
        compare(s, S[ARMS[0], s], S[ARMS[1], s])
    print()

    print("Gemma minus Hy-MT2, COMET at matched lag (LAAL in words)")
    for name, family in FAMILIES.items():
        matched(name, (ARMS[0], family), (ARMS[1], family), ("Hy-MT2", "Gemma"))
    print()

    print("Verb rule on minus off, COMET at matched lag: is the rule better than waiting longer?")
    for arm in ARMS:
        matched(arm, (arm, FAMILIES["LA"]), (arm, FAMILIES["LA + verb rule"]), ("off", "on"))
    print()

    print("Verb rule on minus off, committed guesses per 100 German words at matched lag (LAAL on the RQ2 sentences)")
    for arm in ARMS:
        matched(arm, (arm, FAMILIES["LA"]), (arm, FAMILIES["LA + verb rule"]), ("off", "on"), "guesses")
    print()

    print("Gemma minus Hy-MT2, committed guesses per 100 German words at matched lag (LAAL on the RQ2 sentences)")
    for name, family in FAMILIES.items():
        matched(name, (ARMS[0], family), (ARMS[1], family), ("Hy-MT2", "Gemma"), "guesses")
    print()

    if M3:
        print("Verb rule on minus off, WRONG committed guesses per 100 German words at matched lag (RQ2 sentences)")
        for arm in ARMS:
            matched(arm, (arm, FAMILIES["LA"]), (arm, FAMILIES["LA + verb rule"]), ("off", "on"), "wrong")
        print()
        print("Gemma minus Hy-MT2, WRONG committed guesses per 100 German words at matched lag (RQ2 sentences)")
        for name, family in FAMILIES.items():
            matched(name, (ARMS[0], family), (ARMS[1], family), ("Hy-MT2", "Gemma"), "wrong")
        print()

    print("Verb rule on minus off")
    for arm in ARMS:
        for stride in (1, 2, 3):
            compare(f"{arm} s{stride}", S[arm, f"la-s{stride}"], S[arm, f"la-s{stride}-verb"])
    print()

    print("Grammar, one more word of delay")
    for arm in ARMS:
        for d in (0, 1):
            compare(f"{arm} d{d} -> d{d + 1}", S[arm, f"grammar-d{d}"], S[arm, f"grammar-d{d + 1}"])
    print()

    print("Streaming minus full sentence, COMET")
    for arm in ARMS:
        for s in SETTINGS:
            compare(f"{arm} {s}", S[arm, FULL], S[arm, s], metrics=("comet",))
