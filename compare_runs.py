"""Paired comparison of evaluated runs (same problems, so differences are tested per problem).

    python compare_runs.py --runs kanvq_hybrid kanvq_text vanilla_text vanilla_hybrid --sets synth gsm8k openmath
    python compare_runs.py ... --ref kanvq_text:text --out results.md

Reads runs/<run>/eval/<set>_<mode>.jsonl (written by eval.py). For each (run, mode) it reports accuracy
with a 95% Wilson interval, mean text/latent/total steps, and an exact McNemar test against the reference
on the problems both answered. For synthetic sets it splits in-distribution (train depths) vs OOD depths.
"""
import argparse
import json
import math
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from discordant counts b (ref right, other wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["synth"])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--ref", default="kanvq_text:text", help="run:mode used as reference for McNemar")
    ap.add_argument("--train_depth", nargs=2, type=int, default=[2, 8])
    ap.add_argument("--out", default=None, help="also write the tables to this markdown file")
    args = ap.parse_args()
    lines = []
    out = lambda s="": (print(s), lines.append(s))

    for st in args.sets:
        recs = {}
        for run in args.runs:
            for f in sorted((Path(args.root) / run / "eval").glob(f"{st}_*.jsonl")):
                mode = f.stem[len(st) + 1:]
                recs[f"{run}:{mode}"] = load(f)
        if not recs:
            out(f"(no results for set '{st}')")
            continue
        ref = recs.get(args.ref)
        splits = [("all", lambda r: True)]
        if any(r.get("depth") is not None for v in recs.values() for r in v):
            lo, hi = args.train_depth
            splits += [(f"ID d{lo}-{hi}", lambda r: r.get("depth") is not None and lo <= r["depth"] <= hi),
                       (f"OOD d>{hi}", lambda r: r.get("depth") is not None and r["depth"] > hi)]
        for split_name, keep in splits:
            out(f"\n### {st} — {split_name}  (reference: {args.ref})\n")
            out("| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |")
            out("|---|---|---|---|---|---|---|---|---|")
            ref_map = {r["question"]: r for r in ref if keep(r)} if ref else {}
            for key, rs in recs.items():
                rs = [r for r in rs if keep(r)]
                n = len(rs)
                if n == 0:
                    continue
                k = sum(r["correct"] for r in rs)
                lo_ci, hi_ci = wilson(k, n)
                mean = lambda f: sum(r[f] for r in rs) / n
                cmp = "—"
                if ref_map and key != args.ref:
                    pairs = [(ref_map[r["question"]]["correct"], r["correct"]) for r in rs if r["question"] in ref_map]
                    b = sum(1 for a, o in pairs if a and not o)
                    c = sum(1 for a, o in pairs if o and not a)
                    cmp = f"{b}/{c} | {mcnemar_p(b, c):.3g}"
                else:
                    cmp = "— | —"
                out(f"| {key} | {n} | {100*k/n:.1f} | {100*lo_ci:.1f}–{100*hi_ci:.1f} | {mean('n_text'):.1f} | "
                    f"{mean('n_latent'):.1f} | {mean('steps'):.1f} | {cmp} |")
            if split_name == "all" and any(r.get("depth") is not None for v in recs.values() for r in v):
                depths = sorted({r["depth"] for v in recs.values() for r in v if r.get("depth") is not None})
                out(f"\nAccuracy by depth (%):\n")
                out("| run:mode | " + " | ".join(str(d) for d in depths) + " |")
                out("|---|" + "---|" * len(depths))
                for key, rs in recs.items():
                    row = []
                    for d in depths:
                        dd = [r for r in rs if r.get("depth") == d]
                        row.append(f"{100*sum(r['correct'] for r in dd)/len(dd):.0f}" if dd else "–")
                    out(f"| {key} | " + " | ".join(row) + " |")
    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
