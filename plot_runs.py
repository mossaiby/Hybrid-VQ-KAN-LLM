"""Compare runs:  python plot_runs.py runs/kanvq_hybrid runs/kanvq_text runs/vanilla_text --out compare.png"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()
    panels = [("train CE", "log", "ce"), ("val CE (openmath)", "val", "val_ce_openmath"),
              ("val CE (synth)", "val", "val_ce_synth"), ("val CE (general)", "val", "val_ce_general"),
              ("quick acc (hybrid)", "val", "quick_hybrid_acc"), ("quick acc (text)", "val", "quick_text_acc"),
              ("link loss (train)", "log", "link"), ("latent readability", "log", "readability"),
              ("quick steps (hybrid)", "val", "quick_hybrid_steps")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (title, src, key) in zip(axes.flat, panels):
        for r in args.runs:
            rows = read_csv(Path(r) / "log.csv") if src == "log" else read_jsonl(Path(r) / "val.jsonl")
            xs = [float(x["tokens"]) / 1e6 for x in rows if x.get(key) not in (None, "")]
            ys = [float(x[key]) for x in rows if x.get(key) not in (None, "")]
            if xs:
                ax.plot(xs, ys, label=Path(r).name, lw=1.4)
        ax.set_title(title); ax.set_xlabel("tokens (M)"); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
