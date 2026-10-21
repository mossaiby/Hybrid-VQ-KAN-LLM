"""Evaluate a run on the prepared eval sets.

    python eval.py --run runs/kanvq_hybrid --sets synth gsm8k openmath --limit 500
    python eval.py --run runs/kanvq_hybrid --sets synth --interventions all

Writes runs/<name>/eval/<set>_<mode>.json (summary) and .jsonl (per-sample records).
Generation is batched (--batch_size, default 32) for speed. ms/sample is then batch time / batch size;
for latency numbers in the paper, rerun with --batch_size 1. Evaluation runs in fp32 by default
(--precision), which makes results identical across batch sizes (verified by check_batching.py).

Modes: hybrid (latent allowed) and text (latent entry blocked), the analogue of the paper's
"Block Latent-Mode Entry" row. Interventions (paper Table 5 analogues):
  noise0.5, noise2.0  : Gaussian noise of norm sigma*||z|| on every latent input
  onestep             : every latent span forced to stop after one step
  quantize            : feed the nearest codebook vector instead of z (discrete "VQ" latent)
"""
import argparse
import json
from pathlib import Path

from hybrid.checkpoint import load_model, pick_amp_dtype
from hybrid.data import load_jsonl
from hybrid.generation import evaluate_items

INTERVENTIONS = {
    "noise0.5": dict(latent_noise=0.5),
    "noise2.0": dict(latent_noise=2.0),
    "onestep": dict(icfg=dict(max_latent_steps=1)),
    "quantize": dict(icfg=dict(latent_quantize=True)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["synth", "gsm8k"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--modes", nargs="+", default=None, help="hybrid text (default: both if hybrid)")
    ap.add_argument("--interventions", nargs="*", default=[], help=f"any of {list(INTERVENTIONS)} or 'all'")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="problems generated in parallel; use 1 for latency (ms/sample) measurements")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 (default) makes results independent of batch size; bf16 is faster but "
                         "rounding can flip greedy choices, so outputs vary with batch size")
    ap.add_argument("--exit_rule", choices=["link", "head", "either", "threshold"], default=None)
    ap.add_argument("--exit_threshold", type=float, default=None, help="for --exit_rule threshold")
    ap.add_argument("--max_latent_steps", type=int, default=None)
    ap.add_argument("--min_latent_steps", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _thr0.4 (avoids overwriting)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model, cfg, tok, sp, dev = load_model(args.run, args.device)
    if args.precision == "fp32":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        amp = None
    else:
        amp = pick_amp_dtype(dev)
    hybrid_on = cfg["hybrid"].get("enabled", False)
    modes = args.modes or (["hybrid", "text"] if hybrid_on else ["text"])
    data_dir = Path(args.data_dir or cfg["data"]["dir"])
    run_dir = Path(args.run) if Path(args.run).is_dir() else Path(args.run).parent
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    ivs = list(INTERVENTIONS) if args.interventions == ["all"] else args.interventions

    table = []
    for name in args.sets:
        path = data_dir / f"eval_{name}.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        items = load_jsonl(path)
        if args.limit and len(items) > args.limit:  # stratified-ish: keep a fixed, shuffled subset
            import random
            random.Random(0).shuffle(items)
            items = items[: args.limit]
        runs = [(m, m == "hybrid", {}) for m in modes]
        if hybrid_on:
            runs += [(f"hybrid+{iv}", True, INTERVENTIONS[iv]) for iv in ivs]
        for tag, allow, iv in runs:
            icfg = dict(cfg["inference"], **iv.get("icfg", {}))
            for k in ("exit_rule", "exit_threshold", "max_latent_steps", "min_latent_steps"):
                if getattr(args, k) is not None and k not in iv.get("icfg", {}):
                    icfg[k] = getattr(args, k)
            tag = tag + args.tag
            summ, recs = evaluate_items(model, tok, items, icfg, allow, iv.get("latent_noise", 0.0),
                                        amp, keep_records=True, progress=True, batch_size=args.batch_size)
            (out / f"{name}_{tag}.json").write_text(json.dumps(summ, indent=2))
            with open(out / f"{name}_{tag}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            table.append((name, tag, summ))
            print(f"{name:>9} {tag:<18} acc {summ['acc']*100:5.1f}% | steps {summ['steps']:6.1f} "
                  f"(text {summ['text_tokens']:6.1f}, latent {summ['latent_steps']:6.1f}) | "
                  f"{summ['sec_per_sample']*1e3:6.0f} ms/sample"
                  + (f" | read {summ['readability']:.3f}" if summ["readability"] is not None else ""))
            if "acc_by_depth" in summ:
                print("           by depth: " + " ".join(f"{d}:{a*100:.0f}" for d, a in summ["acc_by_depth"].items()))
    (out / "summary.json").write_text(json.dumps([dict(set=n, mode=t, **s) for n, t, s in table], indent=2))


if __name__ == "__main__":
    main()
