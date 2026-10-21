"""Held-out synthetic DEV set for tuning inference knobs (exit rule/threshold, latent caps) without
touching the test set eval_synth.jsonl. Different seed; any question also in eval_synth is skipped.

    python make_dev_set.py --config config.json --per_depth 50
    python eval.py --run runs/kanvq_hybrid --sets synthdev --limit 0 --modes hybrid --exit_rule threshold --exit_threshold 0.4 --tag _thr0.4
"""
import argparse
import json
import random
from pathlib import Path

from hybrid.config import load_config
from hybrid.data import load_jsonl, make_chain_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--per_depth", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc, root = cfg["data"]["synth"], Path(cfg["data"]["dir"])
    test_q = {e["question"] for e in load_jsonl(root / "eval_synth.jsonl")}
    rng = random.Random(args.seed)
    lo, hi = sc["eval_depth"]
    out = []
    for d in range(lo, hi + 1):
        n = 0
        while n < args.per_depth:
            q, _, a = make_chain_problem(rng, d, rng.randint(*sc.get("distractors", [0, 2])))
            if q in test_q:
                continue
            out.append(dict(question=q, answer=a, depth=d))
            n += 1
    path = root / "eval_synthdev.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(out)} problems to {path}")


if __name__ == "__main__":
    main()
