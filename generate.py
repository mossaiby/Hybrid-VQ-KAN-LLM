"""Generate from a trained run.

    python generate.py --run runs/kanvq_hybrid --question "Let a = 3. Let b = a + 4. What is b?"
    python generate.py --run runs/kanvq_hybrid                 # interactive
    python generate.py --run runs/kanvq_hybrid --mode text     # block latent mode
    python generate.py --run runs/kanvq_hybrid --raw "Once upon a time"   # free text continuation

Latent spans are printed as ⟨nearest-token readout⟩ (paper Fig. 2); --hide-latent prints only the
number of latent steps, i.e. what an external monitor would see.
"""
import argparse

import torch

from hybrid.checkpoint import load_model, pick_amp_dtype
from hybrid.data import format_prompt
from hybrid.generation import extract_boxed, hybrid_generate, render, visible_text
from hybrid.tokenizer import encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory or checkpoint .pt")
    ap.add_argument("--question", default=None, help="formatted with the training prompt template")
    ap.add_argument("--raw", default=None, help="raw text prompt (no template)")
    ap.add_argument("--mode", choices=["hybrid", "text"], default="hybrid")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_new_steps", type=int, default=None)
    ap.add_argument("--max_latent_steps", type=int, default=None)
    ap.add_argument("--min_latent_steps", type=int, default=None)
    ap.add_argument("--exit_rule", choices=["link", "head", "either"], default=None)
    ap.add_argument("--hide-latent", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model, cfg, tok, sp, dev = load_model(args.run, args.device)
    icfg = dict(cfg["inference"])
    for k in ("temperature", "top_k", "top_p", "max_new_steps", "max_latent_steps", "min_latent_steps", "exit_rule"):
        if getattr(args, k) is not None:
            icfg[k] = getattr(args, k)
    amp = pick_amp_dtype(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    allow = args.mode == "hybrid" and cfg["hybrid"].get("enabled", False)

    def run_one(prompt_text, templated):
        text = format_prompt(prompt_text) if templated else prompt_text
        r = hybrid_generate(model, [sp.bos] + encode(tok, text), icfg, allow, 0.0, amp, gen)
        print(text + render(tok, r["events"], not args.hide_latent, sp))
        ans = extract_boxed(visible_text(tok, r["events"], sp)) if templated else None
        print(f"\n--- steps {r['steps']} (text {r['n_text']}, latent {r['n_latent']}) | "
              f"{r['seconds']:.2f}s" + (f" | answer: {ans}" if ans is not None else ""))

    if args.question or args.raw:
        run_one(args.question or args.raw, args.question is not None)
        return
    print("Interactive mode. Prefix a line with 'raw:' for an untemplated prompt. Ctrl-C to quit.")
    while True:
        try:
            line = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line:
            raw = line.startswith("raw:")
            run_one(line[4:].strip() if raw else line, not raw)


if __name__ == "__main__":
    main()
