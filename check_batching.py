"""Is batched generation exactly equivalent to one-at-a-time generation (up to rounding)?

    python check_batching.py --run runs/smoke

Test 1 (the decisive one, fp32, no TF32): compare hidden states of padded-batch vs single-prompt
         forward passes, for the prompt prefill and for 40 decode steps that mix token inputs and
         continuous latent-like inputs. A masking/padding bug gives relative errors of order 1e-1..1;
         correct code gives ~1e-6..1e-5.
Test 2: greedy generation, batch size 1 vs 32, in fp32 and in bf16. Counts problems whose output
         differs and where the first divergence happens (text token / latent step / exit decision).
"""
import argparse
import random

import torch

from hybrid.checkpoint import load_model
from hybrid.data import format_prompt, load_jsonl
from hybrid.generation import hybrid_generate_batch, left_pad
from hybrid.tokenizer import encode
from pathlib import Path


@torch.no_grad()
def hidden_trace(model, prompts, steps_inputs):
    """Prefill (left-padded batch) then feed steps_inputs[j] (B, d) for each decode step.
    Returns list of (B, d) last-position hidden states: [prefill, step1, ...]."""
    dev = next(model.parameters()).device
    ids, kpm, mask = left_pad(prompts, model.sp.pad, dev)
    caches = model.new_caches()
    h = model.backbone(model.embed_ids(ids), caches, 0, mask)
    out, pos = [h[:, -1].float()], ids.shape[1]
    for x in steps_inputs:
        step_mask = None
        if mask is not None:
            kpm = torch.cat([kpm, torch.ones(len(prompts), 1, dtype=torch.bool, device=dev)], 1)
            step_mask = kpm[:, None, None, :]
        h = model.backbone(x[:, None, :], caches, pos, step_mask)
        pos += 1
        out.append(h[:, -1].float())
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0] or x[1] != y[1]:
            return i, x[0]
    return min(len(a), len(b)), "length"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, cfg, tok, sp, dev = load_model(args.run)
    items = load_jsonl(Path(cfg["data"]["dir"]) / "eval_synth.jsonl")
    random.Random(0).shuffle(items)
    items = items[: args.n]
    prompts = [[sp.bos] + encode(tok, format_prompt(e["question"])) for e in items]
    lens = [len(p) for p in prompts]
    print(f"{len(prompts)} prompts, lengths {min(lens)}..{max(lens)} tokens (so the batch is padded)")

    # ---- Test 1: hidden-state equivalence in fp32 -------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(0)
    steps = []
    for j in range(40):
        if j % 3 == 2:  # continuous, latent-like input on the sphere / embedding scale
            v = torch.randn(len(prompts), model.d, generator=g).to(dev)
            steps.append(model.slot_input(v).float())
        else:
            t = torch.randint(10, model.vocab_size, (len(prompts),), generator=g).to(dev)
            steps.append(model.embed_ids(t).float())
    batched = hidden_trace(model, prompts, steps)
    worst = 0.0
    for i, p in enumerate(prompts):
        single = hidden_trace(model, [p], [s[i:i + 1] for s in steps])
        for hb, hs in zip(batched, single):
            rel = ((hb[i] - hs[0]).norm() / hs[0].norm().clamp_min(1e-9)).item()
            worst = max(worst, rel)
    verdict = "OK (numerics only)" if worst < 1e-3 else "PROBLEM: batched forward differs -> likely a bug"
    print(f"Test 1  max relative hidden-state difference (fp32, 41 positions x {len(prompts)} rows): "
          f"{worst:.2e}  -> {verdict}")

    # ---- Test 2: generation agreement --------------------------------------------------------
    icfg = dict(cfg["inference"], temperature=0.0)
    for label, amp in (("fp32", None), ("bf16", torch.bfloat16 if dev.type == "cuda" else None)):
        if amp is None and label == "bf16":
            continue
        one = [hybrid_generate_batch(model, [p], icfg, True, 0.0, amp)[0] for p in prompts]
        bat = hybrid_generate_batch(model, prompts, icfg, True, 0.0, amp)
        diffs = []
        for a, b in zip(one, bat):
            ea, eb = [(k, t) for k, t, _ in a["events"]], [(k, t) for k, t, _ in b["events"]]
            if ea != eb:
                diffs.append(first_divergence(ea, eb))
        kinds = {}
        for _, k in diffs:
            kinds[k] = kinds.get(k, 0) + 1
        at = sorted(i for i, _ in diffs)
        print(f"Test 2  {label}: {len(diffs)}/{len(prompts)} outputs differ"
              + (f"; first divergence at event index {at} (kinds: {kinds})" if diffs else ""))


if __name__ == "__main__":
    main()
