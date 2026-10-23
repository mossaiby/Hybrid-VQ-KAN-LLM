"""Pretrain a HybridLM (or a baseline) from scratch.

    python train.py --config config.json
    python train.py --config configs/vanilla_text.json
    python train.py --config config.json --resume
    python train.py --config config.json --set training.micro_batch=2 --set training.total_tokens=2000000
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from hybrid.checkpoint import pick_amp_dtype, pick_device
from hybrid.config import load_config, ramp
from hybrid.data import BatchBuilder, Prefetcher, load_jsonl
from hybrid.generation import evaluate_items
from hybrid.model import HybridLM, compute_losses
from hybrid.tokenizer import load_tokenizer


def lr_at(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * prog))


def build_val_set(val_b, cfg, n_batches, latent_prob_end):
    """Fixed validation set, built once from a fixed seed (identical across evals, resumes and arms).
    text[name]: pure-text batches per source; hybrid: reasoning batches at the final latent probability.
    TF and self-conditioned metrics are computed on the SAME hybrid batches, so their gap is meaningful."""
    bs = cfg["training"]["micro_batch"]
    as_t = lambda pair: tuple(torch.from_numpy(a) for a in pair)
    vs = {"text": {}, "hybrid": []}
    val_b.set_latent_prob(0.0)
    for name in val_b.names:
        vs["text"][name] = [as_t(val_b.next(bs, only=name)) for _ in range(n_batches)]
    reasoning = [n for n in val_b.names if n != "general"]
    if cfg["hybrid"].get("enabled") and reasoning:
        val_b.set_latent_prob(latent_prob_end)
        vs["hybrid"] = [as_t(val_b.next(bs, only=reasoning[i % len(reasoning)])) for i in range(n_batches)]
    return vs


@torch.no_grad()
def validate(model, vs, cfg, amp_ctx):
    """Validation CE per source (pure text) + link metrics on fixed hybrid batches (TF and self-conditioned)."""
    model.eval()
    h = cfg["hybrid"]
    out = {}
    for name, batches in vs["text"].items():
        vals = []
        for ids, grp in batches:
            with amp_ctx():
                _, m = compute_losses(model, ids.to(DEVICE), grp.to(DEVICE), h, 0, with_metrics=False)
            vals.append(m["ce"].item())
        out[f"val_ce_{name}"] = float(np.mean(vals))
    if vs["hybrid"]:
        for sc, tag in ((0, "tf"), (max(1, h["self_cond"].get("iters", 1)), "sc")):
            acc = {}
            for ids, grp in vs["hybrid"]:
                with amp_ctx():
                    _, m = compute_losses(model, ids.to(DEVICE), grp.to(DEVICE), h, sc)
                for k, v in m.items():
                    acc.setdefault(k, []).append(float(v))
            for k, v in acc.items():
                out[f"val_hyb_{tag}_{k}"] = float(np.mean(v))
    model.train()
    return out


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. training.max_lr=1e-3")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run, m, t, h = cfg["run"], cfg["model"], cfg["training"], cfg["hybrid"]
    run_dir = Path(run.get("out_root", "runs")) / run["name"]
    if not args.resume and any((run_dir / f).exists() for f in ("final.pt", "last.pt", "last.pt.prev")):
        raise SystemExit(f"{run_dir} already contains a checkpoint. Use --resume to continue it, or choose "
                         f"another name (--set run.name=...). Refusing to append to an existing run's logs.")
    if args.resume and (run_dir / "final.pt").exists():
        print(f"{run_dir} is already finished (final.pt exists); nothing to do.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = run.get("seed", 1337)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    DEVICE = pick_device(args.device)
    amp_dtype = pick_amp_dtype(DEVICE, t.get("amp", "auto"))
    # Autocast's weight-cast cache stays on; compute_losses clears it after the self-conditioning
    # no-grad pass (see the comment there).
    amp_ctx = ((lambda: torch.autocast(DEVICE.type, dtype=amp_dtype)) if amp_dtype
               else (lambda: torch.autocast("cpu", enabled=False)))
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(cfg["data"]["dir"])
    tok, sp = load_tokenizer(data_dir / "tokenizer.json")
    shutil.copy(data_dir / "tokenizer.json", run_dir / "tokenizer.json")
    vocab = tok.get_vocab_size()

    model = HybridLM(m, vocab, sp).to(DEVICE)
    n_params, n_ne = model.num_params(), model.num_params(non_embedding=True)
    tokens_per_step = t["micro_batch"] * t["grad_accum"] * m["max_seq_len"]
    if t["total_tokens"] == "chinchilla":
        # Same token budget for every arm: 20 x params of the *reference* arm (the full KAN-VQ hybrid
        # model at this scale), counted on the meta device so nothing is allocated.
        # logit_bias excluded so the budget is identical to the v1 runs (keeps vanilla_text comparable)
        ref_m = dict(m, **t.get("budget_reference", {"mlp_type": "kan", "sphere": True, "link_type": "kan",
                                                      "logit_bias": False}))
        with torch.device("meta"):
            ref_params = HybridLM(ref_m, vocab, sp).num_params()
        total_tokens = int(t.get("chinchilla_mult", 20) * ref_params)
        print(f"chinchilla budget from reference arm: {ref_params/1e6:.2f}M params")
    else:
        total_tokens = int(t["total_tokens"])
    max_steps = max(1, math.ceil(total_tokens / tokens_per_step))
    warmup = max(1, int(t.get("warmup_frac", 0.02) * max_steps))
    print(f"run={run['name']}  device={DEVICE}  amp={amp_dtype}")
    print(f"params: total {n_params/1e6:.2f}M | non-embedding {n_ne/1e6:.2f}M | link "
          f"{sum(p.numel() for p in model.link_mod.parameters())/1e6:.2f}M")
    print(f"tokens: {total_tokens/1e6:.1f}M ({total_tokens/n_params:.1f} tok/param) | "
          f"{tokens_per_step} tok/step | {max_steps} steps")
    cfg["_resolved"] = dict(n_params=n_params, n_params_non_embedding=n_ne, total_tokens=total_tokens,
                            max_steps=max_steps, vocab_size=vocab)
    (run_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2))

    opt = torch.optim.AdamW(model.param_groups(t["weight_decay"]), lr=t["max_lr"],
                            betas=tuple(t.get("betas", (0.9, 0.95))), eps=1e-8,
                            fused=DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    clip_sep = bool(t.get("clip_link_separately", True))
    link_params = list(model.link_mod.parameters())
    link_ids = {id(p) for p in link_params}
    backbone_params = [p for p in model.parameters() if id(p) not in link_ids]
    if t.get("compile", False):
        # Compile only the FFNs: in the KAN model they are dominated by memory-bound elementwise work
        # (tanh, basis, concat) that compilation fuses, and they take no cache/mask arguments, so the
        # generation path (KV cache, varying lengths) never forces graph breaks or many recompiles.
        for blk in model.blocks:
            blk.mlp.compile()
        print("torch.compile enabled for FFN modules (first steps are slow while compiling)")

    start_step, tokens_seen = 0, 0
    if args.resume and ((run_dir / "last.pt").exists() or (run_dir / "last.pt.prev").exists()):
        ck = None
        for cand in ("last.pt", "last.pt.prev"):  # .prev covers a crash in the middle of saving
            if (run_dir / cand).exists():
                try:
                    ck = torch.load(run_dir / cand, map_location="cpu", weights_only=False)
                    print(f"loaded {cand}")
                    break
                except Exception as e:  # e.g. truncated by a crash in an older version without atomic saves
                    print(f"{cand} is unreadable ({e})")
        if ck is None:
            raise SystemExit(f"No readable checkpoint in {run_dir}; restart this run by deleting the folder.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_step, tokens_seen = ck["step"], ck["tokens_seen"]
        print(f"resumed from step {start_step}")

    def save(name, step):
        # Atomic: write a temporary file, then rename over the old checkpoint, so a crash during saving
        # can never leave a corrupt last.pt. The previous checkpoint is kept as <name>.prev.
        tmp, dst = run_dir / (name + ".tmp"), run_dir / name
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                        step=step, tokens_seen=tokens_seen, config=cfg), tmp)
        if dst.exists():
            os.replace(dst, run_dir / (name + ".prev"))
        os.replace(tmp, dst)

    train_b = BatchBuilder(cfg, sp, "train", seed + start_step)
    val_b = BatchBuilder(cfg, sp, "val", 12345)  # fixed seed: same val set for every arm and resume
    lp_end_val = h["latent_prob"]["end"] if h.get("enabled") else 0.0
    val_set = build_val_set(val_b, cfg, t.get("eval_batches", 20), lp_end_val)
    pf = Prefetcher(train_b, t["micro_batch"], t.get("prefetch", 4), pin=DEVICE.type == "cuda")
    quick = []
    qe = t.get("quick_eval", {})
    if qe.get("samples", 0) > 0 and (data_dir / qe.get("file", "eval_synth.jsonl")).exists():
        quick = load_jsonl(data_dir / qe.get("file", "eval_synth.jsonl"))
        random.Random(0).shuffle(quick)
        quick = quick[: qe["samples"]]

    log_f = open(run_dir / "log.csv", "a", newline="")
    log_w = None
    val_f = open(run_dir / "val.jsonl", "a")
    min_lr = t["max_lr"] * t.get("min_lr_frac", 0.1)
    model.train()
    t_last, tok_last = time.time(), tokens_seen

    for step in range(start_step, max_steps):
        prog = step / max_steps
        lr = lr_at(step, warmup, max_steps, t["max_lr"], min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        lp = ramp(prog, h["latent_prob"]) if h.get("enabled") else 0.0
        sc_p = ramp(prog, h["self_cond"]) if h.get("enabled") else 0.0
        train_b.set_latent_prob(lp)

        agg = {}
        for _ in range(t["grad_accum"]):
            ids, grp = pf.get()
            ids, grp = ids.to(DEVICE, non_blocking=True), grp.to(DEVICE, non_blocking=True)
            sc_iters = h["self_cond"].get("iters", 1) if random.random() < sc_p else 0
            with amp_ctx():
                loss, mets = compute_losses(model, ids, grp, h, sc_iters,
                                            with_metrics=(step % t["log_interval"] == 0),
                                            detach_link_input=prog < h.get("link_detach_until", 0.0))
            scaler.scale(loss / t["grad_accum"]).backward()
            for k, v in mets.items():
                agg.setdefault(k, []).append(v)
        scaler.unscale_(opt)
        if clip_sep:  # v3: the link is clipped on its own, so its large early gradients cannot shrink
            # every backbone update through a shared global norm (observed in vanilla_hybrid_v2)
            gnorm = torch.nn.utils.clip_grad_norm_(backbone_params, t.get("grad_clip", 1.0))
            lnorm = torch.nn.utils.clip_grad_norm_(link_params, t.get("link_grad_clip", t.get("grad_clip", 1.0)))
        else:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            lnorm = torch.zeros(())
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if step % t["log_interval"] == 0 or step == max_steps - 1:
            dt = time.time() - t_last
            row = dict(step=step, tokens=tokens_seen, lr=lr, latent_prob=lp, self_cond_prob=sc_p,
                       grad_norm=float(gnorm), link_grad_norm=float(lnorm),
                       tok_per_s=(tokens_seen - tok_last) / max(dt, 1e-9))
            row.update({k: float(torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in v]).mean())
                        for k, v in agg.items()})
            if model.sphere:
                row["logit_scale"] = float(model.logit_scale())
            if DEVICE.type == "cuda":
                row["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            t_last, tok_last = time.time(), tokens_seen
            keys = ["step", "tokens", "lr", "ce", "link", "readability", "link_cos", "exit_acc", "false_exit",
                    "slot_tok_acc", "n_latent", "latent_prob", "self_cond_prob", "grad_norm", "link_grad_norm", "tok_per_s",
                    "logit_scale", "mem_gb"]
            if log_w is None:
                log_w = csv.DictWriter(log_f, fieldnames=keys, extrasaction="ignore")
                if log_f.tell() == 0:
                    log_w.writeheader()
            log_w.writerow(row)
            log_f.flush()
            extra = f" link {row['link']:.3f} exit {row.get('exit_acc', float('nan')):.2f}" if h.get("enabled") else ""
            print(f"step {step:>6}/{max_steps} | {tokens_seen/1e6:8.1f}M tok | lr {lr:.2e} | ce {row['ce']:.3f}"
                  f"{extra} | p_lat {lp:.2f} p_sc {sc_p:.2f} | {row['tok_per_s']/1e3:.1f}k tok/s")

        last = step == max_steps - 1
        if (step > 0 and step % t["eval_interval"] == 0) or last:
            res = dict(step=step, tokens=tokens_seen)
            res.update(validate(model, val_set, cfg, amp_ctx))
            if quick:
                model.eval()
                icfg = dict(cfg["inference"], max_new_steps=qe.get("max_new_steps", 256))
                modes = [("hybrid", True), ("text", False)] if h.get("enabled") else [("text", False)]
                for tag, allow in modes:
                    print(f"  [quick eval] {len(quick)} problems, {tag} mode ...", flush=True)
                    s, _ = evaluate_items(model, tok, quick, icfg, allow, amp_dtype=amp_dtype, progress=True,
                                         batch_size=qe.get("batch_size", 50))
                    res.update({f"quick_{tag}_{k}": v for k, v in s.items() if not isinstance(v, dict)})
                model.train()
            val_f.write(json.dumps(res) + "\n")
            val_f.flush()
            print("  [val] " + "  ".join(f"{k}={v:.4g}" for k, v in res.items()
                                        if isinstance(v, float) and any(s in k for s in ("ce", "acc", "steps", "false_exit"))))
            t_last = time.time()  # keep eval time out of the tok/s measurement
        if (step > 0 and step % t["save_interval"] == 0) or last:
            save("last.pt", step + 1)
            t_last = time.time()
    save("final.pt", max_steps)
    print(f"done. checkpoints in {run_dir}")


DEVICE = None
if __name__ == "__main__":
    main()