from pathlib import Path

import torch

from .model import HybridLM
from .tokenizer import load_tokenizer


def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_ckpt(path):
    p = Path(path)
    if not p.exists():
        runs = sorted(x.name for x in p.parent.glob("*") if x.is_dir()) if p.parent.exists() else []
        raise FileNotFoundError(
            f"Run '{p}' not found (relative to {Path.cwd()}). "
            + (f"Runs in '{p.parent}': {runs}" if runs else f"'{p.parent}' does not exist or is empty."))
    if p.is_dir():
        for name in ("final.pt", "last.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"No final.pt/last.pt in {p}")
    return p


def load_model(path, device="auto"):
    """Load a trained run (directory or .pt). Returns model, cfg, tokenizer, specials, device."""
    ckpt_path = resolve_ckpt(path)
    dev = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(cfg["data"]["dir"]) / "tokenizer.json"
    tok, sp = load_tokenizer(tok_path)
    model = HybridLM(cfg["model"], tok.get_vocab_size(), sp)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    return model, cfg, tok, sp, dev