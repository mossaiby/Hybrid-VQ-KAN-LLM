"""Hybrid generation (paper Sec. 3.1) with a KV cache as the continuous working memory (Sec. 3.2).

Text mode : sample from the head; emitting <latent> switches to latent mode.
Latent mode: z = R(h_t) is fed back as the next input embedding. Exit when the nearest codebook
             entry of z is </latent> (exit_rule="link", Eq. 5), or the head's argmax is </latent>
             ("head"), or either ("either"), or cos(z, Emb(</latent>)) >= exit_threshold ("threshold"); spans are clamped to [min_latent, max_latent] steps.
Every forward pass is one "step"; latent steps are readable through their nearest-token readout.
"""
import re
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .data import format_prompt
from .tokenizer import decode, encode


def _sample_batch(logits, temperature, top_k, top_p, gen):
    """logits: (n, V) -> (n,) token ids. Greedy when temperature <= 0."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        remove = torch.cumsum(F.softmax(sl, -1), -1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(remove.scatter(1, si, remove), float("-inf"))
    return torch.multinomial(F.softmax(logits, -1), 1, generator=gen).squeeze(1)


def left_pad(prompts, pad_id, dev):
    """Left-pad token lists. Returns ids (B, L), key mask kpm (B, L) and the prefill attention mask
    (B, 1, L, L) = causal & real keys, with padded query rows attending to themselves (avoids NaN rows);
    the mask is None when no padding is needed (then the plain causal kernel is used)."""
    B, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    kpm = torch.zeros((B, L), dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
        kpm[i, L - len(p):] = True
    ids, kpm = ids.to(dev), kpm.to(dev)
    if bool(kpm.all()):
        return ids, kpm, None
    causal = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
    mask = (causal[None] & kpm[:, None, :]) | torch.eye(L, dtype=torch.bool, device=dev)[None]
    return ids, kpm, mask[:, None]


@torch.no_grad()
def hybrid_generate_batch(model, prompts, icfg, allow_latent=True, latent_noise=0.0,
                          amp_dtype=None, generator=None):
    """Generate for a list of prompts (token-id lists) in one batch; every row has its own mode.

    Prompts are left-padded and padding keys are masked. RoPE is relative, so the shared position
    offset does not change attention scores: greedy results match batch size 1 up to bf16 rounding.
    Rows that finished keep stepping with a zero input (ignored) until all rows are done.
    Returns one dict per prompt: events=[(kind, token_id, cos)], n_text, n_latent, steps, seconds
    (seconds = batch wall time / batch size; use batch size 1 for latency measurements)."""
    sp, dev = model.sp, next(model.parameters()).device
    ctx = (lambda: torch.autocast(dev.type, dtype=amp_dtype)) if amp_dtype else nullcontext
    max_steps = icfg.get("max_new_steps", 384)
    min_lat, max_lat = icfg.get("min_latent_steps", 0), icfg.get("max_latent_steps", 128)
    exit_rule, quantize = icfg.get("exit_rule", "link"), icfg.get("latent_quantize", False)
    temp, top_k, top_p = icfg.get("temperature", 0.0), icfg.get("top_k", 0), icfg.get("top_p", 1.0)
    limit = model.rope_cos.shape[0]
    keep = limit - max_steps - 1
    prompts = [list(p)[-keep:] if len(p) > keep else list(p) for p in prompts]
    B, L, d = len(prompts), max(len(p) for p in prompts), model.d

    ids, kpm, prefill_mask = left_pad(prompts, sp.pad, dev)
    padded = prefill_mask is not None

    banned = torch.tensor(sp.banned_in_text() + ([] if allow_latent else [sp.lat_open]), device=dev)
    close_in = model.embed_ids(torch.tensor([sp.lat_close], device=dev)).float()  # (1, d)
    caches = model.new_caches()
    t0 = time.perf_counter()
    with ctx():
        h = model.backbone(model.embed_ids(ids), caches, 0, prefill_mask)
    pos, h_last = L, h[:, -1].float()
    mode = [0] * B  # 0 text, 1 latent, 2 done
    run = torch.zeros(B, dtype=torch.long, device=dev)
    events = [[] for _ in range(B)]
    n_text, n_lat = [0] * B, [0] * B

    for _ in range(max_steps):
        nxt = torch.zeros(B, d, device=dev)
        text_rows = [i for i in range(B) if mode[i] == 0]
        lat_rows = [i for i in range(B) if mode[i] == 1]
        new_mode = list(mode)
        if text_rows:
            tr = torch.tensor(text_rows, device=dev)
            logits = model.logits(h_last[tr]).float()
            logits[:, banned] = float("-inf")
            toks = _sample_batch(logits, temp, top_k, top_p, generator)
            nxt[tr] = model.embed_ids(toks).float()
            for r, t in zip(text_rows, toks.tolist()):
                events[r].append(("text", t, None))
                n_text[r] += 1
                if t == sp.eos:
                    new_mode[r] = 2
                elif t == sp.lat_open:
                    new_mode[r] = 1
                    run[r] = 0
        if lat_rows:
            lr_ = torch.tensor(lat_rows, device=dev)
            z = model.link(h_last[lr_]).float()
            if latent_noise > 0:  # causal intervention (paper Table 5): noise of norm sigma*||z||
                n = torch.randn(z.shape, device=dev, generator=generator)
                z = z + latent_noise * z.norm(dim=-1, keepdim=True) * n / n.norm(dim=-1, keepdim=True)
            rid, rcos = model.readout(z)
            run[lr_] += 1
            want = rid == sp.lat_close
            if exit_rule in ("head", "either"):
                head = model.logits(h_last[lr_]).float().argmax(-1) == sp.lat_close
                want = head if exit_rule == "head" else (want | head)
            elif exit_rule == "threshold":  # exit if </latent> is close enough, even when not the argmax
                close_dir = F.normalize(model.embed.weight[sp.lat_close].float(), dim=-1)
                want = want | ((F.normalize(z, dim=-1) @ close_dir) >= icfg.get("exit_threshold", 0.5))
            r_run = run[lr_]
            exit_now = (r_run >= max_lat) | (want & (r_run > min_lat))
            zin = model.codebook()[rid].float() if quantize else z
            lat_in = model.slot_input(zin).float()
            nxt[lr_] = torch.where(exit_now[:, None], close_in.expand_as(lat_in), lat_in)
            for r, e, t, c in zip(lat_rows, exit_now.tolist(), rid.tolist(), rcos.tolist()):
                n_lat[r] += 1
                if e:
                    events[r].append(("exit", sp.lat_close, c))
                    new_mode[r] = 0
                else:
                    events[r].append(("latent", t, c))
        mode = new_mode
        if all(m == 2 for m in mode) or pos >= limit:
            break
        step_mask = None
        if padded:
            kpm = torch.cat([kpm, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
            step_mask = kpm[:, None, None, :]
        with ctx():
            h = model.backbone(nxt[:, None, :].to(model.embed.weight.dtype), caches, pos, step_mask)
        pos += 1
        h_last = h[:, -1].float()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / B
    return [dict(events=events[i], n_text=n_text[i], n_latent=n_lat[i], steps=len(events[i]), seconds=per)
            for i in range(B)]


def hybrid_generate(model, prompt_ids, icfg, allow_latent=True, latent_noise=0.0, amp_dtype=None, generator=None):
    """Single prompt (batch of one)."""
    return hybrid_generate_batch(model, [prompt_ids], icfg, allow_latent, latent_noise, amp_dtype, generator)[0]


def render(tok, events, show_latent=True, sp=None):
    """Text with latent spans shown as ⟨readout pseudo-text⟩ (or ⟨n latent steps⟩)."""
    out, buf_text, buf_lat = [], [], []

    def flush_text():
        if buf_text:
            out.append(decode(tok, buf_text))
            buf_text.clear()

    for kind, t, _ in events:
        if kind == "text":
            if sp is not None and t in (sp.lat_open, sp.eos):
                flush_text()
                continue
            buf_text.append(t)
        elif kind == "latent":
            flush_text()
            buf_lat.append(t)
        else:  # exit
            flush_text()
            if show_latent:
                out.append("⟨" + decode(tok, buf_lat).replace("\n", " ⏎ ") + "⟩")
            else:
                out.append(f"⟨{len(buf_lat)} latent steps⟩")
            buf_lat.clear()
    flush_text()
    if buf_lat:
        out.append("⟨" + decode(tok, buf_lat) + "…⟩")
    return "".join(out)


def visible_text(tok, events, sp):
    """Only the text-mode tokens (what an external monitor sees; used for answer extraction)."""
    return decode(tok, [t for k, t, _ in events if k == "text" and t not in (sp.lat_open, sp.eos)])


# --------------------------------------------------------------------------- answers
def extract_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        m = re.findall(r"answer is[:\s]*\$?(-?[\d,]*\.?\d+)", s)
        return m[-1] if m else None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j:k]
    return None


def normalize_answer(a):
    if a is None:
        return None
    a = a.strip().replace("$", "").replace("\\!", "").replace(",", "").replace("\\%", "").replace("%", "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a).replace(" ", "").rstrip(".")
    a = re.sub(r"\\[dt]?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", a)
    return a


def answers_match(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(eval(p, {"__builtins__": {}}, {})) - float(eval(g, {"__builtins__": {}}, {}))) < 1e-4
    except Exception:
        return False


# --------------------------------------------------------------------------- evaluation
def evaluate_items(model, tok, items, icfg, allow_latent=True, latent_noise=0.0, amp_dtype=None,
                   seed=0, keep_records=False, progress=False, batch_size=1):
    sp = model.sp
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    chunks = [items[i:i + batch_size] for i in range(0, len(items), max(1, batch_size))]
    if progress:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="eval", leave=False)
    recs = []
    for chunk in chunks:
        prompts = [[sp.bos] + encode(tok, format_prompt(ex["question"])) for ex in chunk]
        outs = hybrid_generate_batch(model, prompts, icfg, allow_latent, latent_noise, amp_dtype, gen)
        for ex, r in zip(chunk, outs):
            vis = visible_text(tok, r["events"], sp)
            pred = extract_boxed(vis)
            lat_cos = [c for k, _, c in r["events"] if k == "latent"]
            rec = dict(correct=answers_match(pred, ex["answer"]), pred=pred, gold=ex["answer"],
                       n_text=r["n_text"], n_latent=r["n_latent"], steps=r["steps"], seconds=r["seconds"],
                       readability=(sum(lat_cos) / len(lat_cos)) if lat_cos else None,
                       n_spans=sum(1 for k, _, _ in r["events"] if k == "exit"),
                       depth=ex.get("depth"))
            if keep_records:
                rec["question"] = ex["question"]
                rec["output"] = render(tok, r["events"], True, sp)
            recs.append(rec)
    summ = summarize(recs)
    summ["batch_size"] = batch_size
    return summ, recs


def summarize(recs):
    n = max(1, len(recs))
    reads = [r["readability"] for r in recs if r["readability"] is not None]
    s = dict(n=len(recs), acc=sum(r["correct"] for r in recs) / n,
             text_tokens=sum(r["n_text"] for r in recs) / n,
             latent_steps=sum(r["n_latent"] for r in recs) / n,
             steps=sum(r["steps"] for r in recs) / n,
             sec_per_sample=sum(r["seconds"] for r in recs) / n,
             latent_frac=sum(r["n_latent"] for r in recs) / max(1, sum(r["steps"] for r in recs)),
             readability=(sum(reads) / len(reads)) if reads else None)
    depths = sorted({r["depth"] for r in recs if r.get("depth") is not None})
    if depths:
        s["acc_by_depth"] = {str(d): sum(r["correct"] for r in recs if r["depth"] == d) /
                             max(1, sum(1 for r in recs if r["depth"] == d)) for d in depths}
    return s