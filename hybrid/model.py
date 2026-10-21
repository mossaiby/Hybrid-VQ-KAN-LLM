"""HybridLM: a decoder-only Transformer that can think in text or in latent mode.

One class covers every arm of the experimental design through config flags:

  model.mlp_type   "kan" (bounded ReLU-KAN FFN) | "swiglu" (vanilla FFN)
  model.sphere     true  -> VQ-style unit-sphere codebook: token embeddings, the output head and
                            the latent readout all use the same L2-normalised table (cosine logits)
                   false -> ordinary tied embeddings + dot-product head
  model.link_type  "kan" | "mlp" (paper's Eq. 4) | "identity" (Coconut-style h_t fed back as is)
  hybrid.enabled   false -> pure text-CoT training (baseline)

Latent mode (paper Sec. 3.1), adapted to pretraining:
  * text positions are trained with cross-entropy (CE);
  * at a latent "source" position t (input is <latent> or a latent slot) the HybridLink output
    R(h_t) is regressed onto the NEXT input vector: the target embedding of the next slot, or
    Emb(</latent>) at the exit (weighted more, App. B.2);
  * exit at inference: nearest codebook entry of R(h_t) is </latent> (Eq. 5) - i.e. a VQ lookup.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .kan import BoundedReLUKANLinear, KANMLP, SwiGLU, matched_swiglu_hidden


# --------------------------------------------------------------------------- building blocks
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight.to(x.dtype)


def rope_tables(head_dim, length, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(length).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d, n_heads, qk_norm=True, dropout=0.0, out_init_scale=1.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.hd, self.dropout = n_heads, d // n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02 * out_init_scale)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin, cache=None, pos0=0, attn_mask=None):
        """attn_mask: optional bool (B, 1, T, Tk), True = may attend. Used by batched generation with
        left-padded prompts; training uses the plain causal path."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        c, s = cos[pos0:pos0 + T], sin[pos0:pos0 + T]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        p = self.dropout if self.training else 0.0
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        elif Tk == T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:  # chunk appended to a non-empty cache
            i = torch.arange(T, device=x.device)[:, None]
            j = torch.arange(Tk, device=x.device)[None, :]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=j <= (Tk - T) + i, dropout_p=p)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, m, n_layers):
        super().__init__()
        d = m["d_model"]
        out_scale = 1.0 / math.sqrt(2 * n_layers)
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, m["n_heads"], m.get("qk_norm", True), m.get("dropout", 0.0), out_scale)
        if m["mlp_type"] == "kan":
            self.mlp = KANMLP(d, m["d_ff"], m["kan_grid"], m.get("kan_overlap", 1.5), out_scale)
        elif m["mlp_type"] == "swiglu":
            self.mlp = SwiGLU(d, resolve_swiglu_hidden(m), out_scale)
        else:
            raise ValueError(m["mlp_type"])
        self.drop = nn.Dropout(m.get("dropout", 0.0))

    def forward(self, x, cos, sin, cache=None, pos0=0, attn_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache, pos0, attn_mask))
        return x + self.drop(self.mlp(self.norm2(x)))


def resolve_swiglu_hidden(m):
    h = m.get("swiglu_d_ff", "auto")
    if h == "auto":  # parameter-match the KAN FFN of the same d_model/d_ff/grid
        return matched_swiglu_hidden(m["d_model"], m["d_ff"], m["kan_grid"])
    return int(h)


class HybridLink(nn.Module):
    """R(h) = h + f(h), f zero-initialised so R starts as the identity (pure residual)."""

    def __init__(self, m):
        super().__init__()
        d, hdim, kind = m["d_model"], m.get("link_hidden", m["d_model"]), m["link_type"]
        self.kind = kind
        if kind == "mlp":  # paper Eq. 4: h + W2 GELU(W1 h)
            self.w1, self.w2 = nn.Linear(d, hdim), nn.Linear(hdim, d)
            nn.init.normal_(self.w1.weight, std=0.02)
            nn.init.zeros_(self.w2.weight)
            nn.init.zeros_(self.w2.bias)
        elif kind == "kan":
            self.k1 = BoundedReLUKANLinear(d, hdim, m["kan_grid"], m.get("kan_overlap", 1.5))
            self.k2 = BoundedReLUKANLinear(hdim, d, m["kan_grid"], m.get("kan_overlap", 1.5), zero_init=True)
        elif kind != "identity":
            raise ValueError(kind)

    def forward(self, h):
        if self.kind == "identity":
            return h
        if self.kind == "mlp":
            return h + self.w2(F.gelu(self.w1(h)))
        return h + self.k2(self.k1(h))


# --------------------------------------------------------------------------- model
class HybridLM(nn.Module):
    def __init__(self, m, vocab_size, specials):
        super().__init__()
        self.m, self.sp, self.vocab_size = m, specials, vocab_size
        d = m["d_model"]
        self.d, self.sphere = d, bool(m.get("sphere", True))
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=1.0 if self.sphere else 0.02)
        if self.sphere:
            # unit-sphere codebook; input embeddings are rescaled so the residual stream starts at O(1)/dim
            self.emb_log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(d))))
            self.logit_log_scale = nn.Parameter(torch.tensor(math.log(m.get("logit_scale_init", 16.0))))
        self.blocks = nn.ModuleList([Block(m, m["n_layers"]) for _ in range(m["n_layers"])])
        self.norm_f = RMSNorm(d)
        self.link_mod = HybridLink(m)
        self.grad_ckpt = bool(m.get("grad_checkpoint", False))
        cos, sin = rope_tables(d // m["n_heads"], 4 * m["max_seq_len"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ---- codebook / embedding space -------------------------------------------------------
    def codebook(self):
        """Target space for latent regression and readout (unit sphere if `sphere`)."""
        w = self.embed.weight
        return F.normalize(w, dim=-1) if self.sphere else w

    def embed_ids(self, ids):
        w = self.embed(ids)
        return F.normalize(w, dim=-1) * self.emb_log_scale.exp() if self.sphere else w

    def slot_input(self, vec):
        """Map a target-space vector (or a HybridLink output) to an input embedding."""
        return F.normalize(vec, dim=-1) * self.emb_log_scale.exp() if self.sphere else vec

    def target_vecs(self, groups):
        """groups: (..., k) token ids, -1 padded. Each latent slot stands for up to k tokens
        (k = hybrid.compression; k = 1 reproduces the paper). Target = renormalised sum on the sphere,
        mean embedding otherwise."""
        cb = self.codebook()
        mask = groups >= 0
        v = cb[groups.clamp_min(0)] * mask.unsqueeze(-1).to(cb.dtype)
        s = v.sum(-2)
        if self.sphere:
            return F.normalize(s, dim=-1)
        return s / mask.sum(-1, keepdim=True).clamp_min(1).to(s.dtype)

    # ---- network ------------------------------------------------------------------------------
    def new_caches(self):
        return [dict(k=None, v=None) for _ in self.blocks]

    def backbone(self, x, caches=None, pos0=0, attn_mask=None):
        use_ckpt = self.grad_ckpt and self.training and caches is None and torch.is_grad_enabled()
        for i, blk in enumerate(self.blocks):
            if use_ckpt:
                x = checkpoint(blk, x, self.rope_cos, self.rope_sin, None, 0, None, use_reentrant=False)
            else:
                x = blk(x, self.rope_cos, self.rope_sin, None if caches is None else caches[i], pos0, attn_mask)
        return self.norm_f(x)

    def logit_scale(self):
        return self.logit_log_scale.exp().clamp(max=100.0)

    def logits(self, h):
        if self.sphere:  # cosine-similarity head against the VQ codebook
            return self.logit_scale() * (F.normalize(h, dim=-1) @ self.codebook().t().to(h.dtype))
        return h @ self.embed.weight.t().to(h.dtype)

    def link(self, h):
        return self.link_mod(h)

    @torch.no_grad()
    def readout(self, z):
        """Nearest-token (VQ) readout of latent vectors, paper Eq. 5 / Eq. 19. Returns (ids, cos)."""
        cos = F.normalize(z.float(), dim=-1) @ F.normalize(self.embed.weight.float(), dim=-1).t()
        cos[..., [self.sp.pad, self.sp.slot, self.sp.bos, self.sp.unk]] = -2.0
        val, idx = cos.max(-1)
        return idx, val

    # ---- bookkeeping ------------------------------------------------------------------------
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def param_groups(self, weight_decay):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_emb = name.startswith("embed.")
            if p.dim() >= 2 and not (is_emb and self.sphere):
                decay.append(p)
            else:  # norms, scales, biases, and the sphere codebook (direction only)
                no_decay.append(p)
        return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


# --------------------------------------------------------------------------- training objective
def build_inputs(model, ids_in, grp_in):
    """Teacher-forced input embeddings: tokens via the embedding table, latent slots via their
    ground-truth target vectors (paper: Emb(y_lat))."""
    sp = model.sp
    is_slot = ids_in == sp.slot
    x = model.embed_ids(ids_in.masked_fill(is_slot, sp.pad))
    if is_slot.any():
        tv = model.target_vecs(grp_in[is_slot])
        x = x.clone()
        x[is_slot] = model.slot_input(tv).to(x.dtype)
    return x


def compute_losses(model, ids, grp, hcfg, self_cond_iters=0, with_metrics=True):
    """ids: (B, S+1) item ids (slots = sp.slot), grp: (B, S+1, k) slot token groups (-1 padded).

    Returns (total_loss, metrics). With self_cond_iters > 0 the latent inputs are replaced by the
    model's own one-step-ahead HybridLink predictions before the graded pass (paper Sec. 4.1, Fig. 4);
    each extra iteration re-runs the no-grad pass on the updated inputs (closer to free running)."""
    sp = model.sp
    ids_in, ids_next = ids[:, :-1], ids[:, 1:]
    grp_next = grp[:, 1:]
    x = build_inputs(model, ids_in, grp[:, :-1])

    is_slot_in = ids_in == sp.slot
    next_slot = ids_next == sp.slot
    next_close = ids_next == sp.lat_close
    src = (ids_in == sp.lat_open) | is_slot_in  # positions in latent mode

    # ---- self-conditioning (no-grad passes) ----------------------------------------------
    if self_cond_iters > 0:
        feed = (src & next_slot)[:, :-1]  # position t feeds input t+1 inside the row
        if feed.any():
            for _ in range(self_cond_iters):
                with torch.no_grad():
                    h1 = model.backbone(x.detach())
                    z = model.slot_input(model.link(h1[:, :-1][feed]))
                x = x.clone()
                x[:, 1:][feed] = z.to(x.dtype)
            # The no-grad pass filled autocast's weight-cast cache with bf16 copies that have no grad
            # history. Clear it so the graded pass re-casts with history; otherwise weight gradients are
            # lost and checkpoint recomputation no longer matches the forward pass.
            torch.clear_autocast_cache()

    h = model.backbone(x)

    # ---- text positions: cross-entropy ------------------------------------------------------
    lbl = ids_next.masked_fill(next_slot | (ids_next == sp.pad), -100)
    m = lbl != -100
    ce = F.cross_entropy(model.logits(h[m]).float(), lbl[m]) if m.any() else h.sum() * 0.0

    # ---- latent positions: HybridLink regression (paper Eq. 7, on the self-conditioned pass) ---
    lsrc = src & (next_slot | next_close)
    metrics = {}
    if lsrc.any():
        z = model.link(h[lsrc]).float()
        tgt = model.target_vecs(grp_next[lsrc]).float()
        close_vec = model.codebook()[sp.lat_close].float()
        is_exit = next_close[lsrc]
        tgt = torch.where(is_exit.unsqueeze(-1), close_vec.expand_as(tgt), tgt).detach()
        zc = F.normalize(z, dim=-1) if model.sphere else z
        cos = F.cosine_similarity(zc, tgt, dim=-1)
        rel = (zc - tgt).pow(2).sum(-1) / tgt.pow(2).sum(-1).clamp_min(1e-6)
        w = torch.where(is_exit, torch.full_like(cos, hcfg.get("exit_weight", 2.0)), torch.ones_like(cos))
        per = hcfg.get("lambda_cos", 1.0) * (1 - cos) + hcfg.get("lambda_mse", 0.2) * rel
        link_loss = (w * per).sum() / w.sum()
        if with_metrics:
            with torch.no_grad():
                rid, rcos = model.readout(z)
                metrics["readability"] = rcos.mean()
                metrics["link_cos"] = cos.mean()
                if is_exit.any():
                    metrics["exit_acc"] = (rid[is_exit] == sp.lat_close).float().mean()
                if (~is_exit).any():
                    first = grp_next[lsrc][~is_exit][:, 0]
                    metrics["slot_tok_acc"] = (rid[~is_exit] == first).float().mean()
                    metrics["false_exit"] = (rid[~is_exit] == sp.lat_close).float().mean()
                metrics["n_latent"] = torch.tensor(float(lsrc.sum()), device=h.device)
    else:
        link_loss = h.sum() * 0.0

    total = ce + hcfg.get("lambda_link", 1.0) * link_loss
    metrics["ce"] = ce.detach()
    metrics["link"] = link_loss.detach()
    return total, metrics
