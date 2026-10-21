"""Bounded ReLU-KAN layers.

ReLU-KAN (Qiu et al., 2024) uses the basis  R(x) = [ReLU(e - x) * ReLU(x - s)]^2 * 16 / (e - s)^4.
With centre c = (s+e)/2 and half-width w = (e-s)/2 this is exactly

        B(x) = ReLU(1 - ((x - c) / w)^2)^2            (compact support, peak 1, C^1)

which needs one ReLU instead of two. The attached script used ReLU(x - g)^2, which grows
quadratically and is the main source of instability. Here every layer is bounded:

        u   = tanh(a * x)                             (a: learnable per-input scale)
        y_o = b_o + sum_i W_base[o,i] * u_i + sum_{i,g} W_spl[o,i,g] * B_g(u_i)

Because |u_i| < 1 and 0 <= B_g <= 1,   |y_o| <= |b_o| + sum_i (|W_base[o,i]| + sum_g |W_spl[o,i,g]|),
i.e. the output is bounded by the weights regardless of the input magnitude, and the layer is
Lipschitz. Base and spline weights are packed into ONE matrix of shape (out, in*(G+1)) so the whole
layer is a single GEMM over the feature map [u, B_1(u), ..., B_G(u)].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_dtype(x):
    """Under CUDA autocast, build the (in x G) basis directly in the low-precision dtype; otherwise the
    fp32 residual stream would make every basis intermediate fp32 and double activation memory."""
    if not x.is_cuda:
        return x.dtype
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # older torch
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return x.dtype
    try:
        return torch.get_autocast_dtype("cuda")
    except AttributeError:
        return torch.get_autocast_gpu_dtype()


class BoundedReLUKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid=6, overlap=1.5, bias=True,
                 out_init_scale=1.0, zero_init=False):
        super().__init__()
        assert grid >= 2
        self.in_features, self.out_features, self.grid = in_features, out_features, grid
        spacing = 2.0 / (grid - 1)
        self.inv_width = 1.0 / (overlap * spacing)  # half-width w = overlap * spacing
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, grid), persistent=False)
        self.in_scale = nn.Parameter(torch.ones(in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features * (grid + 1)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters(out_init_scale, zero_init)

    @torch.no_grad()
    def reset_parameters(self, out_init_scale=1.0, zero_init=False):
        if zero_init:
            self.weight.zero_()
            return
        w = self.weight.view(self.out_features, self.in_features, self.grid + 1)
        # tanh(x) of a unit-variance input has std ~0.63 -> scale base weights up slightly
        w[:, :, 0].normal_(0.0, out_init_scale * 1.5 / math.sqrt(self.in_features))
        w[:, :, 1:].normal_(0.0, out_init_scale * 0.5 / math.sqrt(self.in_features * self.grid))

    def basis(self, u):
        t = (u.unsqueeze(-1) - self.centers.to(u.dtype)) * self.inv_width
        return F.relu(1.0 - t * t).square()

    def forward(self, x):
        x = x.to(_compute_dtype(x))
        u = torch.tanh(x * self.in_scale.to(x.dtype))
        feats = torch.cat([u.unsqueeze(-1), self.basis(u)], dim=-1).flatten(-2)
        return F.linear(feats, self.weight.to(feats.dtype),
                        None if self.bias is None else self.bias.to(feats.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, grid={self.grid}"


def kan_linear_params(i, o, grid):
    return o * i * (grid + 1) + o + i  # weight + bias + in_scale


class KANMLP(nn.Module):
    """Transformer FFN made of two bounded KAN layers. No extra activation is needed: each
    layer is already nonlinear, and kan2's tanh squashes kan1's output."""

    def __init__(self, d_model, d_ff, grid=6, overlap=1.5, out_init_scale=1.0):
        super().__init__()
        self.kan1 = BoundedReLUKANLinear(d_model, d_ff, grid, overlap)
        self.kan2 = BoundedReLUKANLinear(d_ff, d_model, grid, overlap, out_init_scale=out_init_scale)

    def forward(self, x):
        return self.kan2(self.kan1(x))


def kan_mlp_params(d_model, d_ff, grid):
    return kan_linear_params(d_model, d_ff, grid) + kan_linear_params(d_ff, d_model, grid)


class SwiGLU(nn.Module):
    """Standard modern FFN used by the vanilla baseline."""

    def __init__(self, d_model, d_ff, out_init_scale=1.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        for lin in (self.w1, self.w3):
            nn.init.normal_(lin.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02 * out_init_scale)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def matched_swiglu_hidden(d_model, kan_d_ff, grid, multiple=32):
    """Hidden size that gives a SwiGLU FFN (3*d*h params) the same parameter count as the KAN FFN."""
    target = kan_mlp_params(d_model, kan_d_ff, grid)
    h = target / (3 * d_model)
    return max(multiple, int(round(h / multiple)) * multiple)
