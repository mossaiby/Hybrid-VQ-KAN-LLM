# Hybrid latent/text reasoning learned in pretraining

This project pretrains small language models from scratch that reason partly in **latent space** and partly in **text**, using the HybridAgent mechanism (HybridLink, `<latent>`/`</latent>` anchors, self-conditioning, KV working memory) without any post-training. The current main model (v3) is a standard pre-norm Transformer (SwiGLU FFN) with a **sphere head**: unit-sphere tied embeddings, cosine logits with a per-token bias, and the same sphere used for latent targets and for the nearest-token latent readout.

> Status: the v1/v2 study of a bounded ReLU-KAN FFN found it worse than SwiGLU on every measure. Its configs are archived in `configs/archive_kan/`, and the findings are in PLAN.md §10b–10c. v3 runs (main model, seeds, ablations) are listed below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA build
pip install -r requirements.txt
huggingface-cli login     # optional; avoids rate limits when streaming datasets
```

`torch.compile` is on by default. On Windows it needs Triton matching your PyTorch (for PyTorch 2.6: `pip install "triton-windows<3.3"`). Otherwise add `--set training.compile=false`.

## Quick start (laptop, RTX 3050 Ti 4 GB)

```bash
# 1. data: tokenizer + FineWeb-Edu + OpenMathInstruct-2 (GSM8K-derived) + synthetic chains + GSM8K test
python prepare_data.py --config config.json

# 2. smoke test (a few minutes)
python train.py --config config.json --set run.name=smoke --set training.total_tokens=3000000

# 3. main model and comparison arms (364.7M tokens each, ~4-4.5 h on the laptop)
python train.py --config config.json                        # sphere_hybrid (main)
python train.py --config configs/sphere_text.json           # same backbone, text CoT only
python train.py --config configs/vanilla_hybrid.json        # standard head + hybrid
python train.py --config configs/seeds/sphere_hybrid_s2.json   # seed replicates: configs/seeds/

# 4. evaluate, generate, compare
python eval.py --run runs/sphere_hybrid --sets synth gsm8k --limit 0 --interventions all
python eval.py --run runs/sphere_hybrid --sets synth --limit 200 --batch_size 1   # latency numbers
python generate.py --run runs/sphere_hybrid --question "Let a = 4. Let b = a * 3. Let c = b + a. What is c?"
python compare_runs.py --runs sphere_hybrid sphere_text vanilla_text --sets synth gsm8k --ref vanilla_text:text
python plot_runs.py runs/sphere_hybrid runs/sphere_text runs/vanilla_text
```

`train.py` refuses to start in a run folder that already holds a checkpoint, unless you pass `--resume`. This prevents appending to a finished run's logs.

For a quick data-prep test, shrink the budgets, e.g. `--set data.general.train_tokens=5000000 --set data.openmath.max_train_samples=20000 --set data.synth.train_samples=20000 --set data.dir=data/tiny`. Pass the same `data.dir` to `train.py`.

## Other hardware

**RTX 2080 Ti (11 GB).** `configs/2080ti/*.json` combine an arm with `configs/hw_2080ti.json`: micro-batch 16 × 2 accumulation steps (the same 16,384 tokens per step) and fp16 with loss scaling, since Turing has no fast bf16. `training.amp` (`auto` | `bf16` | `fp16` | `fp32`) overrides the precision. `auto` picks bf16 on Ampere or newer GPUs and fp16 otherwise.

**24 GB card.** `configs/24gb/*.json` combine an arm with `configs/scale_24gb.json` (d_model 768, 12 layers, 32k vocabulary, separate data directory):

```bash
python prepare_data.py --config configs/24gb/sphere_hybrid.json
python train.py --config configs/24gb/sphere_hybrid.json
```

Tune `training.micro_batch` and `grad_accum` to the card, keeping their product × `max_seq_len` fixed across arms.

## Configuration

Every run is fully described by a JSON file. Files can `inherit` from others (a string or a list, deep-merged in order). Any key can be overridden from the CLI with `--set a.b.c=value`, where the value is parsed as JSON. The resolved config is saved to `runs/<name>/config.resolved.json` and inside every checkpoint.

| file | arm |
|---|---|
| `configs/base.json` | shared settings for all laptop runs (not run directly) |
| `config.json` | **sphere_hybrid**: SwiGLU + sphere head + hybrid latent pretraining (main) |
| `configs/sphere_text.json` | same backbone, text-only CoT (seed-1337 run: `runs/sep_swiglu_sphere_text`) |
| `configs/vanilla_text.json` | standard tied dot-product head, text-only (seed-1337 run: `runs/vanilla_text`) |
| `configs/vanilla_hybrid.json` | standard head + hybrid (`vanilla_hybrid_v3`) |
| `configs/seeds/*.json` | seeds 2 and 3 of `sphere_hybrid`, `sphere_text`, `vanilla_text` |
| `configs/abl_*.json` | stronger self-conditioning, compression 4, no self-conditioning, identity link |
| `configs/archive_kan/*.json` | the v2 KAN-study runs, kept reproducible |
| `configs/24gb/`, `configs/2080ti/` | hardware variants of the current arms |

Key knobs:

**`model.mlp_type`** is `swiglu` or `kan` (archived). With `swiglu_d_ff: auto`, the SwiGLU hidden size is solved to match the parameter count of the v1 KAN FFN (1792 at laptop size), which keeps all runs comparable.

**`model.sphere`** switches the sphere head on or off. When on, the cosine head (with `logit_bias`), the latent targets and the nearest-token readout all use the same unit-sphere embedding table.

**`model.link_type`** is `kan`, `mlp` (paper Eq. 4), or `identity` (Coconut-style).

**`hybrid.compression`** sets k, the number of hidden tokens per latent step. k = 1 matches the paper.

**`hybrid.latent_prob` and `hybrid.self_cond`** are ramps over training progress (`start`, `end`, `ramp_start`, `ramp_end`).

**`training.total_tokens`** is either `"chinchilla"` or an integer. The Chinchilla budget is 20 × the parameter count of `training.budget_reference` (the v1 reference model), so every arm, old and new, gets the same 364.7M tokens.

**`training.clip_link_separately`** (default true) clips HybridLink gradients separately from the backbone, so large early link gradients cannot shrink every backbone update.

**`inference.exit_rule`** is one of:
- `link`: exit when the nearest token of R(h) is `</latent>` (paper Eq. 5).
- `head`: exit when the head's argmax is `</latent>`.
- `either`: exit on whichever fires first.

## Layout

```
hybrid/kan.py         SwiGLU, bounded ReLU-KAN layer (archived), parameter matching
hybrid/model.py       backbone, sphere head, HybridLink, losses + self-conditioning
hybrid/data.py        step segmentation, synthetic chains, binary storage, hybrid packing, batches
hybrid/generation.py  hybrid decoding with KV cache, latent readout rendering, answer checking
hybrid/config.py      config inheritance / overrides / curriculum ramps
prepare_data.py       build data/<scale>/ (tokenizer, .bin token streams, eval jsonl)
train.py              pretraining loop (bf16 autocast, grad accumulation, checkpoints, val, quick eval)
eval.py               accuracy / efficiency / readability / causal interventions
generate.py           one-shot or interactive generation, shows ⟨latent readouts⟩
plot_runs.py          compare runs
compare_runs.py       paired accuracy comparison (Wilson CIs, exact McNemar tests, by depth)
check_batching.py     verifies batched generation equals one-at-a-time generation
make_dev_set.py       held-out synthetic dev set for tuning inference settings
```

## Outputs

Each run writes the following to `runs/<name>/`:

- `log.csv`: training metrics, including CE, link loss, exit accuracy, false-exit rate, latent readability, and tokens/s.
- `val.jsonl`: per-source validation CE, teacher-forced vs self-conditioned link metrics, and quick accuracy in hybrid and text modes.
- `last.pt` and `final.pt`: checkpoints.
- `tokenizer.json`: copied in so the run is self-contained.
- `eval/`: written by `eval.py`.

## Memory and speed tips (4 GB)

- If you run out of memory, lower `training.micro_batch` to 2 and raise `grad_accum` to 16 so tokens per step are unchanged.
- If you have headroom, set `model.grad_checkpoint=false` for speed.
- `training.compile` is on by default (compiles the FFNs). On the RTX 3050 Ti laptop it raised throughput from ~15k to ~26k tok/s. On Windows it needs Triton matching your PyTorch (PyTorch 2.6: `pip install "triton-windows<3.3"`). Without Triton, set `--set training.compile=false`.
- KAN FFNs build a (G+1)× wider feature map, so lowering `model.kan_grid` reduces memory roughly linearly. Keep it identical across arms.

## Data and licenses

| dataset | used for | license |
|---|---|---|
| HuggingFaceFW/fineweb-edu | general text | ODC-By |
| nvidia/OpenMathInstruct-2 | chain-of-thought | CC-BY-4.0 |
| openai/gsm8k | evaluation | MIT |

The synthetic chain problems are generated locally.

## References

- J. Zou et al., *Hybrid Reasoning Agents*, arXiv:2610.18861 (2026).
- Q. Qiu et al., *ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU* (2024).
- Z. Liu et al., *KAN: Kolmogorov-Arnold Networks* (2024).
- A. van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE, 2017).
- S. Hao et al., *Training LLMs to Reason in a Continuous Latent Space* (Coconut, 2024).
- J. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022).
- S. Toshniwal et al., *OpenMathInstruct-2* (2024).
