# hybrid-kanvq

This project pretrains small language models from scratch that reason partly in **latent space** and partly in **text**, using the HybridAgent mechanism (HybridLink, `<latent>`/`</latent>` anchors, self-conditioning, KV working memory). The backbone is a **bounded ReLU-KAN** FFN with a **VQ unit-sphere codebook**. There is no post-training: the hybrid behaviour is learned during pretraining through a curriculum. See `PLAN.md` for the research plan.

> Status: scaffold. The code is complete but has **not been run yet**; it was written without GPU access. Do the smoke test below first.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA build
pip install -r requirements.txt
huggingface-cli login     # optional; avoids rate limits when streaming datasets
```

## Quick start (laptop, RTX 3050 Ti 4 GB)

```bash
# 1. data: tokenizer + FineWeb-Edu + OpenMathInstruct-2 (GSM8K-derived) + synthetic chains + GSM8K test
python prepare_data.py --config config.json

# 2. smoke test (~5M tokens, a few minutes)
python train.py --config config.json --set run.name=smoke --set training.total_tokens=5000000 \
                --set training.eval_interval=50 --set training.log_interval=10

# 3. full Chinchilla run (20 tokens/param) of the main model and the baselines
python train.py --config config.json                    # KAN-VQ hybrid (ours)
python train.py --config configs/kanvq_text.json        # B1: same backbone, text CoT only
python train.py --config configs/vanilla_text.json      # B2: vanilla Transformer, text CoT
python train.py --config configs/vanilla_hybrid.json    # B3: paper recipe on vanilla backbone

# 4. evaluate, generate, compare
python eval.py --run runs/kanvq_hybrid --sets synth openmath gsm8k --limit 500 --interventions all
python eval.py --run runs/kanvq_hybrid --sets synth --limit 200 --batch_size 1   # latency (ms/sample) numbers
python generate.py --run runs/kanvq_hybrid --question "Let a = 4. Let b = a * 3. Let c = b + a. What is c?"
python plot_runs.py runs/kanvq_hybrid runs/kanvq_text runs/vanilla_text runs/vanilla_hybrid
```

For a quick data-prep test, shrink the budgets, e.g. `--set data.general.train_tokens=5000000 --set data.openmath.max_train_samples=20000 --set data.synth.train_samples=20000 --set data.dir=data/tiny`. Pass the same `data.dir` to `train.py`.

## 24 GB card (Chinchilla-scale study)

```bash
python prepare_data.py --config configs/24gb/kanvq_hybrid.json      # separate data dir, 32k vocab
python train.py --config configs/24gb/kanvq_hybrid.json
python train.py --config configs/24gb/vanilla_text.json             # etc. for every file in configs/24gb/
```

This scale is about 194M params and about 3.9B tokens. Tune `training.micro_batch` and `grad_accum` to your card. Keep their product times `max_seq_len` fixed across arms.

## Configuration

Every run is fully described by a JSON file. Files can `inherit` from others (a string or a list, deep-merged in order). Any key can be overridden from the CLI with `--set a.b.c=value`, where the value is parsed as JSON. The resolved config is saved to `runs/<name>/config.resolved.json` and inside every checkpoint.

| file | arm |
|---|---|
| `config.json` | **KAN-VQ hybrid** (main model, laptop size) |
| `configs/kanvq_text.json` | B1 – same backbone, text-only CoT |
| `configs/vanilla_text.json` | B2 – SwiGLU (param-matched), dot-product head, text-only |
| `configs/vanilla_hybrid.json` | B3 – paper's MLP HybridLink on a vanilla backbone |
| `configs/ablation_*.json` | identity link, MLP link, no self-conditioning, compression 4 |
| `configs/scale_24gb.json` | partial override for 24 GB; combined via `configs/24gb/*.json` |

Key knobs:

**`model.mlp_type`** is `kan` or `swiglu`. With `swiglu_d_ff: auto`, the SwiGLU hidden size is solved so its FFN parameter count equals the KAN FFN's.

**`model.sphere`** switches the VQ unit-sphere codebook on or off. When on, the cosine head, the latent targets, and the nearest-code readout all use the same sphere.

**`model.link_type`** is `kan`, `mlp` (paper Eq. 4), or `identity` (Coconut-style).

**`hybrid.compression`** sets k, the number of hidden tokens per latent step. k = 1 matches the paper.

**`hybrid.latent_prob` and `hybrid.self_cond`** are ramps over training progress (`start`, `end`, `ramp_start`, `ramp_end`).

**`training.total_tokens`** is either `"chinchilla"` or an integer. The Chinchilla budget is 20 × the params of the reference KAN-hybrid arm, so every arm gets the same number of tokens.

**`inference.exit_rule`** is one of:
- `link`: exit when the nearest token of R(h) is `</latent>` (paper Eq. 5).
- `head`: exit when the head's argmax is `</latent>`.
- `either`: exit on whichever fires first.

## Layout

```
hybrid/kan.py         bounded ReLU-KAN layer, KAN FFN, SwiGLU, parameter matching
hybrid/model.py       backbone, VQ sphere codebook, HybridLink, losses + self-conditioning
hybrid/data.py        step segmentation, synthetic chains, binary storage, hybrid packing, batches
hybrid/generation.py  hybrid decoding with KV cache, latent readout rendering, answer checking
hybrid/config.py      config inheritance / overrides / curriculum ramps
prepare_data.py       build data/<scale>/ (tokenizer, .bin token streams, eval jsonl)
train.py              pretraining loop (bf16 autocast, grad accumulation, checkpoints, val, quick eval)
eval.py               accuracy / efficiency / readability / causal interventions
generate.py           one-shot or interactive generation, shows ⟨latent readouts⟩
plot_runs.py          compare runs
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