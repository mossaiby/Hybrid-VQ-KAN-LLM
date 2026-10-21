# PLAN — Hybrid latent/text reasoning learned in pretraining, on a bounded ReLU-KAN + VQ backbone

Working title: **"Thinking Between Tokens from Scratch: Hybrid Latent Pretraining with Bounded KAN–VQ Transformers"**

## 1. Goal and central claims

HybridAgent (Zou et al., arXiv:2610.18861) converts a *pretrained* Qwen3 model into an agent that switches between text and latent reasoning. It uses three post-training stages: format SFT, HybridLink activation plus self-conditioned SFT, and HybridRL. We ask whether the same hybrid mechanism can be learned **during pretraining from scratch**, with **no post-training at all**, on a small, inspectable architecture. We test three claims:

- **C1 (feasibility).** Pretraining with hybrid targets yields a model that enters and exits latent mode on its own and answers correctly through latent spans. Accuracy should be no worse than a text-CoT model trained on the same tokens.
- **C2 (architecture).** A bounded ReLU-KAN FFN combined with a unit-sphere VQ codebook is a better substrate for latent reasoning than a vanilla Transformer. The sphere codebook makes the input space, output head and latent readout one geometry, so readout and exit become an exact VQ lookup.
- **C3 (efficiency/inspectability).** The latent mode keeps the paper's interpretability property (readable nearest-token readout) while cutting text tokens. Real step savings come from latent compression (k tokens per latent step), because there is no RL stage to learn shortening.

## 2. Mapping from the paper to this project

| Paper (HybridAgent) | This project | Why |
|---|---|---|
| Qwen3-Base 4–14B, pretrained | From-scratch decoder, ~18M (laptop) / ~194M (24 GB) | User requirement |
| HybridLink R(h) = h + W₂·GELU(W₁h) | Same (`link_type: mlp`), plus a bounded-KAN link (`kan`, default) and identity (Coconut-style) | Paper's Table 6 ablation and our architecture claim |
| `<latent>` entry token, exit when nearest token of R(h) is `</latent>` (Eq. 5) | Identical; the nearest-token lookup is the VQ codebook lookup | Unifies VQ with the exit rule |
| Data: LLM judges + entropy decide which steps go latent | Heuristic: middle steps go latent with probability p, first and last steps stay text, short spans stay text. Phase 2 option: entropy from our own text baseline | No external LLMs; reproducible |
| Format SFT → link activation → self-conditioned SFT | **One pretraining run with a curriculum.** Text only first, then latent probability ramps up, then the self-conditioning probability ramps up | "No post-training" |
| HybridRL (GRPO, exact replay) | Not used | User requirement; listed as future work |
| Tool calls (Python, search) | Not used | Infeasible to learn from scratch at this scale; future work |
| KV cache as working memory | Same (KV cache in `generate`) | — |
| One latent step per hidden token | `compression: k` (k = 1 matches the paper; k > 1 is our extension) | Only source of step savings without RL |

## 3. Architecture (implemented in `hybrid/`)

**Backbone.** Pre-norm decoder with RMSNorm, RoPE, QK-norm, causal SDPA attention, no dropout, and a weight-tied codebook.

**Bounded ReLU-KAN FFN** (`hybrid/kan.py`):

- Each layer computes u = tanh(a ⊙ x), then y = W_base·u + Σ_g W_g·B_g(u).
- The basis is B_g(u) = ReLU(1 − ((u − c_g)/w)²)². This is algebraically identical to the ReLU-KAN basis [ReLU(e−x)·ReLU(x−s)]²·16/(e−s)⁴ but needs one ReLU instead of two.
- Since |u| < 1 and 0 ≤ B ≤ 1, every output satisfies |y_o| ≤ |b_o| + Σ|W_o·|, which is bounded independently of the input. The layer is also Lipschitz.
- Base and spline weights are packed into a single GEMM.
- This replaces the quadratically growing ReLU(x − g)² of the earlier script.

**VQ sphere codebook** (`model.sphere: true`):

- Token embeddings are L2-normalised and rescaled by a learned scalar.
- The output head is cosine similarity × a learned temperature (clamped ≤ 100).
- Latent targets live on the same sphere.
- The "VQ loss" is the HybridLink regression (cosine + relative MSE, paper Eq. 7) toward the *ground-truth* next embedding. The earlier script instead pulled the codebook toward whatever token was currently nearest, which reinforces wrong predictions.
- `latent_quantize` (eval-time) feeds the nearest code instead of the continuous vector, as a discrete-latent probe.

**HybridLink.** Residual, with the last layer zero-initialised so that R starts as the identity.

## 4. Training objective (`hybrid/model.py::compute_losses`)

Batches mix three kinds of rows:

- **General text rows** (FineWeb-Edu): CE loss.
- **Reasoning rows** (OpenMathInstruct-2 and synthetic chains): built as `<bos> prompt step₁ <latent> SLOT… </latent> … step_n <eos>`.

Losses on reasoning rows:

- CE on every text position, including predicting `<latent>` (the entry decision) and `</latent>` from the last latent state (the exit, backbone side).
- HybridLink regression at every latent-source position toward the next slot's target. At the exit position the target is Emb(`</latent>`), with weight × 2.

**Self-conditioning** (paper Fig. 4): with probability p_sc, a no-grad pass computes R(h) one step ahead. Those predictions replace the teacher-forced latent inputs, and the graded pass runs on them. `iters > 1` repeats the no-grad pass to move closer to free-running.

**Curriculum** (fractions of training):

| Phase | Range | p_latent | p_sc | Purpose |
|---|---|---|---|---|
| A | 0 – 15 % | 0 | 0 | Learn language and text CoT; embeddings become meaningful targets |
| B | 15 – 60 % | 0 → 0.7 | 0 | Learn latent format and link (≈ paper's activation stages) |
| C | 45 – 80 % | ≈0.7 | 0 → 0.5 | Self-conditioning (≈ paper's self-conditioned HybridSFT) |
| D | 80 – 100 % | 0.7 | 0.5 | Consolidation under the cosine LR tail |

In addition, 20 % of reasoning samples always stay fully textual, so text-mode CoT remains available as a fair within-model comparison.

## 5. Data

| Source | Laptop | 24 GB | Role |
|---|---|---|---|
| FineWeb-Edu `sample-10BT` | 150M tok | 1.8B tok | Language |
| OpenMathInstruct-2 (CC-BY-4.0) | `train_1M`, GSM8K-derived only, ≤600k samples | `train` (7M), all sources | Natural-language CoT |
| Synthetic variable chains (generated) | 400k, depth 2–8 | 2M, depth 2–10 | Controlled reasoning depth; OOD eval at depth 9–12/16 |
| GSM8K test (eval only) | 1319 | 1319 | External benchmark |

Mixture weights (general / openmath / synth) are 0.4 / 0.4 / 0.2 on the laptop and 0.45 / 0.45 / 0.1 at 24 GB.

The synthetic task matters for two reasons. A ~18M-parameter model trained from scratch will score near the floor on GSM8K, while it can learn variable chains. Chains also give a *depth* axis, which is exactly what latent recurrence should help with (cf. Coconut's ProsQA).

Contamination: OpenMathInstruct-2 augments the GSM8K *train* set only. Still, run its released contamination checks against GSM8K test before reporting, and report GSM8K as a secondary benchmark.

## 6. Experimental design

**Main 2 × 2 factorial.** All arms use the same data, tokenizer, seq len, optimizer, schedule, seed and token budget:

| | text-only CoT | hybrid latent pretraining |
|---|---|---|
| **Vanilla** (SwiGLU, param-matched; dot-product head) | `vanilla_text` (B2) | `vanilla_hybrid` (B3, = paper's recipe from scratch) |
| **KAN-VQ** (bounded ReLU-KAN, sphere codebook) | `kanvq_text` (B1) | `kanvq_hybrid` (**ours**) |

Fairness rules (implemented):

- The SwiGLU hidden size is solved so the FFN parameters equal the KAN FFN's (`swiglu_d_ff: auto`). At the laptop scale this is 1792 vs KAN d_ff 384 × 7 basis features, giving 1.38M params per FFN for both.
- The token budget is identical across arms: 20 × the parameter count of the reference KAN-hybrid model at that scale.
- Text baselines see exactly the same reasoning samples; only the latent conversion differs.
- Every hybrid model can also be evaluated with latent entry blocked ("text mode").

Known residual mismatch: the KAN link has ~0.9M params vs 0.13M for the MLP link at laptop scale (~5 % of the model). Report this; ablation A2 removes it.

**Ablations** (run with the KAN-VQ backbone):

- **A1** identity link (Coconut-style)
- **A2** MLP link (paper)
- **A3** no self-conditioning
- **A4** compression k = 4 (also sweep k ∈ {1, 2, 4, 8} at laptop scale)
- **A5** kan_grid ∈ {4, 6, 8}
- **A6** p_latent end ∈ {0.3, 0.5, 0.7, 0.9}
- **A7** sphere off with KAN on (separates "KAN" from "VQ")

**Seeds.** 3 seeds for the four main arms at laptop scale; at 24 GB, 1 seed for main arms plus bootstrap CIs over eval items.

## 7. Metrics and analyses (mirroring the paper's sections)

1. **Accuracy.** Synthetic accuracy by depth (in- and out-of-distribution), OpenMath held-out accuracy, and GSM8K accuracy. Report each in hybrid and text modes.
2. **Efficiency.**
   - Text tokens, latent steps and total steps per problem
   - Wall-clock ms per sample
   - FLOPs per problem (steps × 2N)
3. **Inspectability.**
   - Readability: mean max-cosine to the vocabulary, paper Eq. 19, with the Gaussian-noise reference τ_noise.
   - Context specificity (Eq. 20).
   - Example readouts (paper Fig. 2).
4. **Causal reliance** (paper Table 5; already in `eval.py`):
   - noise σ = 0.5 / 2.0
   - one-step-only latent spans
   - quantized latents
   - blocked latent entry

   Still to add: cross-problem and mean-latent replacement, and within-span permutation.
5. **Training dynamics.** CE, link loss, exit accuracy, false-exit rate and readability vs tokens, compared across arms with `plot_runs.py`.
6. **Stability of bounded KAN.** Gradient-norm statistics, activation maxima, and loss spikes vs vanilla, plus divergence counts at high LR (LR sweep ×{0.5, 1, 2, 4}). A cheap but convincing figure for C2.
7. **Monitorability-lite** (paper Table 7, optional). A judge model sees only visible text. Lower priority because it needs an external LLM.

## 8. Compute plan

| Scale | Params | Tokens (20×) | Hardware | Estimate per arm |
|---|---|---|---|---|
| Laptop pilot | ~18.2M (14.0M non-emb) | ~365M | RTX 3050 Ti Laptop 4 GB | ~3–8 h (measure tok/s in first 100 steps) |
| Main | ~194M (169M non-emb) | ~3.9B | 24 GB card (3090/4090) | ~6·N·D ≈ 4.5·10¹⁸ FLOP → roughly 1 day (4090) to 2–3 days (3090) |

Self-conditioning adds a no-grad forward on up to ~50 % of late batches, about +10 % overall. Pilot runs should use `--set training.total_tokens=50000000` first.

Total 24 GB budget: 4 main arms + 3–4 ablations ≈ 8–12 GPU-days. Prioritise `kanvq_hybrid`, `vanilla_text`, `kanvq_text`, then `vanilla_hybrid`.

## 9. Milestones

- [x] **M0 — Scaffold.** Config system, data prep, model, losses, curriculum, generation with KV cache, eval with interventions, baselines, plots.
- [ ] **M1 — Smoke test** (laptop, ~1 h):
  - `prepare_data` with small budgets
  - 5M-token runs of all four arms to check finite losses, link cos > 0.5, and that exit accuracy rises
  - `generate.py` shows latent spans that close
- [ ] **M2 — Laptop study.** Full Chinchilla runs for 4 arms × 3 seeds, plus the compression and grid sweeps. Choose the final hyperparameters here, *before* scaling.
- [ ] **M3 — Write-up v0.** Methods, laptop results, and figures (depth curves, efficiency, readouts, interventions, stability).
- [ ] **M4 — 24 GB runs.** 4 main arms first, then ablations as budget allows.
- [ ] **M5 — Full analysis.** Bootstrap CIs, contamination check, readability vs noise reference, failure analysis.
- [ ] **M6 — Release and submit.** Code, configs, checkpoints and eval outputs (TMLR expects reproducibility).

## 10. Risks and mitigations

- **Latent targets are meaningless early** (random embeddings). Mitigated by curriculum phase A. If the link never aligns, lengthen phase A or freeze nothing but lower `lambda_link` early.
- **The model never exits latent mode** (exit accuracy stays low). Options:
  - `exit_weight` ↑
  - `exit_rule: either` (head or link)
  - `max_latent_steps` cap
  - report the false-exit rate
- **Exposure bias in long spans.** Mitigated by self-conditioning; `iters: 2–3` if free-running quality lags teacher-forced validation (compare the `val_hyb_tf_*` vs `val_hyb_sc_*` logs).
- **Floor effects on GSM8K** at small scale. The synthetic depth task is the primary controlled benchmark, and GSM8K is reported honestly as secondary.
- **No step savings at k = 1.** Expected, since the paper's savings come from RL. Make the compression sweep a headline result, and report text-token savings separately from step savings.
- **KAN cost.** The (G+1)× feature map costs memory and bandwidth; gradient checkpointing is on for the laptop config. Report tokens/s for every arm so that efficiency claims are honest.
- **Hybrid arms see fewer "text tokens" of reasoning** under the same position budget. Report both budgets, and note that latent positions also carry supervision.

## 10b. Design iteration v2 (after the first laptop pilot)

The first four laptop runs (v1) showed:

- **Backbone gap.** The KAN-VQ backbone trails the vanilla one by 0.58 nats on general text and 0.33 on OpenMath at matched parameters and tokens.
- **Hybrid cost.** Hybrid training is nearly free for KAN-VQ (≤ 0.02 nats) but costly for vanilla (0.13–0.20 nats).
- **Accuracy.** Hybrid mode trails text mode by 2–3 points in both backbones (quick eval, not yet significance-tested).

Two causes are attributable to the implementation rather than to the ideas being tested, so v2 fixes them.

1. **Cosine head (KAN-VQ arms).** The learned logit scale saturated at its hard cap of 100 at about 70 % of training. A cosine head also cannot encode token frequency.
   - *Fix:* a per-token logit bias (`model.logit_bias`) and a configurable cap (`model.logit_scale_max = 1000`).
   - The latent readout and the exit rule stay pure cosine (paper Eq. 5).
2. **Link loss scale (vanilla hybrid).** The relative-error term divides by ‖Emb(y)‖², which is tiny in a non-sphere model. When latent training began, link loss reached ~61 and the gradient norm 161. The median gradient norm then stayed at 2–10 against a clip of 1.0, effectively cutting the learning rate 3–10×.
   - *Fix:* a link warm-up (`hybrid.link_detach_until = 0.3`). Until 30 % of training, the link loss trains only HybridLink, not the backbone. This is the pretraining analogue of the paper's frozen-backbone link activation, and it is applied to both hybrid arms.

The token budget is unchanged (20 × the parameters of the v1 reference arm), so `vanilla_text` (unaffected by either fix) is reused.

**v2 runs:**
- `kanvq_hybrid_v2`
- `kanvq_text_v2`
- `vanilla_hybrid_v2`
- two separation runs with text-only models, completing {KAN, SwiGLU} × {sphere head, plain head}:
  - `sep_kan_plainhead_text`
  - `sep_swiglu_sphere_text`

**Reporting.** The paper reports v1 → v2 honestly as a design iteration. The 24 GB study uses the frozen v2 design.

## 11. Target venues (no article-processing charge)

Verify current policies on each site before submitting.

1. **TMLR (Transactions on Machine Learning Research).**
   - No fees for authors.
   - Open review on OpenReview.
   - Judges technical correctness and claims supported by evidence rather than "significance" — a good fit for a careful small-scale study.
   - Accepts conference-length papers with fast turnaround. **Primary recommendation.**
2. **TACL.** No author or submission charges. NLP-focused; accepted papers can be presented at ACL conferences. Fits if framed around language-model reasoning.
3. **JMLR.** Diamond open access with no fees. Expects a more substantial or long-form contribution — better for a later extended version.
4. **JAIR.** Diamond open access with no fees. Broad AI scope.
5. **Fallback.** Subscription journals (e.g. Neural Networks, Neurocomputing) charge nothing if you choose the non-open-access route. Check each journal's policy.

## 12. Open questions for you

1. **Headline compression.** Should compression (k > 1) be part of the headline method, or stay an ablation with the paper-faithful k = 1 as the main arm?
2. **Tools at 24 GB.** Do you want tool use (e.g. a Python calculator, as in the paper) at the 24 GB scale, or should the paper stay tool-free?
3. **Compute budget.** How many GPU-days can the 24 GB card realistically give (this decides seeds and ablations)?
4. **Annotator models.** Are external LLM judges acceptable for a Phase-2 annotation and monitorability study, or must everything be self-contained?
5. **Target card.** Is it a 3090 or a 4090 (affects `compile`, batch size and the time estimate)?
