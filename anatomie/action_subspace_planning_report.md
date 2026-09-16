# Action Subspace & Forward-Planning Diagnostics — Report

**Repo:** lesaude-dynamics · **Scope:** `papel/metodo.tex` (Dynamics / Inverse Dynamics / Planning)
**Artifacts:** [`demo/probe_action_subspace.ipynb`](probe_action_subspace.ipynb),
[`demo/dynamics_headroom_check.py`](dynamics_headroom_check.py)

## TL;DR

The original hypothesis — that longitudinal ECG pairs differ only in a small, action-proportional
subset of latent features — **does not hold**, in either an axis-aligned or a rotated-linear sense, for
any of the four pretrained encoders. That negative result predicted a specific downstream failure: CEM
planning scored by raw MSE against the forward dynamics model (`Dyn_φ`) carries essentially no signal
about which action occurred, and no amount of distance reweighting (diagonal or full-covariance
whitening) fixes it. Scoring candidates through the **inverse**-dynamics head instead recovers a large,
real discriminative signal, once a base-rate calibration bug in the raw score is corrected. A follow-up
check on the forward model found no evidence that `DynamicsPredictor` is under-capacity — both tested
checkpoints show real train/val generalization gaps, arguing against a pure capacity increase.

---

## 1. Hypothesis

> The ECG signal from longitudinal exam pairs differs only in a small subset of latent features, and
> the size of that subset is proportional to the norm of the pathology transition (action) vector
> $a_t = y_{t+1} - y_t$.

Tested against the encoder from each of the four pretrained checkpoints in `archive/`:

| Encoder | Checkpoint | Produced by |
|---|---|---|
| `dynamics` | `pretrain_val.ckpt` (later also tested: `pretrain_auroc.ckpt`) | `scripts/pretrain.py` |
| `inverse_dynamics` | `inverse.ckpt` | `scripts/inverse_pretrain.py` |
| `finetuned_inverse` | `finetune_inverse.ckpt` | `scripts/finetune.py` (init. from `inverse.ckpt`) |
| `supervised` | `supervised_0.ckpt` | `scripts/supervised.py` |

All experiments sample random pairs `(X_t, X_{t+1}, y_t, a_t)` from the **val** split (fold 18) via
`MIMICLanceDataset(mode="pair")`, so nothing evaluated was seen in training.

---

## 2. Experiment 1 — Axis-aligned sparsity

**Method.** For each pair, compute $\Delta h = h_{t+1}-h_t$ (z-scored per-dimension using pooled
$\{h_t,h_{t+1}\}$ batch statistics, since encoders trained with/without SIGReg have different natural
scales). Measure `dims_90`: the number of the 256 latent dimensions needed to explain 90% of
$\|\Delta h\|^2$, per pair, and correlate it against $\|a_t\|_0$.

**Results** (`BATCH_SIZE=1024`, val split):

| encoder | pearson(‖a‖₀, dims_90) | spearman(‖a‖₀, dims_90) | pearson(‖a‖₀, ‖Δh‖₂) | mean dims_90 / 256 |
|---|---:|---:|---:|---:|
| dynamics | 0.029 | 0.046 | 0.234 | 0.455 |
| inverse_dynamics | -0.041 | -0.059 | 0.202 | 0.448 |
| finetuned_inverse | 0.024 | 0.030 | 0.150 | 0.449 |
| supervised | 0.010 | 0.043 | 0.099 | 0.485 |

For every encoder, ~45–48% of the 256 dimensions are needed to explain 90% of the displacement energy
(not a small subset), and the subset size shows essentially no correlation with the action norm.
A concentration-curve check (cumulative share of mean $|\Delta h_i|$, dimensions sorted descending) sat
close to the diagonal (no-concentration reference) for all four encoders.

**Conclusion.** The axis-aligned form of the hypothesis is **rejected** for all four training
objectives, including the two not regularized with SIGReg.

---

## 3. Experiment 2 — Rotated (non-axis-aligned) subspace

**Method.** SIGReg regularizes the dynamics/inverse/finetuned encoders toward an isotropic
$\mathcal{N}(0,I)$, which forbids a privileged *axis* but not a privileged *subspace*. Tested this with
cross-validated Partial Least Squares regression, predicting $a_t$ from $\Delta h$ with an increasing
number of components, tracking held-out $R^2$ against a shuffled-label null (to guard against
overfitting given $D{=}256$ comparable to the per-fold sample size).

**Results.** For all four encoders, held-out $R^2$ tracked at or below the null curve across all tested
component counts (1–20), with no plateau above zero.

**Conclusion.** No compact linear subspace — axis-aligned or rotated — encodes $a_t$ recoverably from
$\Delta h$, for any of the four encoders. Combined with Experiment 1, this rejects the hypothesis as
originally framed.

---

## 4. Implication for CEM Planning

The paper's Planning procedure (Section "Planning", `metodo.tex`) scores each candidate action by

$$\|\mathrm{Dyn}_\phi(h_{t-1}, \mathrm{Proj}_\omega(a)) - h_t\|^2$$

i.e. raw, uniformly-weighted MSE over all 256 dimensions. Experiment 2's negative result predicts a
specific failure mode: if the action-induced shift is small and diffuse relative to $h_{t+1}$'s
action-independent variance, most of that squared error is an irreducible noise floor no candidate
action can reduce — different candidates, including the correct one, would score nearly identically.
This was tested directly.

---

## 5. Experiment 3 — Does raw MSE favor the true action?

**Method.** Using the actual trained `Dyn_φ`/`Proj_ω` from `archive/pretrain_val.ckpt` (the only
checkpoint of the four with a real dynamics head) on the same held-out batch (mixed
`within_stay`/`cross_stay`, matching `Dyn_φ`'s training distribution):

- **Zero-vs-true**: does the true action beat predicting "nothing changed"?
- **Sparsity-matched decoy ranking**: for 246 GT-active pairs, rank the true action's MSE against 50
  decoys with the *same* $\|a\|_0$ and $\pm1$ composition, flips scattered onto random labels.

**Results.**

| Test | Observed | Chance |
|---|---:|---:|
| P(true action beats zero-action) | 0.472 | 0.500 |
| Decoy-rank: mean rank (of 51) | 23.7 | 26.0 |
| Decoy-rank: P(true action wins) | 0.008 | 0.020 |
| Decoy-rank: P(rank ≤ 5) | 0.049 | 0.098 |

**Conclusion.** Raw MSE against `Dyn_φ` provides **no reliable signal**, even in the easiest possible
comparison (true action vs. doing nothing), and performs at-or-below chance at the harder task CEM
actually needs to solve (identifying which labels changed, given the right count). This is a scoring-
function problem, not a CEM search-quality problem.

---

## 6. Experiment 4 — Does distance reweighting fix it?

**Method.** Two escalating reweightings of the same MSE objective, evaluated identically to Experiment 3
(same decoys, same `Dyn_φ`):

- **Diagonal whitening**: per-dimension noise variance estimated from the zero-action residual across
  the full batch.
- **Full Mahalanobis whitening**: Ledoit-Wolf shrinkage-regularized 256×256 residual covariance.

**Results.**

| Metric | raw MSE | diagonal-whitened | Mahalanobis | chance |
|---|---:|---:|---:|---:|
| P(beats zero-action) | 0.472 | 0.480 | 0.431 | 0.500 |
| Decoy-rank mean | 23.7 | 23.7 | 23.3 | 26.0 |
| Decoy-rank P(win) | 0.008 | 0.012 | 0.012 | 0.020 |
| Decoy-rank P(rank ≤ 5) | 0.049 | 0.057 | 0.061 | 0.098 |

The diagonal noise-floor estimate had only a 4.8× max/min ratio across dimensions (near-isotropic — no
lopsided scale for a diagonal reweight to exploit). Ledoit-Wolf shrinkage was low (0.113, meaning real
off-diagonal structure was found and trusted), yet Mahalanobis whitening still didn't clear chance by any
meaningful margin, and was mildly worse on one metric.

**Conclusion.** Three escalating levels of linear/quadratic reweighting (Euclidean → diagonal → full
covariance) all land within noise of chance and of each other. The bottleneck isn't the distance metric:
nothing in `Dyn_φ`'s plain-MSE training objective ever penalizes a wrong action's prediction for landing
close to the true target, so no post-hoc reweighting of that geometry can recover discriminability.

---

## 7. Experiment 5 — Scoring via the inverse-dynamics head

**Method.** Swapped CEM's fitness function to the trained `Inv_ψ` (`ActionPredictor`), which already
decodes actions with real skill (reported onset/resolution AUROC ≈0.66–0.70, contextualized ≈0.83).
Since `Inv_ψ(\Delta h)` doesn't take a candidate action as input, its onset/resolution logits are
computed **once per pair** from the observed displacement; each candidate is scored by
$\mathrm{BCE}(\text{onset\_logits}, \mathbb{1}[a{=}1]) + \mathrm{BCE}(\text{res\_logits}, \mathbb{1}[a{=}{-1}])$.
Re-sampled a **cross_stay-only** batch (1024 pairs, 522 GT-active) to match `Inv_ψ`'s actual training
distribution (`inverse_pretrain.yaml`: `pair_types: [cross_stay]`), and reran `Dyn_φ`'s raw-MSE score on
the identical pairs/decoys for a fair comparison.

**Results.**

| Scorer | P(beats zero-action) | Decoy-rank mean | P(win) | P(rank ≤ 5) |
|---|---:|---:|---:|---:|
| `Dyn_φ` + raw MSE | 0.452 | 23.9 | 0.011 | 0.079 |
| `Inv_ψ` likelihood (raw) | **0.000** | **4.6** | **0.508** | **0.772** |
| chance | 0.500 | 26.0 | 0.020 | 0.098 |

**Conclusion.** The decoy-rank result is decisive and real: `Inv_ψ` identifies which labels changed far
better than chance, and far better than `Dyn_φ`. The `0.000` on the zero-vs-true test is **not**
contradictory — it's mathematically forced: at a genuinely-flipped label, zero-action's cost
$-\log(1-p)$ beats true-action's cost $-\log(p)$ whenever $p<0.5$, and `Inv_ψ` was trained with
`onset_pos_weight`/`resolution_pos_weight` both `1.0` (no imbalance correction) against real base rates
of ~1–3%, so predicted probabilities for genuine positives plausibly never clear 0.5.

---

## 8. Experiment 6 — Base-rate calibration

**Method.** Replaced raw `BCE(model, target)` with a log-likelihood-ratio score
`BCE(model, target) − BCE(base_rate, target)`, using **per-label** empirical base rates from the same
cross_stay sample, so a candidate is only rewarded for beating blind guessing at each position rather
than for guessing "no change" everywhere.

**Results.**

| Metric | `Inv_ψ` raw | `Inv_ψ` calibrated | chance |
|---|---:|---:|---:|
| P(beats zero-action) | 0.000 | **0.598** | 0.500 |
| Decoy-rank mean | 4.6 | 10.4 | 26.0 |
| Decoy-rank P(win) | 0.508 | 0.103 | 0.020 |
| Decoy-rank P(rank ≤ 5) | 0.772 | 0.375 | 0.098 |

**Conclusion.** Calibration fixed the zero-vs-true test as predicted (0.000 → 0.598, clears chance), but
cost real decoy-rank performance (still far above chance, but roughly halved). Diagnosis: per-label rates
inject a rarity-dependent bonus ($\log r_i$ per position) unrelated to the model's actual confidence at
that position — a decoy landing on a rarer label than the true one gets an unearned discount — compounded
by noisy per-label rate estimates at this sample size (~1024 pairs / 76 labels, ~0.9% average rate).
**Recommended, not yet run:** a single *pooled* scalar rate (not per-label) should fix the zero-vs-true
asymmetry (which is about differing flip *counts*) without disturbing decoy-rank (which compares
same-count candidates, all paying an identical pooled adjustment).

**Status.** The inverse-model contextualized inference pipeline (separate from CEM planning) is already
in production use and unaffected by this calibration question.

---

## 9. Experiment 7 — Forward dynamics: capacity vs. generalization

**Hypothesis under test.** Is `Dyn_φ`'s poor planning-discriminability (Experiments 3–4) a capacity
problem, fixable by a bigger/better `DynamicsPredictor`, rather than an objective problem?

**Method.** Standalone script `demo/dynamics_headroom_check.py` compares `Dyn_φ`'s true-action MSE on a
**train**-split sample against the same metric on **val**, for two checkpoints: `pretrain_val.ckpt`
(selected by dynamics `val/loss`) and `pretrain_auroc.ckpt` (selected by a downstream linear-probe AUROC
callback). If train ≈ val ≈ noise floor, the model underfits even its own training data (capacity would
plausibly help); if train ≪ val, that's a generalization gap (overfitting), and more capacity is likely
to make it worse.

**Results** (`BATCH_SIZE=1024` each split):

| Checkpoint | train MSE (true action) | val MSE (true action) | val zero-action baseline | train–val gap |
|---|---:|---:|---:|---:|
| `pretrain_val.ckpt` | 0.1274 | 0.1846 | 0.1839 | 31.0% |
| `pretrain_auroc.ckpt` | 0.0176 | 0.2298 | 0.2298 | **92.3%** |

On train data, true-action prediction clearly beats zero-action for both checkpoints (e.g.
`pretrain_val.ckpt`: 0.1274 vs 0.1347); on val, true-action ≈ zero-action for both — consistent with
Experiment 3.

**Conclusion.** Neither checkpoint shows the "train ≈ val ≈ floor" signature that would support
underfitting. `pretrain_auroc.ckpt` shows severe overfitting (train loss 13× smaller than val) —
unsurprising, since it was never selected for dynamics generalization, only for a downstream
classification-probe metric. Even the properly-selected `pretrain_val.ckpt` shows a real 31% gap. This
**does not support increasing `DynamicsPredictor` capacity** — on an architecture that already overfits
somewhat, more capacity is more likely to widen the gap than close it. It also surfaces an independent,
actionable finding: checkpoint *selection criterion* matters — `ckpt_path_auroc`-selected checkpoints
should not be used for planning.

---

## 10. Overall Conclusions

1. **Hypothesis rejected**, for all four encoders and both axis-aligned and rotated-linear formulations:
   there is no small, action-proportional subset of latent features. SIGReg's isotropy constraint is a
   plausible mechanism — it forbids a privileged axis by design, and empirically extends to forbidding a
   privileged subspace too.
2. **CEM planning via `Dyn_φ` + raw/whitened/Mahalanobis MSE is not viable as currently trained** — the
   objective never penalizes wrong-action predictions for landing near the true target, so no distance
   reweighting can recover discriminability post hoc.
3. **The inverse-dynamics head carries substantial, decisive action-discriminating signal** — a working,
   cheap (no retraining) planning score is reachable by scoring CEM candidates through `Inv_ψ` instead of
   `Dyn_φ` + distance, once the base-rate calibration is fixed. This departs from the paper's framing of
   Planning as querying the forward simulator specifically.
4. **Increasing `DynamicsPredictor` capacity is not supported by the evidence** — both tested checkpoints
   show real train/val generalization gaps (overfitting signatures), not the flat train≈val≈floor
   signature underfitting would produce.

## 11. Recommended Next Steps

- Re-run Experiment 6 with a **pooled** (not per-label) base rate to recover decoy-rank performance
  while keeping the zero-vs-true fix, before finalizing an inverse-head-based CEM scorer.
- If pursuing the forward dynamics model further: prioritize **regularization** (weight decay, dropout)
  or **training-pair diversity** over capacity increases; consider a **contrastive/InfoNCE-style** loss
  term that explicitly separates wrong-action predictions from the true target, addressing the root cause
  identified in Experiments 3–4 rather than working around it.
- Ensure any `Dyn_φ` checkpoint used for planning is selected by dynamics `val/loss`
  (`ckpt_path`), never by the downstream AUROC callback (`ckpt_path_auroc`) — Experiment 7 shows these
  can diverge dramatically in dynamics-generalization quality.
