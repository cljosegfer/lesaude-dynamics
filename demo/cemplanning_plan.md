# Plan — Onset/Resolution CEM Planning for the forward Dynamics model

## Context
The Planning cell in `papel/experiments.tex:31`, Table `tab:contextualized`, is a leftover
placeholder: `Dynamics (CEM Planning) & 0.740 [pending]`. No run ever produced it.
For comparison, Inverse is 0.8326 [0.8181–0.8456] and Carry-forward is 0.7553 [0.7435–0.7649].

The only CEM code is `anatomie/cem_planning0.py`, which was ported from another codebase:
- Its imports, `LeJEPA_Model` (with BatchNorm), batch keys and `model_state_dict` loading do not exist here.
- It samples a free 3-way categorical {-1,0,+1} per label.
- Its default prior is 0.7 "stable" per label, which gives about 23 flips per candidate. Real active pairs average about 2.5.

Goal: a new planner for the current stack that fills the table cell under exactly the same protocol as
the Inverse and Carry-forward rows. The stack is `ResNet1d` + `ActionProjector` + `DynamicsPredictor`,
with Lightning checkpoints in `archive/`.

## Evaluation of the proposed onset/resolution scheme
**Verdict: adopt it, with y_{t-1} masking and joint scoring. It fixes the real defect, which is not
the one described.**

1. **Where the defect is.** The paper's stay/flip scheme (`metodo.tex:65`) is well-defined at
   y_{t-1}=0: there, a flip is an onset (a=+1). The defect is in `cem_planning0.py`:
   - It proposes a=-1 on absent labels and a=+1 on present labels.
   - These actions are out of distribution for Dyn, because training actions a=y_{t+1}-y_t never
     contain them.
   - `clip()` then erases them in label space, but only after they have already moved ĥ.
2. **Equivalence.** For a given pair, each label has exactly one live head: onset if y_{t-1,i}=0,
   resolution if y_{t-1,i}=1. So the masked two-head search space is identical to stay/flip.
3. **What the two heads add:**
   - (a) **Separate per-class priors.** The train cross-stay base rates are wildly different:
     P(onset|y=0) has mean 0.010 and median 0.003; P(resolution|y=1) has mean 0.48 (range 0.13–0.77).
     One shared "stability" prior is wrong for both heads.
   - (b) **A continuous readout,** ŷ = y_{t-1}(1−p_res) + (1−y_{t-1})p_ons, taken from the final CEM
     marginals. This is exactly the inverse model's formula (`scripts/evaluate_inverse.py:137`), so the
     table row becomes comparable. The old "soft" score (`cem_planning0.py:263`) was always equal to
     the hard one.
   - (c) **Onset and resolution diagnostics.**
4. **Requirements:**
   - The heads are independent **only as proposal parameters**. Every candidate
     a = u_ons⊙(1−y) − u_res⊙y is scored **jointly**, on one population with one elite set.
     Two separate CEM loops would each try to explain the other head's displacement with spurious flips.
   - Sampling and refit are both masked. Inactive entries are frozen, and they are multiplied by 0
     in the readout.
5. **What it cannot fix.** `anatomie/action_subspace_planning_report.md` §5 found that raw-MSE scoring
   through Dyn is at chance on held-out pairs: P(E(a_gt) < E(0)) = 0.472 for `pretrain_val`.
   With no such signal, any parametrization ends up ≈ carry-forward, so the script measures this
   directly (see Diagnostics).
6. **Side finding.** The displacement table in the uncommitted `papel/appendix.tex:130-149` is an artifact.
   - It shows ĥ MSE 1.300 vs a do-nothing error of 0.172.
   - Its numbers match `anatomie/dyn_pred_embedding.ipynb`, which is gitignored. That notebook
     computes `dynamics(ht, proj(at))` with no `ht +` (the tracked `demo/notebook_dynamics_diagnostic.py:111`
     has the same pattern), and uses `SPLIT="train"`, whereas the text says val.
   - A CPU probe shows `pretrain_displacement.ckpt` behaves like a genuine displacement predictor:
     output variance per dim 0.003, cos(out, h) = −0.26.
   - Its training `val/pred_loss` is 0.200, which equals MSE(h_t + d̂, h_{t+1}).

## Design — new `scripts/cem_planning.py` + `configs/cem_planning.yaml`
Follow the idiom of `scripts/evaluate_inverse.py`:
- hydra, with `sys.path` inserting `src/`
- fp32 inference
- DataLoader with `shuffle=False` and the `spawn` context
- a self-contained copy of `_bootstrap_auroc` (`evaluate_inverse.py:83-100`), extended with an optional
  `mask` argument that restricts rows per class.

**Config** (`defaults: [data, _self_]`):
- `ckpt_path: archive/pretrain_displacement.ckpt`
- `dyn_target: auto` (or `next_state` / `displacement`)
- `pair_types: [cross_stay]` (must be a list)
- `split: test`; use `val` for tuning.
- model dims: `embedding_dim 256`, `action_dim 76`, `predictor_hidden_dim 512`
- `batch_size 256`, `num_workers 4`
- `max_batches: null`
- `cem`:
  - `n_samples: 512`, `n_elites: 64`, `n_iters: 16`
  - `smoothing: 0.1`, which is the weight on the old distribution (the `cem_planning0` convention)
  - `prior: base_rate` (or `pooled`)
  - `p_min: 0.005`
- `evaluate`: `n_bootstrap: 1000`, `seed: 42`
- `save_arrays: false`

With `pairs_path` taken from `data.yaml` (`pairs.lance`, k=1), this is the same protocol as the
`inverse_pretrain` config used for both reference rows: 8,062 test pairs, 68/76 valid classes, and
identical bootstrap resamples.

**Loading.**
- Use the strict prefix-stripping loader from `anatomie/dynamics_headroom_check.py:51-77`:
  `backbone.`, `projector.proj.` and `projector.pred.` prefixes, `eval()`, no grad.
- `dyn_target: auto` resolves to `displacement` iff `ckpt["hyper_parameters"]["ckpt_path"]` contains
  "displacement". Nothing else in the checkpoint distinguishes the variants. Print the resolved value.
- Proj and Dyn have no BatchNorm (`src/models/dynamics.py:95-155`), so batched candidate scoring is exact.

**Energy.** `energy_fn(actions[B,N,C]) → [B,N]`, computed on chunks of the B·N rows:
- next_state: E(a) = mean_D ‖Dyn(h_prev, Proj(a)) − h_curr‖²
- displacement: E(a) = mean_D ‖Dyn(h_prev, Proj(a)) − (h_curr − h_prev)‖²
  - This equals ‖h_prev + d̂ − h_curr‖², so both targets are in units of MSE to h_curr.

**`OnsetResolutionCEM.plan(energy_fn, y_prev, prior_ons, prior_res, generator)`**, batched on GPU:
- Masks: `m_ons = y_prev==0`, `m_res = y_prev==1`.
- Initialize `p_ons = prior_ons` and `p_res = prior_res`, expanded to [B,C].
- Initialize the best-so-far as a=0 with E(0).
- Each iteration:
  1. Sample `u_ons ~ Bern(p_ons)⊙m_ons` and `u_res ~ Bern(p_res)⊙m_res`, both [B,N,C].
     Set `a = u_ons − u_res`, so every candidate is admissible by construction.
  2. Compute E and take the elites with `topk(K, largest=False)`.
  3. Refit each head: `p_h ← where(m_h, α·p_h + (1−α)·mean_elite(u_h), p_h)`, then clamp to
     [p_min, 1−p_min]. The floor keeps rare onsets explorable.
  4. Update the best-so-far per pair.
  5. Record the mean best-E, the mean entropy over admissible entries, and the fraction of undecided
     entries (0.1<p<0.9).
- Return `p_ons` and `p_res` (NaN where inactive), `a_star`, `E_star`, `E_zero` and the history.
- Because the energy is injected, an oracle test is trivial.

**Priors.**
- Built from labels only, on the train split with the same `pair_types`. There is no waveform I/O;
  it takes about 2 s. The same pattern is in `demo/dataset_stats.py:51-77`.
  - `Y = lance.dataset(lance_path).to_table(columns=["icd"])` gives an [800k,76] label matrix.
  - The pairs come from `lance.dataset(pairs_path).to_table(filter="fold<=17 AND pair_type IN …", columns=["idx_t","idx_t1"])`.
- Rates, Laplace-smoothed:
  - π_ons,c = #(y_t=0 ∧ y_{t+1}=1) / #(y_t=0)
  - π_res,c = #(y_t=1 ∧ y_{t+1}=0) / #(y_t=1)
- `pooled` uses one pooled ratio per head.
- Priors cannot inflate macro-AUROC: a score that is constant within a class gives per-class AUROC 0.5.

**Readouts and metrics.** All use the bootstrap mean and 95% CI, with the same function and seed:
- **Contextualized soft (headline):** y(1−p_res) + (1−y)p_ons vs y_{t+1}.
- **Contextualized hard:** clip(y + a*), the paper's current formula.
- **In-run carry-forward on the same pairs:** this must reproduce 0.7553 [0.7435–0.7649].
- **Onset AUROC** only over y_{t-1}=0 entries; **resolution AUROC** only over y_{t-1}=1 entries.
  - `evaluate_inverse` does not mask, which would hand a y-aware planner the structural boost.
  - The fair comparison across models is the contextualized AUROC.

**Diagnostics** (port `cem_planning0.py:304-385` and extend):
- **Target sanity check:** mean do-nothing ‖h_curr−h_prev‖² vs mean E(0). Warn if E(0) is more than
  2× the do-nothing value, since that means the wrong `dyn_target`. A wrong target gives ≈1.0–1.3;
  a correct one gives ≈0.1–0.25.
- **Headroom:** on GT-active pairs, P(E(a_gt) < E(0)), where chance is 0.5. E(a_gt) is computed after
  planning and is never used by the search.
- **Failure split** on GT-active pairs:
  - search failure: E(a*) > E(a_gt)
  - model failure: E(a*) ≤ E(a_gt) and a* ≠ a_gt
- The per-iteration convergence table.
- Stable/active confusion; mean #onsets and #resolutions predicted vs GT; micro flip precision/recall.
- AUROC split by GT stable/active, for CEM soft vs carry-forward.
- **Output:**
  - stdout, in the same style as `evaluate_inverse`
  - `cem_results.json` (metrics, resolved config, history) in `HydraConfig.get().runtime.output_dir`
  - with `save_arrays: true`, an `.npz` holding p_ons, p_res, a*, E_zero, E_star and E_gt

## Files
- **New:** `scripts/cem_planning.py` and `configs/cem_planning.yaml`.
- **Optional:** `gorgonoid/cem_planning.sh`, following `gorgonoid/dynamics_displacement.sh`. It needs
  no wandb, so the `WANDB_API_KEY` line is not copied.
- **Untouched:** `anatomie/cem_planning0.py`, kept as an archive.
- **Follow-ups after the results, only if wanted:**
  - Fix the `ht +` reconstruction and the split in the displacement diagnostic, then regenerate
    appendix `tab:embedding-mse-displacement`.
  - Rewrite the Planning paragraph (`metodo.tex:65`): masked heads, base-rate priors, contextualized
    readout.
  - Update `experiments.tex:13,31`.

## Verification
1. **Oracle unit test** (run from the scratchpad):
   - Run `OnsetResolutionCEM.plan` with E(a)=‖a−a_gt‖² on random y_prev/a_gt.
   - Expect a*==a_gt and a soft AUROC of about 1.
   - Assert that every sampled action is admissible: a=+1 ⇒ y=0 and a=−1 ⇒ y=1.
2. **Smoke test:** `HYDRA_FULL_ERROR=1 python scripts/cem_planning.py max_batches=2 evaluate.n_bootstrap=10`.
   Check that E(a*) ≤ E(0) for every pair, that hard == clip(y+a*), and that the target sanity line is clean.
3. **Protocol check:** the in-run carry-forward reproduces 0.7553 [0.7435–0.7649] on the full test set.
4. **Real runs** on the val/loss-selected checkpoints `pretrain_val`, `pretrain_cross`,
   `pretrain_multistep` and `pretrain_displacement`. Never use the `*_auroc` twins (report §9).
   - Tune N/K/T/smoothing/prior on `split=val`, and check that best-E plateaus as N grows over
     {128, 512, 2048}. Then run `split=test` once.
   - Read the headroom line first. If P(E(a_gt)<E(0)) ≈ 0.5, the soft AUROC ≈ carry-forward,
     and that is the honest table entry.
