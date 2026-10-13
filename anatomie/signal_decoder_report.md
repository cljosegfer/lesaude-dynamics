# Signal-Space Test of the Forward Dynamics Model — Report

**Repo:** lesaude-dynamics · **Scope:** `papel/metodo.tex` (Dynamics Model, Planning) · **Date:** 2026-10-06
**Code:** [`anatomie/signal_decoder.py`](signal_decoder.py), [`anatomie/median_beats.py`](median_beats.py),
`load_dynamics()` in [`anatomie/encoders.py`](encoders.py)
**Run:** SLURM job 52836 on gorgona5 (`gorgonoid/decoder.sh`) · **Outputs:** `outputs/signal_decoder/` (gitignored)

## TL;DR

A decoder was trained from each frozen encoder's embedding $h$ back to the ECG (the 12-lead median beat, heart
rate and RR irregularity), and the forward model's predictions were read through it.

- **Pretrained encoders keep less of the ECG than random weights.** Test beat $R^2$ is 0.52–0.62 for the six
  pretrained encoders against 0.72 for a randomly initialized ResNet. The next-state forward encoders lose the most,
  mainly in V2–V3, and lose part of the heart rate.
- **Decoded forward predictions are worse than doing nothing.** For all four next-state models, decoding
  $\mathrm{Dyn}(h_t, a_t)$ estimates the next ECG worse than decoding $h_t$ (beat MSE +9% to +17%, heart-rate error
  15.7–15.9 → 19.6–28.5 bpm), on pairs with and without a diagnosis change, although Dyn is 3.6–4.6% closer to
  $h_{t+1}$ in latent MSE.
- **Dyn's latent gain is regression to the mean.** A parameter-free shrink toward the population mean,
  $\mu + \beta(h_t - \mu)$, with the same contraction as Dyn, is closer to $h_{t+1}$ than Dyn in latent MSE for
  every model and improves the decoded ECG. What Dyn adds beyond the shrink breaks the signal: a constant offset
  that the decoder reads as +6 to +16 bpm, and a pair-dependent remainder that scrambles each patient's heart rate.
- **The action carries little per-pair information.** The true action beats a same-count decoy on 51–53% of
  untied active pairs (about 1% of the beat error), and on its own it does not improve on doing nothing.
- **Averaged over pairs, onset effects point the right way, but generically.** Simulated onset effects align with
  the real ones (cosine 0.20–0.41) mostly along a direction shared by all onsets (label-specific part 0.11–0.21), at
  22–44% of the real size. Atrial fibrillation onset raises the real heart rate by 14.7 bpm; the models give at most
  7.7. Resolutions are not modelled.

---

## 1. Question

The forward model's diagnostics so far stay in latent space. The embedding-space diagnostic
(`papel/appendix.tex`, `tab:embedding-mse`) compares $\mathrm{Dyn}(h_t, a_t)$ with $h_{t+1}$ by MSE, and CEM
planning ranks candidate actions by the same error. Neither says:

1. whether a latent improvement is a better prediction of the next ECG, or
2. whether Dyn moves an embedding the way a diagnosis transition changes the ECG (e.g. atrial fibrillation onset
   → faster, irregular rhythm).

A decoder from $h$ to the signal answers both. Decode each estimate of $h_{t+1}$ and compare it with the real next
ECG. Then decode Dyn's response to a single-label action and compare it with how the ECG really changes when that
label flips.

---

## 2. Setup

### 2.1 What a decoder can recover from $h$

**Method.** $h$ is the encoder's fc output after global average pooling over time, so it should not know where the
beats fall in the 10 s window. On 512 val triage ECGs, 8 s crops starting $s$ samples apart were embedded, and
$\|h(\text{crop at } s) - h(\text{crop at } 0)\|^2$ was measured relative to the mean squared distance between two
different ECGs. 0 means the start position is ignored; 1 means the crops are as different as two unrelated ECGs.

**Results.**

| encoder | shift 10–1000 samples (20 ms–2 s) | full 10 s vs first 8 s |
|---|---:|---:|
| random init | 0.08–0.24 | 0.033 |
| `pretrain_val` | 0.003–0.008 | 0.002 |
| `pretrain_multistep0` | 0.008–0.022 | 0.003 |
| `inverse` | 0.023–0.063 | 0.012 |

**Conclusion.** Pretrained embeddings are essentially phase-invariant. A decoder of the raw 12 × 5000 strip would
learn the average over beat positions, a near-flat trace. The decoder targets are instead what $h$ can carry:

- **beat:** the 12-lead median beat, 0.4 s before to 0.6 s after the QRS (12 × 500 at 500 Hz), in the input's
  per-lead z-units
- **heart rate:** bpm from the median RR interval
- **RR irregularity:** normalized RMSSD, the RMS of successive RR differences over the mean RR

### 2.2 Median-beat extraction

[`anatomie/median_beats.py`](median_beats.py) detects QRS complexes on the lead-summed squared slope of a 5–20 Hz
band-pass (100 ms moving average; peaks at least 250 ms apart and above 30% of the envelope's 99th percentile). It
takes a first-pass median beat, re-aligns every beat to it by cross-correlation (±20 ms), and takes the per-lead
median. An ECG is unusable with fewer than 3 complete beats or a rate outside 20–250 bpm.

Checked on 2,000 random val ECGs:

- 99.8% usable, median 11 beats per ECG;
- heart rate 45 / 75 / 141 bpm at the 1st / 50th / 99th percentile;
- normalized RMSSD median 0.150 on ECGs coded I48 (atrial fibrillation and flutter) against 0.020 otherwise
  (AUROC 0.68 for I48);
- re-alignment lowered the spread of single beats around their median in the QRS window from 0.310 to 0.254.

On the full table, 99.7% of 800,035 ECGs are usable. The cache took 24 min to build with 16 workers.

### 2.3 Encoders

Training settings as stored in each checkpoint's `hyper_parameters`.

| name | checkpoint | script | pairs | lr | λ | weight decay | epoch |
|---|---|---|---|---:|---:|---:|---:|
| `random` | seed-0 random init | — | — | — | — | — | — |
| `inverse` | `archive/inverse.ckpt` | `inverse_pretrain.py` | k=1 cross-stay | 2e-2 | 2e-3 | 1e-2 | 65 |
| `dyn` | `archive/pretrain_val.ckpt` | `pretrain.py` | k=1 within + cross-stay | 1e-3 | 5e-3 | 1e-2 | 19 |
| `multistep` | `archive/pretrain_multistep0.ckpt` | `pretrain.py` | multistep k ≤ 12, cross-stay | 1e-3 | 5e-3 | 1e-2 | 6 |
| `regula` | `archive/pretrain_regula.ckpt` | `pretrain.py` | multistep k ≤ 12, cross-stay | 1e-3 | 5e-2 | 1e-1 | 5 |
| `joint` | `archive/joint_ctx.ckpt` | `joint_pretrain.py` | multistep k ≤ 12, cross-stay | 1e-3 | 5e-3 | 1e-2 | 4 |
| `disp_mlp` | `runs/runs/20261002/224321/bb6b00025771/checkpoints/pretrain_displacement_mlp_.ckpt` | `pretrain_displacement.py`, `action_model=mlp` | multistep k ≤ 12, cross-stay | 1e-3 | 5e-3 | 1e-2 | 0 |

> **Naming.** The job labelled the last encoder `disp_lin`, so its output files are `*_disp_lin.*`, but the
> checkpoint is the **MLP** action-only variant. It is the epoch-0 checkpoint (best `val/pred_loss`), so its
> zero-initialized action map has had one epoch to grow. The linear action-only checkpoint (run
> `20261002/200605`) was not evaluated.

Every encoder except `random` and `inverse` has a forward model, loaded by `load_dynamics()`. For `disp_mlp`,
$\mathrm{Dyn}(h, a) = h + g(a)$, so $\mathrm{Dyn}(h, 0) = h$.

### 2.4 Decoder

- **Input:** $h$ (256-d, cached as float16), standardized with the fit-set mean $\mu$ and standard deviation.
- **Architecture:** an MLP 256 → 1024 → 1024 with ReLU and two linear heads. One head outputs the beat
  (12 × 500), the other the standardized log rhythm ($\log$ heart rate, $\log(\text{nRMSSD} + 0.01)$). The beat
  head's bias starts at the fit-set mean beat.
- **Loss:** MSE on the beat plus MSE on the standardized log rhythm.
- **Data:** fit on every usable ECG of folds 0–17 (718,922), epoch selected by fold-18 beat MSE (39,373 ECGs),
  scored on fold 19 (39,466).
- **Training:** 20 epochs, AdamW (lr 1e-3, weight decay 1e-2), batch 1024, cosine schedule.
- **Metric:** beat $R^2$ is pooled over leads and time, against the fit-set mean beat. Heart-rate and irregularity
  correlations are on the log scale.

The decoders are fit on the training folds although the forward encoders memorize training patients, whose
train-fold embeddings are therefore distributed differently from test ones. The fit–test gaps in §3 (at most
0.016 $R^2$) show this did not matter here.

### 2.5 Forward test

The test set is the k=1 cross-stay test pairs of `scripts/evaluate_inverse.py` (fold 19) whose two ECGs both have
usable beats: 8,020 of 8,062 pairs. Of these, 4,098 are active ($a_t \neq 0$; 2.5 flips per active pair on
average) and 3,922 stable. Each estimate of $h_{t+1}$ is decoded and scored against the real median beat, heart
rate and RR irregularity of $X_{t+1}$:

| condition | estimate of $h_{t+1}$ |
|---|---|
| do-nothing | $h_t$ |
| true | $\mathrm{Dyn}(h_t, a_t)$ |
| zero | $\mathrm{Dyn}(h_t, 0)$ |
| decoy | $\mathrm{Dyn}(h_t, \tilde a)$. $\tilde a$ is admissible given $y_t$ and has $a_t$'s onset and resolution counts, on random labels (the decoys of `_CEMProbe` in `scripts/joint_pretrain.py`); $\tilde a = 0$ on stable pairs |
| ceiling | $h_{t+1}$ itself: the decoder's best case |

The real median beat of $X_t$ (no decoder) is also scored, for scale. On active pairs, each alternative is compared
with the true action pair by pair. The comparison reports the fraction of pairs where the true action is better
and the mean error gap, with a 1,000-sample bootstrap 95% CI.

### 2.6 Label effects

The test covers every label and direction that flips in at least 20 test pairs: 76 entries, 40 onsets and 36
resolutions. Two effects are compared:

- **real effect:** the mean change of the decoded beat, $D(h_{t+1}) - D(h_t)$, over the pairs where the label
  flips, minus the same mean over the pairs where it stays as it was (absent for an onset, present for a
  resolution). The **raw** real effect is the same quantity computed on the real median beats.
- **model effect:** the mean over the same pairs of $D(\mathrm{Dyn}(h_t, \pm e_i)) - D(\mathrm{Dyn}(h_t, 0))$, a
  single-label intervention.

Effects are compared by cosine over the 12 × 500 beat and by the heart-rate change. The **other-labels** control is
the mean cosine of a model effect with the real effects of the other labels that flip in the same direction.

### 2.7 Follow-up analyses

§5 and the tie count in §4.2 come from ad-hoc scripts run after the job; they are not in the repo. They rebuild the
8,020 test pairs' median beats and embeddings on gorgona10, because the job's caches live on gorgona5's local
disk, and load the saved decoders. They reproduce the job's forward-test numbers exactly (for example, `dyn` beat
MSE 0.3292 / 0.3591 / 0.3559 for do-nothing / true / zero).

---

## 3. How much ECG each encoder keeps

**Results** (fold 19 except the fit and val columns):

| encoder | beat R² fit | beat R² val | beat R² test | HR MAE (bpm) | HR r | nRMSSD MAE | nRMSSD r |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.722 | 0.711 | 0.719 | 4.78 | 0.933 | 0.0566 | 0.626 |
| `regula` | 0.630 | 0.609 | 0.618 | 5.40 | 0.923 | 0.0569 | 0.618 |
| `inverse` | 0.627 | 0.603 | 0.611 | 4.48 | 0.938 | 0.0523 | 0.692 |
| `disp_mlp` | 0.596 | 0.583 | 0.592 | 4.56 | 0.939 | 0.0615 | 0.540 |
| `joint` | 0.587 | 0.569 | 0.578 | 5.38 | 0.924 | 0.0569 | 0.626 |
| `dyn` | 0.545 | 0.524 | 0.532 | 7.34 | 0.869 | 0.0611 | 0.575 |
| `multistep` | 0.531 | 0.513 | 0.519 | 7.30 | 0.873 | 0.0594 | 0.589 |

Per-lead test $R^2$:

| encoder | I | II | III | aVR | aVL | aVF | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.686 | 0.722 | 0.751 | 0.696 | 0.744 | 0.725 | 0.683 | 0.697 | 0.717 | 0.732 | 0.733 | 0.719 |
| `regula` | 0.649 | 0.672 | 0.651 | 0.654 | 0.661 | 0.658 | 0.578 | 0.532 | 0.524 | 0.596 | 0.616 | 0.606 |
| `inverse` | 0.590 | 0.655 | 0.672 | 0.604 | 0.675 | 0.626 | 0.562 | 0.553 | 0.569 | 0.598 | 0.598 | 0.589 |
| `disp_mlp` | 0.562 | 0.604 | 0.618 | 0.579 | 0.609 | 0.597 | 0.519 | 0.574 | 0.603 | 0.599 | 0.612 | 0.600 |
| `joint` | 0.601 | 0.633 | 0.601 | 0.613 | 0.621 | 0.597 | 0.558 | 0.483 | 0.479 | 0.563 | 0.586 | 0.579 |
| `dyn` | 0.557 | 0.594 | 0.558 | 0.563 | 0.584 | 0.562 | 0.520 | 0.435 | 0.415 | 0.503 | 0.537 | 0.533 |
| `multistep` | 0.544 | 0.575 | 0.542 | 0.552 | 0.566 | 0.543 | 0.501 | 0.423 | 0.412 | 0.498 | 0.531 | 0.529 |

**Observations.**

- **Every pretrained encoder keeps less of the beat than random weights,** by 0.10 $R^2$ (`regula`, `inverse`) to
  0.20 (`dyn`, `multistep`). Pooled random convolutional features are a rich random projection of the input, and
  training makes an encoder discard what its objective does not use. This $R^2$ therefore measures how much ECG
  detail survives in $h$, not representation quality. It bounds what any forward model on that encoder can
  simulate in signal space.
- **The next-state forward encoders lose the most,** above all in V2–V3 (`dyn` 0.435 / 0.415 against `random`
  0.697 / 0.717), and they lose part of the heart rate (MAE 7.3 against 4.5–5.4 bpm for the others).
  - A plausible reason: heart rate changes by 15.7 bpm on average between consecutive stays, and a feature that
    changes unpredictably between visits only adds error to the forward loss.
  - `joint` and `regula` keep more of the heart rate (r 0.92).
- **`inverse` keeps the most RR irregularity (0.692).** Rhythm features mark atrial fibrillation transitions, which
  its objective rewards.
- **`disp_mlp` keeps the heart rate but the least irregularity.** It is the only encoder whose V2–V3 decode better
  than its limb leads.
- **The decoders do not overfit.** The fit–test $R^2$ gap is at most 0.016, and fold-18 beat MSE moved by at most
  0.0005 over the last three epochs.

---

## 4. Forward test

### 4.1 Decoded estimates of the next ECG

**Beat MSE** against the real median beat of $X_{t+1}$ (all 8,020 pairs):

| encoder | real beat of $X_t$ | do-nothing | true | zero | decoy | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `dyn` | 0.3940 | 0.3292 | 0.3591 | 0.3559 | 0.3604 | 0.2591 |
| `multistep` | 0.3940 | 0.3383 | 0.3749 | 0.3756 | 0.3770 | 0.2654 |
| `regula` | 0.3940 | 0.3230 | 0.3533 | 0.3535 | 0.3552 | 0.2140 |
| `joint` | 0.3940 | 0.3289 | 0.3834 | 0.3839 | 0.3852 | 0.2347 |
| `disp_mlp` | 0.3940 | 0.3627 | 0.3628 | 0.3627 | 0.3630 | 0.2260 |

The same, split into active and stable pairs:

| encoder | active: do-nothing | active: true | stable: do-nothing | stable: true |
|---|---:|---:|---:|---:|
| `dyn` | 0.3505 | 0.3789 | 0.3070 | 0.3383 |
| `multistep` | 0.3593 | 0.3913 | 0.3163 | 0.3578 |
| `regula` | 0.3493 | 0.3723 | 0.2955 | 0.3335 |
| `joint` | 0.3537 | 0.3962 | 0.3029 | 0.3700 |
| `disp_mlp` | 0.3877 | 0.3879 | 0.3365 | 0.3365 |

**Heart-rate error** (mean absolute error, bpm; the first column is the real heart rate of $X_t$):

| encoder | real HR of $X_t$ | do-nothing | true | zero | decoy | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `dyn` | 15.75 | 15.75 | 28.50 | 27.42 | 28.56 | 7.68 |
| `multistep` | 15.75 | 15.88 | 19.83 | 19.88 | 19.87 | 7.63 |
| `regula` | 15.75 | 15.90 | 19.63 | 19.59 | 19.76 | 5.66 |
| `joint` | 15.75 | 15.65 | 22.27 | 22.08 | 22.42 | 5.66 |
| `disp_mlp` | 15.75 | 15.86 | 15.86 | 15.86 | 15.86 | 4.83 |

**RR irregularity error** (mean absolute error in normalized RMSSD):

| encoder | real $X_t$ | do-nothing | true | zero | decoy | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `dyn` | 0.0747 | 0.0711 | 0.0737 | 0.0721 | 0.0739 | 0.0615 |
| `multistep` | 0.0747 | 0.0694 | 0.0714 | 0.0706 | 0.0714 | 0.0598 |
| `regula` | 0.0747 | 0.0729 | 0.0865 | 0.0783 | 0.0877 | 0.0568 |
| `joint` | 0.0747 | 0.0689 | 0.0757 | 0.0734 | 0.0758 | 0.0564 |
| `disp_mlp` | 0.0747 | 0.0755 | 0.0760 | 0.0755 | 0.0756 | 0.0613 |

**Latent MSE** to $h_{t+1}$, the forward model's own metric:

| encoder | do-nothing | true | zero | decoy | active: do-nothing | active: true | active: zero | active: decoy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dyn` | 0.2364 | 0.2279 | 0.2286 | 0.2302 | 0.2911 | 0.2766 | 0.2778 | 0.2810 |
| `multistep` | 0.1999 | 0.1914 | 0.1935 | 0.1933 | 0.2350 | 0.2210 | 0.2249 | 0.2246 |
| `regula` | 0.3729 | 0.3556 | 0.3564 | 0.3585 | 0.4366 | 0.4031 | 0.4048 | 0.4088 |
| `joint` | 0.1880 | 0.1807 | 0.1827 | 0.1832 | 0.2223 | 0.2085 | 0.2125 | 0.2134 |
| `disp_mlp` | 1.1809 | 1.1787 | 1.1809 | 1.1809 | 1.2666 | 1.2625 | 1.2666 | 1.2668 |

**Observations.**

- **Signal and latent space disagree.** For every next-state model, the decoded prediction is worse than decoding
  $h_t$: beat MSE +9.1% (`dyn`), +10.8% (`multistep`), +9.4% (`regula`), +16.6% (`joint`), and heart-rate error up
  by 3.7–12.7 bpm. Yet the same predictions are 3.6–4.6% closer to $h_{t+1}$ in latent MSE.
- **The degradation comes from Dyn's state pathway.** It is as large on stable pairs, where $a_t = 0$ and true =
  zero, so it does not come from the action. RR irregularity follows the same pattern.
- **`disp_mlp` changes nothing.** Its displacement $g(a)$ is too small to move the decoded output.
- **The embeddings do hold the next visit's change.** Decoding $h_{t+1}$ cuts beat MSE by 21–38% relative to
  decoding $h_t$, and heart-rate error from about 15.8 to 4.8–7.7 bpm. Dyn recovers none of it.
- **The real beat of $X_t$ is a scale, not a baseline.** It scores worse than decoded $h_t$ because decoded beats
  are smooth conditional means, which MSE favours. Across these pairs, mean heart rate is 80.8 bpm at $X_t$ and
  80.9 at $X_{t+1}$, with a mean absolute change of 15.7 bpm.

### 4.2 Does the true action help, pair by pair?

Paired comparisons on the 4,098 active pairs. The gap is the error of the alternative minus the error of the true
action, so a positive gap favours the true action; the 95% bootstrap CI is in brackets.

| encoder | beat: vs do-nothing | beat: vs zero | beat: vs decoy | HR: vs do-nothing | HR: vs zero | HR: vs decoy |
|---|---:|---:|---:|---:|---:|---:|
| `dyn` | −0.0284 [−0.0315, −0.0253] | −0.0061 [−0.0078, −0.0045] | +0.0027 [+0.0006, +0.0046] | −13.78 [−14.65, −12.94] | −2.12 [−2.52, −1.72] | +0.11 [−0.38, +0.60] |
| `multistep` | −0.0320 [−0.0352, −0.0288] | +0.0014 [+0.0001, +0.0027] | +0.0040 [+0.0026, +0.0056] | −3.23 [−3.75, −2.65] | +0.11 [−0.12, +0.31] | +0.09 [−0.14, +0.30] |
| `regula` | −0.0230 [−0.0261, −0.0198] | +0.0004 [−0.0014, +0.0023] | +0.0037 [+0.0018, +0.0056] | −2.71 [−3.22, −2.13] | −0.08 [−0.29, +0.13] | +0.26 [+0.05, +0.48] |
| `joint` | −0.0425 [−0.0463, −0.0385] | +0.0009 [−0.0004, +0.0021] | +0.0034 [+0.0021, +0.0049] | −5.65 [−6.23, −5.00] | −0.38 [−0.55, −0.23] | +0.28 [+0.08, +0.46] |
| `disp_mlp` | −0.0001 [−0.0009, +0.0006] | −0.0001 [−0.0009, +0.0006] | +0.0005 [−0.0002, +0.0012] | +0.00 [−0.09, +0.09] | +0.00 [−0.08, +0.09] | −0.00 [−0.08, +0.08] |

**Ties against decoys.** The job's "P(true better)" against decoys reads 0.41–0.43 for every model, below 0.5 even
where the mean gap favours the true action. On 19.2% of active pairs the decoy *is* the true action, for example
when every present label resolves and the decoy must resolve the same ones, and a tie counts as "not better". On
untied pairs, the true action is better on 51.3–53.2% of pairs for the beat and 49.9–52.2% for the heart rate.

**Conclusion.** Per pair, the true action carries a detectable but tiny amount of information: it beats a
same-count decoy by 0.003–0.004 beat MSE, about 1% of the error. It does not beat the zero action consistently; for
`dyn` it is worse than the zero action, in both beat and heart rate. The online CEM probe of
`scripts/joint_pretrain.py` found the same near-chance ranking in latent space (0.53 for the forward-only multistep
checkpoint, true action against same-count decoys on 2,048 multistep val pairs).

---

## 5. Why decoded Dyn is worse although it is closer in latent space

### 5.1 Shrinkage toward the mean helps

**Method.** Dyn's state map is nearly linear and contracting. A least-squares fit of $\mathrm{Dyn}(h_t, 0)$ on
$[h_t, 1]$ has $R^2$ 0.997 (`dyn`), 0.991 (`multistep`), 0.979 (`regula`) and 0.992 (`joint`), and the
total-variance ratio of $\mathrm{Dyn}(h_t, 0)$ to $h_t$ is 0.86, 0.82, 0.67 and 0.81. The first suspect is therefore
the MSE-optimal regression to the mean, pushing predictions where the decoder never saw data. It was tested
directly with a parameter-free shrink toward the fit-set mean $\mu$:

$$\tilde h = \mu + \beta\,(h_t - \mu).$$

The contraction can be matched through the spread $\mathbb{E}\|z\|^2 / D$ in the decoder's standardized units $z$.
$h_t$ has spread 0.87–0.96 and $\mathrm{Dyn}(h_t, a_t)$ 0.61–0.76; $\beta = 0.9$ gives 0.70–0.78 and matches Dyn
for `dyn`, `multistep` and `joint` (0.743 vs 0.762, 0.701 vs 0.694, 0.715 vs 0.699). $\beta = 0.8$ matches `regula`
(0.617 vs 0.613).

**Results** (all 8,020 pairs; $\beta = 1$ is do-nothing; Dyn = true action):

Latent MSE to $h_{t+1}$:

| encoder | β = 1 | β = 0.9 | β = 0.8 | β = 0.7 | Dyn |
|---|---:|---:|---:|---:|---:|
| `dyn` | 0.2364 | **0.2223** | 0.2270 | 0.2506 | 0.2279 |
| `multistep` | 0.1999 | 0.1860 | **0.1842** | 0.1946 | 0.1914 |
| `regula` | 0.3729 | 0.3444 | **0.3350** | 0.3445 | 0.3556 |
| `joint` | 0.1880 | 0.1753 | **0.1752** | 0.1876 | 0.1807 |
| `disp_mlp` | 1.1809 | 1.0738 | 0.9886 | **0.9252** | 1.1787 |

Decoded beat MSE:

| encoder | β = 1 | β = 0.9 | β = 0.8 | β = 0.7 | Dyn |
|---|---:|---:|---:|---:|---:|
| `dyn` | 0.3292 | **0.3256** | 0.3268 | 0.3331 | 0.3591 |
| `multistep` | 0.3383 | **0.3345** | 0.3355 | 0.3418 | 0.3749 |
| `regula` | 0.3230 | 0.3167 | **0.3143** | 0.3161 | 0.3533 |
| `joint` | 0.3289 | **0.3239** | 0.3242 | 0.3300 | 0.3834 |
| `disp_mlp` | 0.3627 | **0.3544** | 0.3546 | 0.3633 | 0.3628 |

Decoded heart-rate error (bpm):

| encoder | β = 1 | β = 0.9 | β = 0.8 | β = 0.7 | Dyn |
|---|---:|---:|---:|---:|---:|
| `dyn` | 15.75 | 15.22 | **15.14** | 15.45 | 28.50 |
| `multistep` | 15.88 | 15.44 | 15.14 | **15.07** | 19.83 |
| `regula` | 15.90 | **15.35** | 15.70 | 16.74 | 19.63 |
| `joint` | 15.65 | 15.05 | 14.64 | **14.38** | 22.27 |
| `disp_mlp` | 15.86 | **15.43** | 15.99 | 17.16 | 15.86 |

On active pairs alone, $\beta = 0.9$ beats do-nothing on 58–66% of pairs for the beat (gap +0.0048 to +0.0098, all
CIs above zero).

**Conclusion.** Shrinkage is not the problem.

- **The shrink beats Dyn in Dyn's own metric.** At matched contraction, a shrink that learns nothing from pairs is
  closer to $h_{t+1}$ than Dyn for every next-state model, and it improves the decoded beat and heart rate where
  Dyn degrades them.
- **Dyn's latent advantage is accounted for by regression to the mean between visits.** An earlier ridge analysis
  (2026-10-02, 16,306 val + test k=1 pairs) found that $h_t$ explains 7–31% of the displacement's variance and the
  action at most 0.1%.
- **Caveat:** $\beta$ was picked on these test pairs from {0.9, 0.8, 0.7}. 0.9, the first value, already beats Dyn
  in latent MSE for all four next-state models, but a reported number should tune $\beta$ on val.

### 5.2 What Dyn adds beyond the shrink

**Method.** The state map is decomposed as

$$\mathrm{Dyn}(h, 0) = \mu + b\,(h - \mu) + c + R(h),$$

with $b = \langle \mathrm{Dyn}(h,0) - \mu,\ h - \mu \rangle / \|h - \mu\|^2$ pooled over the test pairs, $c$ the
mean residual (a constant offset), and $R(h)$ the rest. The shrink alone, $h_t + c$ and $h_t + R(h_t)$ are then
decoded. Sizes are mean squared values per dimension in the decoder's standardized units, against the do-nothing
error $\|h_t - h_{t+1}\|^2$ in the same units:

| encoder | $b$ | $c$ | $R$ | do-nothing error |
|---|---:|---:|---:|---:|
| `dyn` | 0.924 | 0.001 | 0.007 | 0.231 |
| `multistep` | 0.897 | 0.002 | 0.014 | 0.287 |
| `regula` | 0.803 | 0.004 | 0.028 | 0.379 |
| `joint` | 0.893 | 0.003 | 0.015 | 0.267 |

**Results.** For each estimate, the table gives the beat MSE, the mean decoded heart rate, the heart-rate error,
the correlation of its decoded heart rate with that of $h_t$ (across pairs), and the T-wave peak latency (mean
over II, V4, V5). Real $X_{t+1}$ has a mean heart rate of 80.9 bpm and a T-wave peak at 287 ms; decoded
$h_{t+1}$ averages 80.0–80.2 bpm.

| encoder | estimate | beat MSE | decoded HR (bpm) | HR error | HR r vs $h_t$ | T peak (ms) |
|---|---|---:|---:|---:|---:|---:|
| `dyn` | $h_t$ | 0.3292 | 80.1 | 15.75 | 1.00 | 265 |
| | shrink $\mu + b(h_t - \mu)$ | 0.3261 | 78.4 | 15.30 | 1.00 | 262 |
| | $h_t + c$ | 0.3378 | 96.5 | 22.89 | 0.98 | 255 |
| | $h_t + R$ | 0.3498 | 83.2 | 21.15 | 0.25 | 263 |
| | $\mathrm{Dyn}(h_t, 0)$ | 0.3559 | 98.6 | 27.42 | 0.20 | 249 |
| `multistep` | $h_t$ | 0.3383 | 80.1 | 15.88 | 1.00 | 266 |
| | shrink | 0.3345 | 80.0 | 15.43 | 1.00 | 261 |
| | $h_t + c$ | 0.3547 | 85.8 | 17.33 | 0.99 | 275 |
| | $h_t + R$ | 0.3614 | 79.3 | 18.10 | 0.32 | 264 |
| | $\mathrm{Dyn}(h_t, 0)$ | 0.3756 | 85.9 | 19.88 | 0.19 | 272 |
| `regula` | $h_t$ | 0.3230 | 80.1 | 15.90 | 1.00 | 276 |
| | shrink | 0.3143 | 72.3 | 15.68 | 0.99 | 275 |
| | $h_t + c$ | 0.3447 | 87.9 | 18.50 | 0.98 | 283 |
| | $h_t + R$ | 0.3414 | 83.4 | 18.55 | 0.57 | 269 |
| | $\mathrm{Dyn}(h_t, 0)$ | 0.3535 | 82.1 | 19.59 | 0.37 | 268 |
| `joint` | $h_t$ | 0.3289 | 80.2 | 15.65 | 1.00 | 273 |
| | shrink | 0.3238 | 78.9 | 15.02 | 1.00 | 268 |
| | $h_t + c$ | 0.3494 | 96.5 | 22.62 | 0.98 | 250 |
| | $h_t + R$ | 0.3606 | 77.4 | 17.30 | 0.29 | 268 |
| | $\mathrm{Dyn}(h_t, 0)$ | 0.3839 | 93.0 | 22.08 | 0.15 | 241 |

Beat MSE by time window for `dyn` (window boundaries relative to the QRS):

| window | $h_t$ | $h_t + c$ | $h_t + R$ | $\mathrm{Dyn}(h_t, 0)$ |
|---|---:|---:|---:|---:|
| P / PR (−400 to −60 ms) | 0.0917 | 0.0959 | 0.0989 | 0.1004 |
| QRS (−60 to +80 ms) | 1.2657 | 1.2770 | 1.3309 | 1.3455 |
| ST-T (+80 to +450 ms) | 0.2048 | 0.2177 | 0.2264 | 0.2343 |
| late (+450 to +600 ms) | 0.3004 | 0.3061 | 0.3072 | 0.3113 |

**Observations.**

- **The offset $c$ is tiny but reads as heart rate.** It is 0.4–1.1% of the do-nothing error in the decoder's
  standardized units, so latent MSE barely registers it. But it lies along a direction the decoder reads as heart
  rate:
  - adding it to $h_t$ raises the decoded rate by 16 bpm (`dyn`, `joint`) or 6–8 bpm (`multistep`, `regula`);
  - for `dyn` and `joint` it moves the T-wave peak 10–23 ms earlier, as a faster rate would.

  No such shift exists in the data: mean heart rate is 80.8 → 80.9 bpm, and decoded $h_t$ and $h_{t+1}$ both
  average 80.0–80.2.
- **The remainder $R$ scrambles each patient's heart rate.** It is larger (3–7% of the do-nothing error) and
  pair-dependent. The decoded rate of $h_t + R$ correlates with that of $h_t$ at only r = 0.25–0.57, and that of
  the full $\mathrm{Dyn}(h_t, 0)$ at 0.15–0.37. $R$ also adds most of the extra beat error for `dyn`, `multistep`
  and `joint` (for `regula`, $c$ and $R$ contribute about equally), mainly in the QRS and ST-T windows.
- **Together, $c$ and $R$ turn a helpful shrink into a harmful state map.** Neither is supported by the test data,
  and both are cheap in latent MSE, so the forward loss barely penalizes them. This is consistent with the earlier
  diagnosis of forward pretraining: it collapses training patients' recordings, and new patients' embeddings
  shrink and drift. A map fitted to training embeddings carries training-specific structure to new patients.
- **QRS amplitude shows the same thing visually.** Mean QRS peak-to-peak is 6.77 z-units for real $X_{t+1}$ and
  5.77 for decoded $h_t$ (`dyn`); decoded $\mathrm{Dyn}(h_t, a_t)$ is 5.17. The example pairs in the figures show
  Dyn's beats with a smaller QRS and an earlier T wave than do-nothing.

### 5.3 The action, applied on the embedding manifold

**Method.** To separate the action from the state map, only the action-induced displacement is added to $h_t$:
$\hat h = h_t + [\mathrm{Dyn}(h_t, a) - \mathrm{Dyn}(h_t, 0)]$, with $a$ the true action or the decoy.

**Results.** Latent MSE is over all pairs. The gaps are on active pairs: the error of $h_t$ minus that of $\hat h$
(positive means the action helps), and for the last column the error of the decoy displacement minus that of the
true one.

| encoder | latent MSE: $h_t$ | latent MSE: $\hat h$ | beat gap vs $h_t$ | HR gap vs $h_t$ (bpm) | beat gap, true vs decoy |
|---|---:|---:|---:|---:|---:|
| `dyn` | 0.2364 | 0.2327 | −0.0046 [−0.0062, −0.0030] | −1.30 [−1.60, −1.00] | +0.0033 [+0.0013, +0.0054] |
| `multistep` | 0.1999 | 0.1964 | −0.0009 [−0.0021, +0.0003] | −0.45 [−0.63, −0.26] | +0.0041 [+0.0027, +0.0056] |
| `regula` | 0.3729 | 0.3620 | +0.0011 [−0.0004, +0.0029] | −0.53 [−0.70, −0.33] | +0.0035 [+0.0019, +0.0052] |
| `joint` | 0.1880 | 0.1840 | −0.0020 [−0.0033, −0.0007] | −0.34 [−0.48, −0.21] | +0.0034 [+0.0021, +0.0048] |

**Conclusion.** The action displacement brings $h_t$ 1.6–2.9% closer to $h_{t+1}$ in latent MSE, and the true
action's displacement beats a decoy's by 0.003–0.004 beat MSE, so it carries some correct information. But it does
not make the decoded beat better than doing nothing, and it makes the heart rate slightly worse. Its latent gain
sits mostly in directions the decoder does not read as a change of the median beat.

Given that the action explains at most 0.1% of the displacement's variance (§5.1), even an ideal action model could
gain little per pair. The pair-averaged label effects of §6 are the fairer test of the action pathway.

---

## 6. Label effects

**Summary.** Cosines are over the 12 × 500 effect beats. "specific" is the cosine after removing, from both the
model and the real effects, the event-weighted mean effect of all labels of the same direction. "ΔHR r" is the
correlation between the model's and the real heart-rate change across the 40 onset entries. "size" is the median
of $\|\text{model}\| / \|\text{real}\|$ over all 76 entries. "cos(real, raw)" is the mean agreement between the
decoded and the raw real effect.

| encoder | onset: cos | onset: other | onset: specific | resol.: cos | resol.: other | resol.: specific | ΔHR r (onsets) | size model/real | cos(real, raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dyn` | +0.41 | +0.35 | +0.21 | +0.07 | +0.08 | −0.00 | +0.38 | 0.44 | 0.47 |
| `multistep` | +0.26 | +0.20 | +0.13 | +0.06 | +0.02 | +0.06 | +0.07 | 0.37 | 0.46 |
| `regula` | +0.25 | +0.12 | +0.17 | +0.07 | +0.02 | +0.05 | +0.00 | 0.44 | 0.56 |
| `joint` | +0.28 | +0.21 | +0.12 | +0.03 | −0.00 | +0.03 | +0.39 | 0.37 | 0.52 |
| `disp_mlp` | +0.20 | +0.16 | +0.11 | +0.09 | +0.02 | +0.09 | +0.31 | 0.22 | 0.59 |

**Selected labels:** cosine of the model effect with the real decoded effect; n is the number of test pairs with
the flip.

| label | direction | n | `dyn` | `multistep` | `regula` | `joint` | `disp_mlp` |
|---|---|---:|---:|---:|---:|---:|---:|
| I48 atrial fibrillation / flutter | onset | 354 | +0.87 | +0.70 | +0.47 | +0.38 | +0.45 |
| I50 heart failure | onset | 416 | +0.92 | +0.87 | +0.63 | +0.73 | +0.61 |
| I25 chronic ischaemic heart disease | onset | 400 | +0.74 | +0.80 | +0.34 | +0.57 | +0.34 |
| I27 other pulmonary heart disease | onset | 225 | +0.87 | +0.80 | +0.71 | +0.62 | +0.32 |
| I95 hypotension | onset | 367 | +0.72 | +0.03 | +0.67 | +0.75 | +0.30 |
| I10 essential hypertension | onset | 659 | +0.13 | −0.10 | +0.11 | −0.29 | −0.37 |
| I48 atrial fibrillation / flutter | resolution | 250 | +0.41 | +0.76 | +0.04 | −0.09 | +0.01 |
| I10 essential hypertension | resolution | 777 | +0.27 | +0.08 | −0.18 | +0.04 | −0.28 |

**Heart-rate change (bpm).** "real" is measured on the median beats' RR intervals; "real, decoded" is its range
across the five decoders; the encoder columns are the model effects.

| label | direction | real | real, decoded | `dyn` | `multistep` | `regula` | `joint` | `disp_mlp` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| I48 | onset | +14.7 | +11.6 to +13.3 | +7.7 | +2.1 | −1.0 | +2.6 | +3.4 |
| I48 | resolution | −7.2 | −6.9 to −4.4 | +1.5 | −1.0 | −2.0 | −0.3 | −1.6 |
| I50 | onset | +4.8 | +4.2 to +4.6 | +2.8 | +2.0 | −0.8 | +0.5 | +2.0 |

**Agreement between models.** The model effects of the same label agree across the four next-state models at cosine
0.31–0.37; `disp_mlp`'s agree with theirs at only 0.06–0.13. The real decoded effects are stable across decoders
(0.51–0.69, each against `dyn`'s decoder), so the reference is not an artifact of one decoder.

**Observations.**

- **Onsets point the right way on average, but mostly along a shared direction.** A model effect is nearly as
  similar to other labels' real onset effects (0.12–0.35) as to its own (0.20–0.41); the label-specific agreement is
  0.11–0.21. In `effects_dyn.png`, the shared component is mainly a lower QRS amplitude in II and V5.
- **Best cases:** `dyn` on I50, I48, I27 and I25 onsets (0.74–0.92) and `multistep` on I50, I27 and I25
  (0.80–0.87). Each of these also matches other onsets' real effects at 0.26–0.45.
- **Size:** model effects are 22–44% of the real decoded effects.
- **Heart rate:**
  - Atrial fibrillation onset raises the real rate by 14.7 bpm; the models give at most 7.7 (`dyn`), and `regula`
    lowers it.
  - Across onset labels, only `dyn`, `joint` and `disp_mlp` track the real rate changes at all (r 0.31–0.39).
  - `dyn` raises the rate on 89% of resolution entries, including atrial fibrillation resolution (+1.5 bpm against
    −7.2 real). Its rate response is closer to "any change, faster" than to a label-specific one.
- **Resolutions are not modelled** (cosine 0.03–0.09, label-specific about 0), even where the real change is
  clear. Atrial fibrillation resolution lowers the real rate by 7.2 bpm, and its decoded real effect matches the
  raw-beat effect at cosine 0.75–0.83.

---

## 7. Implications

1. **The embedding-space diagnostic needs a shrinkage baseline.** `tab:embedding-mse` credits Dyn for beating the
   do-nothing baseline (e.g. 0.147 vs 0.169 for the standard checkpoint). On the checkpoints tested here,
   $\mu + \beta(h_t - \mu)$ with a single $\beta$ beats Dyn in that metric. It should be added, with $\beta$ tuned on
   val, before the do-nothing gap is read as learned dynamics.
2. **The planning failure is not only a scoring problem.** Per pair, the true action barely beats a same-count decoy
   (51–53% of untied pairs), which is what CEM needs. Beyond that, the forward prediction itself is not a better
   next ECG than the current one, for any action.
3. **The Planning "simulator check" reading is not supported for these checkpoints.** The data supports at most
   that pair-averaged onset effects point roughly the right way, mostly along a shared direction, at 22–44% of the
   real size.
4. **Future forward objectives should be checked through the decoder.** Two cheap indicators of a distorting state
   map are the decoded heart-rate bias of $\mathrm{Dyn}(h_t, 0)$ and the correlation of its decoded rate with that
   of $h_t$. Both take seconds with the saved decoders.
5. **The encoder limits what a forward model can simulate.** Next-state forward training removes part of the heart
   rate and of the V2–V3 morphology from $h$. That limits how well any forward model on such an encoder can simulate
   rate changes such as atrial fibrillation onset.

---

## 8. Caveats

- **Decoder capacity.** The decoder is a two-layer MLP. A stronger one would raise every $R^2$, but the gaps between
  encoders are large relative to the training curve's remaining slope.
- **What median beats miss.** Beat-to-beat events other than the rate and RR irregularity (ectopic beats, pauses,
  AV-conduction timing) are not targets.
- **Observational real effects.** The real effect of a label includes the other diagnoses that flip in the same pairs
  (2.5 flips per active pair), while the model effect is a single-label intervention. The two estimands differ.
- **$\beta$ chosen on test** (§5.1).
- **§5 uses test pairs only.** Whether $c$ and $R$ also appear on training pairs, that is, whether they are learned
  behaviour or an effect of the test-time distribution shift, was not checked.
- **One seed.** One decoder seed, and one decoy draw per pair.
- **`disp_mlp`** is the MLP action-only checkpoint, from epoch 0. The linear action-only model was not run.

---

## 9. Reproduction

```bash
# gorgonoid/decoder.sh (job 52836, gorgona5). The first run builds the median-beat cache (~24 min, 16 workers)
# and caches each encoder's embeddings; both live next to the Lance table on that node's /scratch.
python anatomie/signal_decoder.py \
    --ckpt random=random --ckpt inverse=archive/inverse.ckpt \
    --ckpt dyn=archive/pretrain_val.ckpt --ckpt multistep=archive/pretrain_multistep0.ckpt \
    --ckpt regula=archive/pretrain_regula.ckpt --ckpt joint=archive/joint_ctx.ckpt \
    --ckpt disp_lin=runs/runs/20261002/224321/bb6b00025771/checkpoints/pretrain_displacement_mlp_.ckpt \
    --out-dir outputs/signal_decoder
```

The last line is as the job ran it; `disp_lin` there is the MLP checkpoint.

**Outputs** in `outputs/signal_decoder/`:

- `signal_decoder.json`: every number in §3, §4.1 and §6, and the paired gaps of §4.2;
- `decoder_<name>.pt`: decoder weights and standardization;
- `effects_<name>.npz` and `.png`: label effects;
- `pairs_<name>.png`: example test pairs.

**Caches** on gorgona5: `/scratch/josefernandes/data/median_beats/` (~10 GB) and
`/scratch/josefernandes/data/embeddings/`.

**Follow-up analyses.** §5 and the tie count in §4.2 were computed with ad-hoc scripts that are not in the repo.
They rebuild the test pairs' beats and embeddings with `median_beats.median_beats` and `encoders.load_encoder`, load
`decoder_<name>.pt` into `signal_decoder.BeatDecoder` / `Readout`, and evaluate the estimates defined in §5.

---

## Figures

Local only: `outputs/` is gitignored.

![dyn: example active test pairs](../outputs/signal_decoder/pairs_dyn.png)

*`dyn`, four random active test pairs, leads II, V1 and V5. The traces are real $X_t$ (grey), real $X_{t+1}$
(black), decoded $h_{t+1}$ (green, the ceiling), decoded $h_t$ (orange, do-nothing) and decoded
$\mathrm{Dyn}(h_t, a_t)$ (blue). In three of the four pairs, Dyn's beat has a smaller QRS and an earlier T wave than
do-nothing.*

![dyn: label effects](../outputs/signal_decoder/effects_dyn.png)

*`dyn`, the four label/direction entries with the most events. The traces are the real change on the real beats
(black), the real change decoded (green), and the model's single-label effect (blue).*

The same figures exist for the other forward models: `pairs_{multistep,regula,joint,disp_lin}.png` and
`effects_{multistep,regula,joint,disp_lin}.png`.
