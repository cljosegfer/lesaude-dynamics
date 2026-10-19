# CNN vs ViT Encoders under Forward Pretraining — Report

**Repo:** lesaude-dynamics · **Scope:** `papel/metodo.tex` (Dynamics Model) · **Date:** 2026-10-14
**Code:** §1 [`anatomie/memorization.py`](memorization.py); §2 [`anatomie/linear_probe.py`](linear_probe.py); §3
[`anatomie/cem_test.py`](cem_test.py). Encoders and forward models load through [`anatomie/encoders.py`](encoders.py).
**Outputs:** `outputs/memorization/`, `outputs/linear_probe/` and `outputs/cem_test/` (gitignored): `results.json` and
a figure each

## TL;DR

- **With the same forward-pretraining recipe, the CNN encoder memorizes its training pairs and the ViT encoder does
  not.** A membership test on the final embedding h separates training patients' pairs from held-out patients'
  pairs with AUROC 0.930 for the ResNet1d and 0.518 for the ViT1d. Their random-init controls score 0.513 and 0.508.
- **The CNN memorizes specific recordings, not only patients.**
  - A trained ECG sits 7× closer to the patient's next-stay ECG than a held-out patient's ECG does (relative
    distance 0.046 against 0.339).
  - It also sits closer than a never-trained ECG from the same stay (recording test 0.868).
  - All of this is in the last stage: stage 4 and h score 0.92–0.93, stage 3 0.61, and earlier stages match the
    control.
- **CNN memorization follows exposure.**
  - It rises from 0.867 for ECGs in one training pair per epoch to 0.984 for ECGs in twelve.
  - With k=1 pairs, where every ECG gets 1–2 pairs per epoch, it is flat across patients.
  - It had not started by epoch 5 (0.528).
- **The ViT's encoder stays clean even after 405 epochs on k=1 pairs (0.518), but its forward model memorizes the
  transitions.** On training transitions the true action removes more of the prediction error than on held-out ones:
  AUROC 0.641 with multistep pairs and 0.832 with k=1 pairs. The smaller encoder moves the memorization into the
  forward model; it does not remove it.
- **At h, the paper's readout, a linear probe finds no difference between CNN and ViT, and neither beats a
  random-init ResNet.** Test macro AUROC is 0.669 for both, against 0.677 for the random-init ResNet.
- **The diagnostic information sits in the middle layers, and there the CNN is better.** Its best layer reaches 0.710
  against the ViT's 0.681, with non-overlapping intervals. Forward pretraining improves the middle layers over random
  init by about +0.03 (CNN) and +0.04 (ViT).
- **The CNN's loss from stage 3 to h is not caused by memorization.** It is as large at epoch 5, before any
  memorization (0.707 → 0.671), as at epoch 99 (0.713 → 0.670): the last stage specializes to the forward loss.
- **No sign that labels leak into the encoder through the action.** Only the supervised encoder probes better on ECGs
  it trained on (+0.021); every forward and inverse encoder is within the range of the random-init controls.
- **CEM planning fails with either encoder.** On the 8,062 test pairs, the paper's CEM readout reaches a soft AUROC of
  0.470 (CNN) and 0.519 (ViT), against 0.750 for carry-forward. The search is exploited on every forward checkpoint:
  4–12 onsets per pair, against 0.67 in the ground truth.
- **With either encoder, the forward energy carries evidence only about training transitions.** The test ranks the
  true labels against decoys with the same number of flips.
  - On training pairs the CNN scores 0.722 and the ViT 0.723.
  - On held-out pairs they score 0.524 and 0.563, and the ViT's score is fully explained by h_t alone.
  - On held-out pairs the ViT's true action is worse than doing nothing (0.384).
- **A per-label readout of the same energy beats carry-forward, but only about 0.02 of the gain comes from the
  outcome ECG.** It scores 0.772 for the CNN and 0.781 for the ViT; the rest is read from h_t. Every forward
  checkpoint stays below the Inverse model (0.827), and the two architectures do not differ (paired +0.000).
- **Swapping the encoder moves the memorization, it doesn't fix planning.** The ViT removes the encoder-level
  collapse, but Dyn memorizes the training transitions with either encoder. The planning failure lies in how the
  forward model generalizes.
- **Caveat:** the ViT is also 4.6× smaller (0.87M against 3.98M encoder parameters). These results do not yet
  separate architecture from capacity.

---

## Setup

All runs use `scripts/pretrain.py` with lr 1e-3 and batch 1024, and are read at `last.ckpt` unless stated.

| Name | Encoder | Params | Pairs | Epochs | λ | wd | Predictor | Run (`runs/runs/…`) |
|---|---|---|---|---|---|---|---|---|
| **CNN** (matched) | ResNet1d | 3.98M | multistep k≤12 | 100 | 0.05 | 0.1 | 512 | `20260930/084313/e9ebd97377ce` |
| **ViT** (matched) | ViT1d 128×4 | 0.87M | multistep k≤12 | 100 | 0.05 | 0.1 | 512 | `20261006/044610/064d3fbda75f` |
| CNN, epoch 5 | ResNet1d | 3.98M | multistep k≤12 | 6 | 0.05 | 0.1 | 512 | same run, best-val/loss checkpoint |
| CNN, λ 0.005 / wd 0.01 | ResNet1d | 3.98M | multistep k≤12 | 56 | 0.005 | 0.01 | 512 | `20260930/062649/71f4c433519a` |
| CNN, k=1 | ResNet1d | 3.98M | k=1 cross-stay | 100 | 0.005 | 0.01 | 512 | `20260909/123412/ac1e0c66cc08` |
| ViT, k=1 | ViT1d 128×4 | 0.87M | k=1 cross-stay | 405 | 0.05 | 0.1 | 512 | `20261006/134920/e98705006684` |
| ViT, k=1, predictor 128 | ViT1d 128×4 | 0.87M | k=1 cross-stay | 405 | 0.05 | 0.1 | 128 | `20261006/142437/c68ff538b143` |
| Controls | random-init ResNet1d and ViT1d (seed 0) | 3.98M / 0.87M | — | — | — | — | — | — |

The two matched runs differ only in the encoder. The k=1 ViT runs have 405 epochs so that they see about as many
batches as the multistep runs. The secondary CNN and ViT runs are not matched to each other.

---

## 1. Memorization

### Test

A membership-inference design. Members and non-members are drawn the same way, so any statistic that separates them
measures memorization, with chance at exactly 0.5.

- **Anchor:** a stay s that has a second ECG and a following stay. Each anchor gives three ECGs:
  - **A**, the first ECG of stay s. The pair A → B is a k=1 cross-stay training pair, which every checkpoint here
    trained on.
  - **A′**, the second ECG of stay s. No cross-stay checkpoint trains on it.
  - **B**, the first ECG of stay s+1.
- **Members:** 5,000 training-fold patients. **Non-members:** all 2,518 patients of folds 18–19. Each contributes one
  random anchor, so observations are independent.
- **Membership is verified per checkpoint** against its own pair file. All members' A → B pairs are in it, and no A′
  and no non-member pair is (1.000 / 0.000 / 0.000 for every checkpoint).
- **The random-init encoders have no members, so they are the control.** They score 0.48–0.53 throughout, which
  confirms the two populations are matched.

Encoder tests, at every readout layer. Distances are relative: a pair's squared distance divided by the mean squared
distance from the same ECG to other patients' B in its own population. This removes population-level differences
in scale or centring.

| Test | Score | What a high AUROC means |
|---|---|---|
| pair | closeness of A to B | trained pairs sit closer: the recording and/or the patient is memorized |
| patient | closeness of A′ to B | a never-trained ECG of a training patient also sits closer: the patient is memorized |
| recording | closeness of A to B relative to A′ to B | the trained ECG beats an untrained ECG of the same stay, beyond what non-members show: the specific recording is memorized |
| spread | distance of A from the pooled centre | trained recordings occupy more of the embedding space than unseen ones |
| dose | the pair test at h, by the patient's number of stays | memorization grows with how often the ECG was trained on |

Forward-model tests, on the A → B pairs with their true action a:

| Test | Score | What a high AUROC means |
|---|---|---|
| total | low E(a) / E_nothing | the forward loss itself separates training pairs |
| zero | low E(0) / E_nothing | Dyn fits training pairs better even without the action |
| action | 1 − E(a)/E(0), active pairs | the true action removes more of the error on training transitions: the action effects are memorized |

Here E(a) is the squared error of Dyn(h_A, a) against h_B, and E_nothing is that of h_A itself.
- **"total" and "zero"** divide by E_nothing, which an encoder that collapsed its member pairs drives to about zero.
  Read them only for encoders whose pair test is near 0.5, which here means the ViTs.
- **Population aggregates:**
  - The action's R² on the residual h_B − Dyn(h_A, 0).
  - Its cosine with that residual.
  - The best one-parameter shrinkage toward the mean.

  These are as in [`dyn_decompose.py`](dyn_decompose.py).

All AUROCs come with DeLong 95% confidence intervals.

### Results

![Memorization: CNN vs ViT](../outputs/memorization/memorization.png)

*(a) Pair test at each readout layer. (b) Pair test at h by the patient's number of stays, with the training pairs
per epoch that contain the anchor ECG. (c) Forward-model tests. (d) Ratio of train to val forward loss per epoch, from
the runs' logs. Gray lines are the random-init controls.*

**Table 1. Encoder tests at h.** AUROC [95% CI]. "Relative A–B" is the mean pair distance relative to unrelated
ECGs (0 = same point, 1 = no closer than another patient's ECG).

| Encoder | Params | Relative A–B, members / non-members | pair | patient | recording | spread |
|---|---|---|---|---|---|---|
| CNN, random init | 3.98M | 0.552 / 0.559 | 0.513 [0.499–0.526] | 0.511 [0.498–0.525] | 0.499 [0.485–0.512] | 0.500 [0.487–0.514] |
| ViT, random init | 0.87M | 0.401 / 0.409 | 0.508 [0.494–0.521] | 0.500 [0.487–0.514] | 0.502 [0.489–0.516] | 0.498 [0.484–0.512] |
| **CNN** (matched) | 3.98M | **0.046** / 0.339 | **0.930** [0.924–0.935] | **0.741** [0.730–0.752] | **0.868** [0.860–0.876] | **0.751** [0.740–0.762] |
| **ViT** (matched) | 0.87M | 0.258 / 0.273 | 0.518 [0.504–0.532] | 0.510 [0.496–0.524] | 0.505 [0.492–0.519] | 0.507 [0.494–0.521] |
| CNN, epoch 5 | 3.98M | 0.226 / 0.252 | 0.528 [0.514–0.542] | 0.517 [0.503–0.531] | 0.510 [0.496–0.524] | 0.529 [0.515–0.543] |
| CNN, λ 0.005 / wd 0.01 | 3.98M | 0.024 / 0.310 | 0.960 [0.956–0.965] | 0.708 [0.696–0.720] | 0.908 [0.901–0.914] | 0.710 [0.699–0.722] |
| CNN, k=1 | 3.98M | 0.029 / 0.233 | 0.944 [0.939–0.949] | 0.684 [0.671–0.696] | 0.926 [0.919–0.932] | 0.626 [0.613–0.639] |
| ViT, k=1, 405 epochs | 0.87M | 0.253 / 0.271 | 0.518 [0.504–0.532] | 0.512 [0.499–0.526] | 0.503 [0.489–0.517] | 0.506 [0.492–0.520] |
| ViT, k=1, predictor 128 | 0.87M | 0.273 / 0.289 | 0.520 [0.506–0.534] | 0.505 [0.491–0.519] | 0.514 [0.500–0.527] | 0.504 [0.490–0.518] |

**Table 2. Pair test by readout layer** (point estimates; CIs are ±0.007–0.014). CNN stages are stem, layer1–4; ViT
stages are patch embedding, block1–4.

| Encoder | stem / patch | stage 1 | stage 2 | stage 3 | stage 4 | h |
|---|---|---|---|---|---|---|
| CNN, random init | 0.506 | 0.505 | 0.504 | 0.503 | 0.512 | 0.513 |
| ViT, random init | 0.499 | 0.503 | 0.503 | 0.504 | 0.506 | 0.508 |
| **CNN** (matched) | 0.500 | 0.506 | 0.513 | 0.607 | **0.924** | **0.930** |
| **ViT** (matched) | 0.497 | 0.513 | 0.513 | 0.513 | 0.519 | 0.518 |
| CNN, epoch 5 | 0.503 | 0.509 | 0.514 | 0.517 | 0.526 | 0.528 |
| CNN, λ 0.005 / wd 0.01 | 0.501 | 0.508 | 0.509 | 0.537 | 0.955 | 0.960 |
| CNN, k=1 | 0.501 | 0.508 | 0.516 | 0.546 | 0.933 | 0.944 |
| ViT, k=1, 405 epochs | 0.498 | 0.511 | 0.513 | 0.514 | 0.521 | 0.518 |
| ViT, k=1, predictor 128 | 0.496 | 0.512 | 0.513 | 0.513 | 0.522 | 0.520 |

**Table 3. Dose: pair test at h by the patient's number of stays.** In parentheses: training pairs per epoch that
contain the anchor ECG, in the checkpoint's own pair file. Members / non-members per bucket: 1,260 / 629, 1,670 /
868, 1,231 / 627, 839 / 394.

| Encoder | 2 stays | 3–4 stays | 5–8 stays | ≥9 stays |
|---|---|---|---|---|
| CNN, random init | 0.502 [0.474–0.530] | 0.524 [0.501–0.548] | 0.505 [0.477–0.532] | 0.513 [0.479–0.548] |
| ViT, random init | 0.477 [0.450–0.505] | 0.529 [0.505–0.553] | 0.509 [0.481–0.537] | 0.509 [0.475–0.543] |
| **CNN** (matched) | 0.867 [0.851–0.884] (1.0) | 0.926 [0.916–0.936] (2.4) | 0.965 [0.957–0.973] (5.1) | 0.984 [0.976–0.992] (12.3) |
| **ViT** (matched) | 0.500 [0.473–0.527] (1.0) | 0.524 [0.500–0.548] (2.4) | 0.493 [0.465–0.521] (5.1) | 0.566 [0.531–0.601] (12.3) |
| CNN, epoch 5 | 0.504 [0.476–0.532] (1.0) | 0.533 [0.509–0.557] (2.4) | 0.505 [0.477–0.533] (5.1) | 0.589 [0.556–0.623] (12.3) |
| CNN, λ 0.005 / wd 0.01 | 0.921 [0.908–0.934] (1.0) | 0.957 [0.949–0.965] (2.4) | 0.985 [0.981–0.990] (5.1) | 0.990 [0.984–0.996] (12.3) |
| CNN, k=1 | 0.940 [0.930–0.951] (1.0) | 0.937 [0.927–0.946] (1.6) | 0.947 [0.937–0.958] (1.8) | 0.959 [0.947–0.971] (1.9) |
| ViT, k=1, 405 epochs | 0.503 [0.476–0.531] (1.0) | 0.530 [0.506–0.553] (1.6) | 0.504 [0.476–0.532] (1.8) | 0.533 [0.498–0.568] (1.9) |
| ViT, k=1, predictor 128 | 0.498 [0.470–0.525] (1.0) | 0.534 [0.511–0.558] (1.6) | 0.505 [0.477–0.533] (1.8) | 0.544 [0.510–0.579] (1.9) |

**Table 4. Forward model.** Errors are relative to doing nothing, as members / non-members, aggregated over pairs.
Then the three membership AUROCs.

| Encoder | E(a) | E(0) | Shrink | Action R² | cos(effect, residual) | total | zero | action |
|---|---|---|---|---|---|---|---|---|
| **CNN** (matched) | 0.604 / 0.965 | 0.878 / 0.968 | 0.970 / 0.835 | +0.418 / +0.004 | +0.233 / +0.078 | 0.533 [0.520–0.546]* | 0.492 [0.479–0.505]* | 0.633 [0.617–0.648] |
| **ViT** (matched) | 0.803 / 0.927 | 0.872 / 0.906 | 0.871 / 0.869 | +0.114 / −0.034 | +0.222 / +0.075 | 0.599 [0.586–0.612] | 0.549 [0.535–0.562] | 0.641 [0.624–0.657] |
| CNN, epoch 5 | 0.902 / 0.912 | 0.925 / 0.925 | 0.890 / 0.880 | +0.034 / +0.021 | +0.093 / +0.083 | 0.507 [0.493–0.521] | 0.506 [0.492–0.519] | 0.510 [0.493–0.527] |
| CNN, λ 0.005 / wd 0.01 | 0.640 / 0.989 | 0.892 / 0.989 | 0.992 / 0.854 | +0.383 / +0.000 | +0.208 / +0.029 | 0.535 [0.522–0.547]* | 0.508 [0.496–0.521]* | 0.588 [0.572–0.604] |
| CNN, k=1 | 0.243 / 0.999 | 0.927 / 1.007 | 0.982 / 0.892 | +0.809 / +0.011 | +0.545 / +0.125 | 0.847 [0.838–0.856]* | 0.722 [0.711–0.734]* | 0.821 [0.809–0.833] |
| ViT, k=1, 405 epochs | 0.637 / 0.999 | 0.868 / 0.891 | 0.874 / 0.871 | +0.374 / −0.171 | +0.453 / +0.085 | 0.725 [0.714–0.737] | 0.545 [0.531–0.559] | 0.832 [0.820–0.844] |
| ViT, k=1, predictor 128 | 0.687 / 0.953 | 0.863 / 0.870 | 0.866 / 0.863 | +0.287 / −0.133 | +0.357 / +0.086 | 0.656 [0.644–0.669] | 0.518 [0.504–0.532] | 0.766 [0.752–0.780] |

\* The encoder collapsed the member pairs, so the do-nothing denominator is close to zero; don't read these as forward-model memorization.

**Table 5. Ratio of train to val forward loss during training,** from the runs' wandb logs (1.0 = no gap).

| Run | epoch 1 | 5 | 10 | 25 | 50 | last |
|---|---|---|---|---|---|---|
| **CNN** (matched) | 1.05 | 0.73 | 0.58 | 0.29 | 0.15 | 0.05 (ep 99) |
| **ViT** (matched) | 1.10 | 1.05 | 0.96 | 0.86 | 0.75 | 0.66 (ep 99) |
| CNN, λ 0.005 / wd 0.01 | 0.98 | 0.42 | 0.28 | 0.11 | 0.05 | 0.04 (ep 55) |
| CNN, k=1 | 0.98 | 1.13 | 0.97 | 0.60 | 0.18 | 0.04 (ep 99) |
| ViT, k=1, 405 epochs | 0.77 | 0.98 | 0.91 | 0.95 | 0.93 | 0.68 (ep 404) |
| ViT, k=1, predictor 128 | 0.77 | 1.00 | 0.89 | 0.97 | 0.94 | 0.76 (ep 404) |

### Interpretation

1. **The CNN encoder memorizes specific training recordings, in its last stage.**
   - Trained pairs collapse: relative distance 0.046 for members against 0.339 for non-members.
   - The trained ECG is closer to B than an untrained ECG of the same stay (recording 0.868).
   - Stages 1–2 match the control, stage 3 begins (0.607) and stage 4 carries it (0.924).
   - In the less regularized CNN (λ 0.005), stage 3 shows almost none (0.537). Stronger regularization lowers h only
     a little (0.930 against 0.960), with 44 more epochs.
2. **The CNN also partly memorizes the patients.**
   - A never-trained ECG of a training patient sits closer to B than a held-out patient's ECG does (patient 0.741;
     relative distance 0.152 against 0.330).
   - Trained recordings are spread further from the centre (spread 0.751; 45% larger mean squared distance).
   - So unseen ECGs, including new patients', fall into a compressed region of h.
3. **The CNN's memorization is driven by exposure.**
   - With multistep pairs, the pair test rises with the number of training pairs that contain the ECG per epoch:
     0.867 at 1, 0.984 at 12.3.
   - With k=1 pairs, every ECG is in 1–2 pairs per epoch whatever the patient's number of stays, and the test is flat
     at 0.94–0.96. So the dependence is on exposure, not on the kind of patient.
   - It had not begun at epoch 5 of the matched run (0.528 at h; 0.589 for patients with ≥9 stays). The train/val
     loss ratio then falls steadily from 0.73 at epoch 5 to 0.05 at epoch 99.
4. **The ViT encoder does not memorize, even under heavy repetition.**
   - At every layer, every test is within 0.02 of its control, including after 405 epochs on k=1 pairs, where each
     training pair was seen 405 times.
   - The only trace is the most exposed group in the multistep run: patients with ≥9 stays, 12.3 pairs per epoch, at
     0.566 [0.531–0.601] against 0.509 for the control.
5. **The ViT's forward model memorizes instead.**
   - With multistep pairs, the true action explains +0.114 of the residual on training transitions and −0.034 on
     held-out ones.
   - Per pair, the action test is 0.641, the same as the CNN's 0.633.
   - Repetition strengthens it: on k=1 pairs for 405 epochs the action test reaches 0.832 (+0.374 against −0.171), and
     the forward loss itself separates members (total 0.725).
   - A forward model with 3.8× fewer parameters (hidden size 128: 0.28M against 1.07M) only lowers it to 0.766.
   - The ViT's train/val loss ratio ends at 0.66, not the CNN's 0.05, but the gap that remains is in the forward
     model.
6. **What this means for the dynamics objective.**
   - Replacing the CNN with a smaller ViT removes the encoder-level failure: the collapse of training pairs, the
     compression of unseen ECGs, and the dependence on exposure.
   - It does not make the forward model generalize. On held-out transitions the ViT's action effect still explains
     nothing (R² −0.034) and its forward loss is worse than shrinkage (0.927 against 0.869).
   - Whatever the forward model learns about actions, it learns about the training transitions.

### Caveats

- **Capacity confound.** The ViT has 0.87M encoder parameters and the CNN 3.98M, so "ViT does not memorize" may be
  "a smaller encoder does not memorize". A capacity-matched control would separate the two: a ResNet1d of about
  1M (for example base_channels 32, or without layer4), or ViT 192×8 (3.67M).
- **One seed per configuration.** The secondary runs differ from the matched ones in several settings at once (λ, wd,
  pairs, epochs). Use them for trends, not for single comparisons.
- **The control floor is 0.48–0.53, not exactly 0.5.** The 3–4-stays bucket reads about 0.525 for every encoder,
  both controls included, which points to a small population difference in that bucket. Compare each result with
  its architecture's control, not with 0.5.
- **The "total" and "zero" tests are invalid for collapsed encoders** (marked * in Table 4). The aggregate errors
  weight pairs by their do-nothing distance and the per-pair AUROCs weight them equally, so the two can disagree: the
  CNN's per-pair action test matches the ViT's, but its aggregate action R² is much larger, which concentrates its
  memorized action effects in the pairs with large changes.
- **Fold 18 selected one checkpoint.** It is part of the non-member pool and selected the CNN epoch-5 checkpoint (best
  val/loss), which could make that checkpoint's memorization look slightly smaller. All other results use
  `last.ckpt`, which no validation metric selected.
- **Only the k=1 hop is tested.** The multistep runs also trained on longer hops between stays.
- **Figure colors** are slots 1–2 of the dataviz skill's documented palette, recorded there as validated for all
  pairs in both modes. The skill's validator script could not run on the compute node (no `node`).

### Reproduce

```bash
python anatomie/memorization.py \
    --ckpt cnn_random=random --ckpt vit_random=random-vit \
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \
    --ckpt cnn_ep5=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/pretrain_multistep_.ckpt \
    --ckpt cnn_lowreg=runs/runs/20260930/062649/71f4c433519a/checkpoints/last.ckpt \
    --ckpt cnn_k1=runs/runs/20260909/123412/ac1e0c66cc08/checkpoints/last.ckpt \
    --ckpt vit_k1=runs/runs/20261006/134920/e98705006684/checkpoints/last.ckpt \
    --ckpt vit_k1_p128=runs/runs/20261006/142437/c68ff538b143/checkpoints/last.ckpt \
    --out outputs/memorization/results.json --fig outputs/memorization/memorization.png
```

About 5 minutes on one RTX 3090 Ti with the data on local `/scratch` (22,554 ECGs, nine encoders, one pass).
`--plot-only` redraws the figure from `results.json`.

---

## 2. Linear probe

### Test

The paper's triage evaluation, run offline and identically for every encoder:

- **Probe:** logistic regression for the 76 labels on the standardized features of one readout layer.
  - Fitted with full-batch L-BFGS.
  - The L2 penalty is chosen on val per layer, from {0, 1e-6, 1e-5, 1e-4, 1e-3}.
  - Deterministic.
- **Training data:** 100,000 first-of-stay ECGs of training-fold patients (60,279 patients).
- **Validation:** all 14,456 first-of-stay ECGs of fold 18. They choose the penalty and the best readout layer.
- **Test:** all 14,237 first-of-stay ECGs of fold 19, which is `scripts/evaluate.py`'s triage test set.
  - Macro AUROC over the classes with both outcomes.
  - 1,000 bootstrap resamples of ECGs, as `evaluate.py` does.
- **Calibration check:** the supervised CNN's frozen features with a fresh probe give 0.741 at h, matching its own
  finetuned evaluation (0.736).

**Why not the earlier probes.** The first version of this script, like `anatomie/layer_readout.py`, trained the probe
with mini-batch AdamW and early stopping. That had not converged:
- Val AUROC was still rising after 8,000 steps.
- It plateaued below the L-BFGS optimum, for example 0.676 against 0.699 for CNN stage 3.

The best penalty also differs between encoders: the trained ViT's h prefers none, CNN stage 3 prefers 1e-4. So it is
chosen per layer. Probe numbers in earlier notes (for example 0.652 as the best frozen probe, 0.626 for random init)
come from the undertrained protocol and are superseded by these.

**Seen vs unseen.** The forward objective conditions on y(t+1) − y(t), so the labels of training ECGs reach the
encoder; §1 also showed that the CNN places unseen ECGs in a compressed region. The probe is therefore scored on two
populations it never trained on, drawn the same way:
- **Seen:** 19,764 first-of-stay ECGs of 5,000 training-fold patients with ≥2 stays, held out from the probe. Every
  one is in the checkpoint's own training pairs (verified).
- **Unseen:** 10,841 first-of-stay ECGs of the 2,779 fold-19 patients with ≥2 stays.

The gap is AUROC(seen) − AUROC(unseen). Its interval comes from a patient-level bootstrap: an ECG-level one ignores
the correlation between a patient's ECGs, and in a dry run it produced a false positive for a random encoder. The
random-init encoders are the null control. The supervised CNN is the positive control: it trained on these ECGs'
labels.

**Reference encoders.** All three are ResNet1d:

| Encoder | Checkpoint | Objective and data | Epoch |
|---|---|---|---|
| Paper Dynamics | `archive/pretrain_val.ckpt` | Forward objective, k=1 within- and cross-stay pairs, λ 0.005 | 19 |
| Paper Inverse | `archive/inverse.ckpt` | Inverse objective, k=1 cross-stay pairs | 65 |
| Supervised | `archive/supervised_triage.ckpt` | Labels of training-fold first-of-stay ECGs | 33 |

### Results

![Linear probe: CNN vs ViT](../outputs/linear_probe/linear_probe.png)

*(a) Test macro AUROC of the probe at each readout layer, labeled at each matched encoder's val-selected layer. Gray
lines are the random-init controls: ResNet1d above, ViT1d below. (b) AUROC(seen) − AUROC(unseen) at h, for every
encoder.*

**Table 6. Probe summary.** Test macro AUROC [95% CI]. "Best layer" is chosen on val.

| Encoder | Test at h | Best layer | Test at best layer | Seen / unseen at h | Gap at h |
|---|---|---|---|---|---|
| CNN, random init | 0.677 [0.665–0.689] | stage 4 | 0.683 [0.671–0.697] | 0.651 / 0.665 | −0.015 [−0.037 to +0.006] |
| ViT, random init | 0.645 [0.633–0.657] | h | 0.645 [0.633–0.657] | 0.632 / 0.631 | +0.001 [−0.021 to +0.023] |
| **CNN** (matched) | **0.669** [0.657–0.681] | stage 2 | **0.710** [0.699–0.720] | 0.633 / 0.656 | −0.024 [−0.044 to −0.002] |
| **ViT** (matched) | **0.669** [0.656–0.682] | stage 3 | **0.681** [0.668–0.694] | 0.644 / 0.657 | −0.013 [−0.037 to +0.011] |
| CNN, epoch 5 | 0.672 [0.661–0.683] | stage 3 | 0.707 [0.696–0.719] | 0.659 / 0.662 | −0.003 [−0.025 to +0.018] |
| CNN, λ 0.005 / wd 0.01 | 0.659 [0.647–0.671] | stage 3 | 0.732 [0.721–0.742] | 0.635 / 0.649 | −0.014 [−0.035 to +0.009] |
| CNN, k=1 | 0.677 [0.665–0.689] | stage 3 | 0.727 [0.715–0.738] | 0.649 / 0.660 | −0.011 [−0.034 to +0.012] |
| ViT, k=1, 405 epochs | 0.658 [0.643–0.673] | stage 3 | 0.682 [0.667–0.696] | 0.635 / 0.648 | −0.013 [−0.039 to +0.013] |
| ViT, k=1, predictor 128 | 0.665 [0.653–0.677] | stage 2 | 0.691 [0.680–0.704] | 0.652 / 0.651 | +0.000 [−0.024 to +0.024] |
| Paper Dynamics (CNN) | 0.692 [0.682–0.704] | stage 3 | 0.735 [0.724–0.747] | 0.669 / 0.676 | −0.007 [−0.027 to +0.014] |
| Paper Inverse (CNN) | 0.716 [0.703–0.729] | stage 4 | 0.732 [0.720–0.742] | 0.683 / 0.696 | −0.013 [−0.036 to +0.009] |
| Supervised (CNN) | 0.741 [0.730–0.754] | stage 4 | 0.745 [0.734–0.757] | 0.752 / 0.730 | **+0.021** [+0.001 to +0.041] |

**Table 7. Test macro AUROC by readout layer** (point estimates; the val-selected layer in bold).

| Encoder | stem / patch | stage 1 | stage 2 | stage 3 | stage 4 | h |
|---|---|---|---|---|---|---|
| CNN, random init | 0.654 | 0.654 | 0.665 | 0.668 | **0.683** | 0.678 |
| ViT, random init | 0.505 | 0.617 | 0.627 | 0.631 | 0.634 | **0.644** |
| **CNN** (matched) | 0.671 | 0.680 | **0.709** | 0.713 | 0.682 | 0.670 |
| **ViT** (matched) | 0.500 | 0.652 | 0.669 | **0.680** | 0.682 | 0.668 |
| CNN, epoch 5 | 0.658 | 0.678 | 0.714 | **0.707** | 0.696 | 0.671 |
| CNN, λ 0.005 / wd 0.01 | 0.659 | 0.679 | 0.696 | **0.731** | 0.677 | 0.659 |
| CNN, k=1 | 0.670 | 0.677 | 0.704 | **0.724** | 0.697 | 0.681 |
| ViT, k=1, 405 epochs | 0.497 | 0.653 | 0.665 | **0.679** | 0.670 | 0.657 |
| ViT, k=1, predictor 128 | 0.509 | 0.661 | **0.690** | 0.693 | 0.685 | 0.667 |
| Paper Dynamics (CNN) | 0.673 | 0.685 | 0.706 | **0.734** | 0.705 | 0.693 |
| Paper Inverse (CNN) | 0.662 | 0.669 | 0.676 | 0.701 | **0.732** | 0.718 |
| Supervised (CNN) | 0.645 | 0.666 | 0.691 | 0.714 | **0.740** | 0.737 |

### Interpretation

1. **At h, the paper's readout, the matched CNN and ViT tie, and neither beats a random-init ResNet.**
   - Both score 0.669, against 0.677 for the random-init ResNet.
   - For the CNN, forward pretraining adds nothing at h (−0.008 against its random init).
   - The ViT gains +0.024 over its own random init, but random ViT features start much lower than random ResNet
     features (0.645 against 0.677): convolutions give a strong prior for ECGs even untrained.
2. **The diagnostic information sits in the middle layers, and there the CNN is better.**
   - Forward-pretrained CNNs reach 0.707–0.735 at stages 2–3, which is +0.03 to +0.05 over the random ResNet's best
     layer (0.683).
   - The ViTs reach 0.681–0.691 at stages 2–3, +0.04 over the random ViT's best (0.645).
   - For the matched pair, the CNN's best layer beats the ViT's: 0.710 [0.699–0.720] against 0.681 [0.668–0.694].
3. **The CNN loses its middle-layer advantage in the last stage, and memorization is not the main cause.**
   - From stage 3 to h, every forward CNN loses 0.04–0.07. The checkpoint at epoch 5, whose pair test in §1 was 0.528
     (no memorization), loses as much (0.707 → 0.671) as the matched one at epoch 99 (0.713 → 0.670, pair test 0.930).
   - The most memorized CNN (λ 0.005, pair test 0.960) loses the most (0.731 → 0.659), so memorization may add to
     the loss, but the last stage is mainly specializing to the forward loss.
   - Neither the random ResNet (0.683 → 0.678) nor the inverse or supervised CNNs, which peak at stage 4, show this
     drop.
   - The ViT's drop is smaller: 0.680 → 0.668, and 0.679 → 0.657 at k=1.
4. **There is no sign that labels leak into the encoder through the action.**
   - Every forward and inverse encoder's gap lies between −0.024 and +0.000, the same range as the random-init
     controls (−0.015 and +0.001).
   - Only the supervised encoder probes better on ECGs it trained on (+0.021 [+0.001 to +0.041]). So the test can
     detect memorized labels, and the label-derived action leaves no detectable trace.
   - The matched CNN's −0.024 is the only interval that excludes zero, but it is within 0.01 of its architecture's
     control (−0.015). It reflects the two populations, not the encoder.
5. **The matched multistep recipe is not the best CNN for probing.**
   - At its best layer the matched CNN (0.710) is below the paper's Dynamics encoder (0.735) and below the less
     regularized multistep (0.732) and k=1 (0.727) CNNs. The longer, more regularized multistep training did not
     help the probe.
   - The paper's Inverse encoder is the best self-supervised encoder at h (0.716), where its last stage is not
     trained to predict h(t+1).

### Caveats

- **Capacity confound** (as in §1). The ViT has 4.6× fewer encoder parameters. Its lower middle-layer ceiling (0.68
  against 0.71) could come from size rather than architecture.
- **Offline protocol.** This is not `scripts/finetune.py --freeze_encoder`'s online probe: that trains on all
  ECGs, including the 298,869 linked to no stay as all-negative. The absolute values are therefore not the paper's
  linear-probe rows, but every encoder here gets the identical protocol.
- **Penalty grid edges.** CNN stage 4 and h chose the largest penalty (1e-3); the convergence check found the val
  AUROC flat from 1e-3 to 1e-2 there. Some ViT layers chose no penalty, the other end.
- **Layer selection.** The best layer is chosen on val. For the matched CNN, val picked stage 2 (test 0.709), while
  stage 3 is slightly higher on test (0.713).
- **One draw of the probe-training set.** The probe itself is deterministic, but the 100,000 training ECGs are one
  random draw. The per-ECG test intervals do not include that variance.
- **The seen-vs-unseen populations differ slightly.** The random-init ResNet's gap is −0.015 (not significant), so
  read each encoder's gap against its architecture's control, not against zero.

### Reproduce

```bash
python anatomie/linear_probe.py \
    --ckpt cnn_random=random --ckpt vit_random=random-vit \
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \
    --ckpt cnn_ep5=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/pretrain_multistep_.ckpt \
    --ckpt cnn_lowreg=runs/runs/20260930/062649/71f4c433519a/checkpoints/last.ckpt \
    --ckpt cnn_k1=runs/runs/20260909/123412/ac1e0c66cc08/checkpoints/last.ckpt \
    --ckpt vit_k1=runs/runs/20261006/134920/e98705006684/checkpoints/last.ckpt \
    --ckpt vit_k1_p128=runs/runs/20261006/142437/c68ff538b143/checkpoints/last.ckpt \
    --ckpt paper_dyn=archive/pretrain_val.ckpt --ckpt paper_inv=archive/inverse.ckpt \
    --ckpt supervised=archive/supervised_triage.ckpt \
    --cache /scratch/josefernandes/cache/linear_probe_feats.pt \
    --out outputs/linear_probe/results.json --fig outputs/linear_probe/linear_probe.png
```

Embedding the 148,457 ECGs takes about 8 minutes on one RTX 3090 Ti with the data on local `/scratch`. The features
are cached (4.1 GB, float16), then each encoder takes about 3.5 minutes for probes and bootstraps. The script saves
after each encoder and skips finished ones when rerun, so an interrupted run resumes. `--plot-only` redraws the
figure.

## 3. CEM planning

### Test

The paper's planning procedure, plus two readouts of the same energy that need no search. Each runs on a forward
checkpoint's own encoder and Dyn.

- **Pairs:** the k=1 cross-stay pairs of `pairs.lance`, which `scripts/cem_planning.py` and
  `scripts/evaluate_inverse.py` also use.
  - **Test (non-members):** fold 19, 8,062 pairs of 2,779 patients; 4,129 pairs have a change.
  - **Members:** fold 17, 8,271 pairs of 2,729 patients. This is a training fold, built the same way as the test
    fold.
    - Every member pair is in every checkpoint's own training pairs, and no val or test pair is (1.000 / 0.000 /
      0.000, verified per checkpoint).
    - The multistep checkpoints also trained on the longer hops.
  - **Val:** fold 18, 8,244 pairs. They only choose the per-label readout's temperature.
- **Energy:** E(a) = mean_D ‖Dyn(h_t, Proj(a)) − h_t+1‖².
  - Only admissible actions are scored: onsets where y_t,i = 0, resolutions where y_t,i = 1.
  - The priors are the training base rates of `cem_planning.py`.

| Readout | What it does | Question |
|---|---|---|
| CEM | `cem_planning.py`'s `OnsetResolutionCEM` with plain E and its defaults (512 samples, 64 elites, 16 iterations). Contextualized AUROC of the soft readout (final marginals) and the hard readout (clip(y_t + a*)) | the paper's planning result |
| Decoys | No search. The true action is ranked by E against admissible decoys, using the families of `pretrain.py`'s online decoy probe. *Same-count:* 16 decoys per pair with the true numbers of onsets and resolutions on random labels. *Zero:* doing nothing, i.e. P(E(a_gt) < E(0)). Within-pair AUROC; chance is 0.5 | does E single out the true transition? |
| Per-label | The readout of [`flip_readout.py`](flip_readout.py): δ_i = E(0) − E(e_i) for the single admissible flip e_i of label i, then p_i = σ(logit π_i + δ_i/τ), read contextually. τ is chosen on val from multiples of the checkpoint's own δ spread, since embedding scales differ between backbones | the best a per-label posterior can read from E |

**Nothing-changed control.** Every decoy and per-label metric is repeated with the pair's own h_t as the target in
place of h_t+1. This keeps the patient and removes the change. What survives is read from h_t, i.e. which changes are
plausible for this patient, not from the outcome ECG. **Evidence** is the real score minus the nothing-changed score.

**Statistics.**
- Point estimates are on all pairs.
- 95% intervals come from a patient-level bootstrap (1,000 resamples, seed 42), because a patient's pairs share ECGs.
- Every metric and checkpoint uses the same resamples, so evidence and CNN − ViT differences are paired.

**References** on the same pairs and resamples:
- **Test:** carry-forward 0.750 [0.735–0.763] and the paper's Inverse 0.827 [0.811–0.843]. The paper's pair-level
  bootstrap gives means of 0.7515 and 0.8286.
- **Fold 17:** carry-forward 0.743 and Inverse 0.902. The Inverse also trained on fold 17 and scores 0.075 higher
  there than on test.

### Results

![CEM planning: CNN vs ViT](../outputs/cem_test/cem_test.png)

*(a) Test contextualized AUROC for three readouts: CEM's soft readout (squares), the per-label readout (filled
circles), and its nothing-changed control (open circles). Dashed line: carry-forward. Dotted line: the paper's
Inverse. (b) Within-pair AUROC of the true action against same-count decoys, on pairs with a change. Diamonds are
training pairs (fold 17), filled circles test pairs, and open circles test pairs scored against h_t. (c)
P(E(a_gt) < E(0)) on pairs with a change. Colored: the matched pair. Gray: the other forward checkpoints.*

**Table 8. CEM planning.** Contextualized macro AUROC [95% CI]. Flips are per pair (ground truth on test: 0.67
onsets, 0.60 resolutions). E ratios are means over test pairs. "Exact" is the share of fold-17 pairs with a change
where a* = a_gt.

| Checkpoint | Test soft | Test hard | Test onsets / resolutions | E(a*)/E(0) | E(a_gt)/E(0) | Fold 17 soft / hard | Fold 17 onsets | Fold 17 exact |
|---|---|---|---|---|---|---|---|---|
| **CNN** (matched) | 0.470 [0.445–0.480] | 0.574 [0.557–0.579] | 9.2 / 0.98 | 0.60 | 1.02 | 0.558 / 0.637 | 6.3 | 0.000 |
| **ViT** (matched) | 0.519 [0.501–0.540] | 0.612 [0.599–0.623] | 6.7 / 0.79 | 0.45 | 1.07 | 0.538 / 0.628 | 6.4 | 0.001 |
| CNN, epoch 5 | 0.563 [0.542–0.575] | 0.624 [0.608–0.634] | 11.8 / 0.82 | 0.66 | 1.02 | 0.562 / 0.623 | 11.5 | 0.000 |
| CNN, λ 0.005 / wd 0.01 | 0.508 [0.492–0.529] | 0.590 [0.579–0.603] | 9.6 / 0.87 | 0.32 | 1.01 | 0.547 / 0.635 | 6.2 | 0.000 |
| CNN, k=1 | 0.435 [0.421–0.451] | 0.567 [0.558–0.576] | 7.5 / 0.96 | 0.27 | 1.06 | 0.669 / 0.714 | 2.9 | 0.031 |
| ViT, k=1, 405 epochs | 0.542 [0.527–0.566] | 0.633 [0.620–0.647] | 4.2 / 0.71 | 0.50 | 1.18 | 0.605 / 0.665 | 4.0 | 0.022 |
| ViT, k=1, predictor 128 | 0.513 [0.487–0.529] | 0.618 [0.603–0.629] | 4.1 / 0.78 | 0.57 | 1.14 | 0.539 / 0.635 | 4.1 | 0.011 |
| Paper Dynamics (CNN) | 0.517 [0.497–0.534] | 0.605 [0.590–0.617] | 8.3 / 0.96 | 0.44 | 1.04 | 0.542 / 0.622 | 7.7 | 0.000 |
| *Carry-forward* | 0.750 [0.735–0.763] | | | | | 0.743 | | |

**Table 9. Per-label readout.** Contextualized macro AUROC [95% CI]. Evidence is real minus nothing changed, with a
paired interval. Onset and resolution AUROCs use δ alone over each head's live entries, with the nothing-changed score
in parentheses. τ is given as a multiple of the val δ spread.

| Checkpoint | τ | Test | Test, nothing changed | Test evidence | Onset AUROC | Resolution AUROC | Fold 17 / nothing changed | Fold 17 evidence |
|---|---|---|---|---|---|---|---|---|
| **CNN** (matched) | 10⁴ | 0.772 [0.753–0.787] | 0.751 [0.732–0.769] | +0.020 [−0.002 to +0.042] | 0.540 (0.502) | 0.548 (0.465) | 0.793 / 0.763 | +0.030 [+0.013 to +0.055] |
| **ViT** (matched) | 3·10³ | 0.781 [0.762–0.798] | 0.761 [0.745–0.782] | +0.020 [+0.005 to +0.031] | 0.567 (0.531) | 0.521 (0.525) | 0.809 / 0.767 | +0.042 [+0.030 to +0.060] |
| CNN, epoch 5 | 10⁴ | **0.800** [0.786–0.813] | 0.779 [0.760–0.798] | +0.021 [+0.005 to +0.039] | 0.607 (0.563) | 0.552 (0.483) | 0.806 / 0.758 | +0.048 [+0.020 to +0.058] |
| CNN, λ 0.005 / wd 0.01 | 10⁴ | 0.755 [0.737–0.777] | 0.769 [0.749–0.784] | −0.014 [−0.027 to +0.008] | 0.518 (0.539) | 0.495 (0.520) | 0.813 / 0.748 | +0.065 [+0.043 to +0.085] |
| CNN, k=1 | 10⁴ | 0.759 [0.740–0.780] | 0.740 [0.721–0.761] | +0.019 [+0.003 to +0.035] | 0.519 (0.479) | 0.533 (0.486) | 0.829 / 0.743 | +0.086 [+0.066 to +0.107] |
| ViT, k=1, 405 epochs | 3·10³ | 0.769 [0.753–0.787] | 0.761 [0.741–0.779] | +0.009 [−0.008 to +0.030] | 0.539 (0.524) | 0.555 (0.481) | 0.781 / 0.742 | +0.040 [+0.026 to +0.061] |
| ViT, k=1, predictor 128 | 10³ | 0.773 [0.756–0.793] | 0.754 [0.729–0.771] | +0.018 [+0.003 to +0.045] | 0.545 (0.506) | 0.587 (0.476) | 0.781 / 0.759 | +0.022 [+0.006 to +0.042] |
| Paper Dynamics (CNN) | 10³ | 0.790 [0.774–0.806] | 0.782 [0.761–0.796] | +0.008 [−0.006 to +0.032] | 0.579 (0.575) | 0.555 (0.492) | 0.813 / 0.760 | +0.052 [+0.024 to +0.067] |
| *Carry-forward* | ∞ | 0.750 | | | | | 0.743 | |
| *Paper Inverse* | | 0.827 [0.811–0.843] | | | | | 0.902 | |

**Table 10. Truth vs decoys under E, without search.** Within-pair AUROC on pairs with a change; chance is 0.5.
Same-count decoys are shown as real / nothing changed, then the evidence. The last column is the forward loss relative
to doing nothing, E(a_gt) / ‖h_t+1 − h_t‖², on test / fold 17.

| Checkpoint | Same-count, test | Test evidence | Same-count, fold 17 | Fold 17 evidence | Zero, test | Zero, fold 17 | E(a_gt) / do-nothing |
|---|---|---|---|---|---|---|---|
| **CNN** (matched) | 0.524 / 0.484 | +0.040 [+0.027 to +0.055] | 0.722 / 0.561 | +0.161 [+0.145 to +0.178] | 0.516 [0.501–0.531] | 0.583 [0.564–0.602] | 0.973 / 0.716 |
| **ViT** (matched) | 0.563 / 0.563 | +0.001 [−0.011 to +0.013] | 0.723 / 0.591 | +0.132 [+0.119 to +0.145] | **0.384** [0.367–0.399] | 0.578 [0.563–0.594] | 0.965 / 0.835 |
| CNN, epoch 5 | 0.544 / 0.512 | +0.032 [+0.016 to +0.046] | 0.592 / 0.522 | +0.070 [+0.055 to +0.085] | 0.442 [0.425–0.460] | 0.457 [0.440–0.473] | 0.952 / 0.944 |
| CNN, λ 0.005 / wd 0.01 | 0.500 / 0.525 | −0.025 [−0.039 to −0.011] | 0.671 / 0.565 | +0.106 [+0.092 to +0.121] | 0.469 [0.454–0.483] | 0.552 [0.532–0.570] | 0.995 / 0.806 |
| CNN, k=1 | 0.538 / 0.570 | −0.033 [−0.048 to −0.018] | 0.890 / 0.577 | +0.313 [+0.297 to +0.330] | 0.543 [0.527–0.558] | 0.816 [0.801–0.830] | 1.007 / 0.378 |
| ViT, k=1, 405 epochs | 0.561 / 0.567 | −0.006 [−0.018 to +0.006] | 0.849 / 0.579 | +0.270 [+0.255 to +0.285] | **0.346** [0.330–0.361] | 0.763 [0.748–0.778] | 1.000 / 0.721 |
| ViT, k=1, predictor 128 | 0.586 / 0.576 | +0.010 [−0.003 to +0.023] | 0.790 / 0.583 | +0.207 [+0.192 to +0.222] | **0.397** [0.381–0.413] | 0.683 [0.664–0.702] | 0.956 / 0.766 |
| Paper Dynamics (CNN) | 0.550 / 0.516 | +0.034 [+0.021 to +0.048] | 0.672 / 0.510 | +0.162 [+0.147 to +0.178] | 0.491 [0.476–0.505] | 0.557 [0.540–0.574] | 0.964 / 0.900 |

**Table 11. Matched CNN − ViT, paired** (same pairs, same patient-level resamples).

| Metric | Test | Fold 17 |
|---|---|---|
| CEM, soft | −0.048 [−0.086 to −0.032] | +0.019 [−0.010 to +0.038] |
| CEM, hard | −0.039 [−0.060 to −0.029] | +0.010 [−0.007 to +0.021] |
| Per-label readout | −0.009 [−0.030 to +0.010] | −0.015 [−0.032 to +0.008] |
| Per-label readout, evidence | +0.000 [−0.023 to +0.026] | −0.011 [−0.036 to +0.013] |
| Onset AUROC, evidence | +0.002 [−0.032 to +0.047] | −0.009 [−0.042 to +0.029] |
| Same-count decoys | −0.039 [−0.053 to −0.026] | −0.001 [−0.014 to +0.012] |
| Same-count decoys, nothing changed | −0.079 [−0.092 to −0.067] | −0.030 [−0.047 to −0.014] |
| Same-count decoys, evidence | +0.040 [+0.023 to +0.056] | +0.029 [+0.011 to +0.049] |
| Zero action | +0.132 [+0.112 to +0.152] | +0.005 [−0.019 to +0.030] |

### Interpretation

1. **CEM planning fails with either encoder.**
   - On test, all eight checkpoints are far below carry-forward (0.750): soft AUROC 0.435–0.563, hard 0.567–0.633.
   - Every search is exploited. It returns 4–12 onsets per pair against 0.67 in the ground truth, at 27–66% of
     E(0), while the true action sits 1–18% *above* E(0).
   - The matched ViT is less bad than the CNN (soft +0.048, hard +0.039, paired) and its search flips fewer labels
     (6.7 onsets against 9.2). It is still 0.23 below carry-forward.
   - CEM stays below carry-forward even on the pairs the checkpoints trained on.
     - The closest is the most memorized forward model, the k=1 CNN, whose forward loss on fold 17 is 0.378 of doing
       nothing. It reaches 0.669 / 0.714 against 0.743, with 2.9 onsets per pair.
     - It returns the exact transition in only 3.1% of changed pairs.
2. **On held-out pairs, the energy carries almost no evidence about the transition, with either encoder.**
   - **Same-count decoys:** the evidence is +0.040 for the CNN and +0.001 for the ViT, and ranges from −0.033 to
     +0.040 across all checkpoints. The ViT's higher raw score (0.563 against 0.524) is all in its nothing-changed
     control (0.563): it comes from h_t, not from the outcome.
   - **Doing nothing:**
     - The CNNs score 0.44–0.54, near chance.
     - The ViTs score 0.346–0.397, below chance: their true action makes held-out predictions *worse* than doing
       nothing (E(a_gt)/E(0) 1.07–1.18).
     - The ViT's action effects move held-out predictions further, and in the wrong direction. This also gives CEM
       more room (E(a*)/E(0) 0.45 against 0.60).
3. **On training pairs, the energy carries strong evidence with either encoder: Dyn memorizes the training
   transitions.**
   - **Same-count decoys:** 0.722 for the CNN and 0.723 for the ViT (paired difference −0.001), with evidence +0.161
     and +0.132. After 100–405 epochs of k=1 repetition the scores reach 0.890 (CNN) and 0.849 (ViT).
   - **Doing nothing:** 0.58 for both matched models; 0.816 and 0.763 at k=1.
   - **Before memorization:** only the CNN at epoch 5 has a small gap between members and test (0.592 against
     0.544). §1 showed that it had not yet memorized.
   - **Consistent with §1:** this is the action test seen from the planner's side. The ViT's encoder does not
     memorize, but its forward model memorizes the training transitions as much as the CNN's does.
4. **The per-label readout beats carry-forward, but mostly through h_t.**
   - On test, every checkpoint scores 0.755–0.800, above carry-forward (0.750).
   - The nothing-changed control alone already gives 0.740–0.782.
   - The evidence from the outcome is at most +0.021: CNN +0.020 [−0.002 to +0.042], ViT +0.020 [+0.005 to +0.031],
     paired difference +0.000.
   - **The temperature confirms the evidence is weak.**
     - Val AUROC rises up to about 3× the δ spread, then stays flat within 0.001 out to 10⁴×.
     - In that limit E only breaks ties within each label's present and absent groups.
     - Letting E override the carry-forward ordering lowers val AUROC, to 0.601 at 0.03× for the CNN.
   - **What the nothing-changed gain is.** Its gain over carry-forward (up to +0.03) is a patient-conditional
     transition prior learned by Dyn: how far flipping label i moves the prediction away from the patient's own h_t.
   - **The least memorized forward models read best.** CNN epoch 5 reaches 0.800 [0.786–0.813] and the paper's
     Dynamics 0.790. Both are still below the paper's Inverse (0.827).
5. **CNN vs ViT: no difference where it matters.**
   - The evidence on held-out pairs is the same: per-label +0.000, onset +0.002, both paired.
   - The memorization of training transitions in Dyn is the same: same-count decoys 0.722 against 0.723.
   - The differences are all about how the ViT's action effects fail to transfer: a less exploited CEM (+0.048), and
     a true action worse than doing nothing on test (zero 0.384 against 0.516).
6. **What this means for the dynamics objective.**
   - Swapping the encoder removes §1's encoder-level memorization but leaves planning where it was.
   - With either encoder, Dyn learns the action effects of its training transitions, and they do not transfer. On
     held-out pairs, the planner has only the transition prior that h_t carries.
   - The planning failure lies in how the forward model generalizes, not in the encoder's collapse.

### Caveats

- **Capacity confound** (as in §1 and §2). The ViT has 4.6× fewer encoder parameters.
- **Plain-E CEM only.** The prior-regularized objective (paper appendix, `tab:cem-tau`) was not rerun. The per-label
  readout stands in for a posterior readout.
- **Bootstrap shift for the CEM readouts.** Their bootstrap replicates sit up to 0.007 below the point estimate
  (0.8 SD for the matched CNN's soft readout), probably because rare classes change composition across resamples.
  The intervals are percentile intervals, so they are asymmetric around the point. The other metrics are centred.
- **Member exposure differs.** k=1 checkpoints see each fold-17 pair once per epoch. Multistep checkpoints see the
  k=1 hop plus the longer hops from the same ECGs, so one cannot compare member scores across pair sets one to one.
- **Only the k=1 hop is tested,** as in §1.
- **One seed per configuration.** CEM uses the same generator seed for every checkpoint.
- **Reference values differ slightly from the paper's.** Carry-forward and Inverse here are point estimates (0.750 /
  0.827), whereas the paper reports bootstrap means under a pair-level resampling (0.7515 / 0.8286).

### Reproduce

```bash
python anatomie/cem_test.py \
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \
    --ckpt cnn_ep5=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/pretrain_multistep_.ckpt \
    --ckpt cnn_lowreg=runs/runs/20260930/062649/71f4c433519a/checkpoints/last.ckpt \
    --ckpt cnn_k1=runs/runs/20260909/123412/ac1e0c66cc08/checkpoints/last.ckpt \
    --ckpt vit_k1=runs/runs/20261006/134920/e98705006684/checkpoints/last.ckpt \
    --ckpt vit_k1_p128=runs/runs/20261006/142437/c68ff538b143/checkpoints/last.ckpt \
    --ckpt paper_dyn=archive/pretrain_val.ckpt \
    --cache /scratch/josefernandes/cache/cem_test_h.pt \
    --out outputs/cem_test/results.json --fig outputs/cem_test/cem_test.png
```

Embedding the 32,805 ECGs with the nine encoders (eight forward checkpoints plus `--inverse`, default
`archive/inverse.ckpt`) takes about 5 minutes on one RTX 3090 Ti with the data on local `/scratch`. The h cache is
0.3 GB and node-local. Each checkpoint then takes about 35 s: CEM, decoys and the per-label readout on both
populations, plus the bootstraps.
- **Resuming:** the script saves after each checkpoint (`results.json`, and the bootstrap replicates in
  `results_reps.pt`) and skips finished checkpoints when rerun.
- **`--plot-only`** redraws the figure.
