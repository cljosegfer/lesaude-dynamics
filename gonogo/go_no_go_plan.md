# Go/No-Go Plan: World Models with Weak, Confounded, Partially Grounded Action Labels

*Revision 13 (11 Nov 2026). User decisions, recorded before running anything (Section 3, "Revision 13"):*
- *An input-variant decomposition (naive, z − s, z − s + A, oracle) on the γ_mix = 1 and 0.5 sweep draws.*
- *A semi-synthetic test on h (I48 vector, noise measured on h, Γ̂² from a new rotation on h), with rule (c), the sweep and the decomposition. C2 is still read on b.*
- *No stronger driver is added.*
- *The tex now names the CRT.*

*Revision 12 (11 Nov 2026). User decisions, recorded before any UC/GC data (Section 3, "Revision 12"):*
- *A CRT replaces the stratified permutation test (validated on 96 U codes per estimator).*
- *Realistic noise on injected event pairs: G codes take their source control's residual-variance ratio, and U, UC and GC codes the median C1 ratio, 1.11.*
- *γ_mix = 0 is rerun and rule (c) reapplied; then the UC/GC sweep, Part 1 estimator and oracle only.*

*Revision 11 (11 Nov 2026). User decisions, recorded before any semi-synthetic data (Section 3, "Revision 11"):*
- *Part 2's ψ̂² inference: the permutation null for tests against 0, and the bias-corrected plug-in variance for intervals (the percentile bootstrap is secondary). It is validated for coverage first.*
- *Rule (c) is applied in the semi-synthetic test at γ_mix = 0; the latent 0.5× is preliminary.*
- *The injected vectors are shrunk to their debiased norms.*
- *The p-driven risk is kept as a report-only stress scenario.*
- *Failure-table wording: the anchors did not remove K21; only the margin did.*

*Revision 10 (10 Nov 2026). Part 2 is pre-declared after Phase 0.5 and before any Part 2 data (Section 3, "Revision 10"):*
- *b is primary and h secondary.*
- *The injected vectors are I47, I48 and I49 on b, plus the one-feature, two-feature and diffuse scenarios.*
- *4× is added to the scale sweep, and C2 is read at the oracle-chosen scale.*
- *The latent environment is calibrated to b.*
- *Weak labels use κ < 1, lower for ED-only next stays.*
- *Scoring is asymmetric between baselines and method.*
- *The execution order is fixed and the timeline updated.*
- *Two real-data analyses on b are reported only: margin sensitivity and the failure table. The decision rules of 2.8, 3.5 and Phase 0.5 are unchanged.*

*Revision 9 (9 Nov 2026, late). The sanity rule for candidate encoders (0.5a) was declared before any new number on g was looked at. The trained g reached a probe AUROC below 0.55, and a sequential diagnosis of g follows (outcome in "Phase 0.5 results"). No ψ̂² on g, and no 0.5a, 3d or 3e, until the sanity outcome is settled. The 0.5b rule is unchanged. `wandb_resume.json` was already absent from the repository root on 9 Nov 21:58; nothing was deleted.*

*Revision 8 (9 Nov 2026). Before any result on g (the g run is neither read, exported nor evaluated in this revision); the 0.5b rule is unchanged. The 0.5a and 0.5c rules change only in their metric and states.*
- *Metric (0.5a statistic; 0.5c).*
  - *Primary: Ledoit–Wolf shrinkage of Σ (analytical intensity, toward a scaled identity in standardized coordinates), estimated per representation on the stable pairs of the folds the statistic uses. It has no tuning parameter, and the intensity is recorded.*
  - *Sensitivity, always reported: revision 7's 99.9% principal-subspace pseudo-inverse and the full inverse.*
  - *Decisions use the primary only. A disagreement with the pseudo-inverse on qualification is reported. The same metric is used for the positive controls, the withheld codes, Γ̂ and the tested codes.*
  - *Why: revision 7's pseudo-inverse was chosen after the folds-0–9 references were seen, and it changed whether b qualified (2 → 3 of 5).*
- *Concatenation. In population ψ²([g; b]) ≥ ψ²(g), ψ²(b), so the choosing rule weighs added information against added estimation noise (0.5a).*
- *States. [b, g, m] for g and [g; b]; [b, m] for b, without h; [b, h, m] for the references. `gain.py` accepts `--state bm`.*
- *References recomputed under all three metrics (Phase 0.5 results), with the per-feature Δb readout. `method_section.tex` §wm-test gets the Ledoit–Wolf paragraph (§7 item 13).*

*Revision 7 (3 Nov 2026, evening). After the first Phase 0.5 result (0.5b statistic 1, label-only), so the 0.5b decision rule is unchanged. The 0.5a and 0.5c rules change; no result on g exists yet.*
- *0.5a, candidates. Eligible: the single-recording encoder g, [g; b] and b, with ties broken in the order [g; b], then g, then b. (Corrected on 9 Nov, before any intra result: this line had listed "g, [g; b] and b … in that order", which contradicted the decision rule.) The current h and [h; b] are references only: they are the empirical check of the method's exclusion of same-patient objectives. If h beats g on folds 0–9, that is reported as a finding against the encoder requirement, and h is not chosen.*
- *0.5a, views of g. One global amplitude factor within ±5% (3 Nov runs with ±20% are invalid), a time shift of up to ±0.5 s, noise at 5% of each lead's SD, one 0.5 s mask, and no stretch or resampling. Reason: SIGReg learns invariance to what the views vary, and amplitude and rate are part of the signal.*
- *(Superseded by revision 8: the primary metric is Ledoit–Wolf shrinkage, and this pseudo-inverse is a sensitivity analysis.) 0.5a and 0.5c, statistic. Σ⁻¹ in ψ̂² is the pseudo-inverse on the principal subspace of Σ that holds 99.9% of the stable-pair variance. The full inverse is ill-conditioned: for h, 42 of 256 directions hold 99.9% of the variance and the condition number is about 6×10⁷, so ψ̂² is dominated by noise. Adopted after the folds-0–9 references of h, [h; b] and b were computed and before any result on g. The full-inverse values are reported alongside. To be confirmed by the user.*
- *0.5c. Positive controls are compared on folds 10–19 with quarter-fold estimates. The adjusted subset is reported alongside as a power check, and decisions stay on the clean subset.*
- *Terminology. "Pre-specified" replaces "confirmatory": the Phase 0.5 reruns use the same patients as Phase 0, and genuine confirmation needs new data, for example MIMIC-CXR.*
- *Collider check (`method_section.tex`, §7 item 12; note in 2.1). Each decision is repeated with and without the other codes' transitions in the conditioning set, keeping z and the anchors. The single-change estimate is only a descriptive sensitivity analysis.*
- *Information only.*
  - *0.5b power pre-check: "beyond provenance" for 3 of 5 codes has a 0.77 chance if A's reduction persists in full within constant setting, and 0.07 if half of it persists.*
  - *0.5a references on folds 0–9: Phase 0.5 results below.*

*Revision 6 (3 Nov 2026). Phase 0.5 rules revised before any Phase 0.5 result exists; no new experiment.*
- *Status of evidence. Phase 0 results that use set A are exploratory: A's definition and the analyses were chosen after earlier results were seen. The Phase 0.5 reruns, run under the frozen anchor rule (0.5d) and the rules below, are pre-specified (revision 7 wording; revision 6 said "confirmatory").*
- *0.5a: the encoder is chosen on folds 0–9, and the positive controls are reported on folds 10–19. Results on the selection folds are checks, not evidence. The tie-break order is now `[h; b]`, then a learned encoder, then `b`, and `b` stays the validation readout. For Task 2 reruns, the setting-only withdrawal is pre-registered as primary.*
- *0.5b: the recovery ratio within constant-setting pairs had a zero denominator, because withholding a constant setting removes nothing. It is replaced by the absolute gain reduction by A, E[Δt] − E[Δt + A], within constant-setting pairs against all pairs, and the rule is restated on that statistic.*
- *0.5c: Γ̂² is no longer the 95th percentile of the withheld codes' ψ̂². It is a random-effects (DerSimonian–Laird) estimate, μ̂ + 1.645 τ̂, or the upper 95% bound of μ̂ when τ̂ = 0, with secondary margins per recording profile. Positive controls are compared on folds 10–19.*
- *Row (a) of the Phase 0 decision labels is reworded: A "recovers a substantial share" of the withheld setting, read on the setting-only withdrawal, instead of "recovers most".*
- *The anchor rule is frozen (new 0.5d): one frequency criterion, the separation screen on every candidate, and `label_sets.json` with its date and SHA-256. The 25 Oct approval of the 11 anchors and 11 held-out anchors (2.2) is superseded and kept for the record. The C1 controls remain the fixed validation subset; they have already been examined, so validation is not blind.*
- *The decision criteria of 2.8 and 3.5 are unchanged.*

*Revision 5 (27 Oct 2026). Phase 0 consolidated, with no new experiment.*
- *A section "Phase 0 results (provisional, encoder h)" follows 2.8. It has one table per check, the C1 outcome, the corrected decision labels of the Phase 0 task, and the findings that change the method and Part 2.*
- *Every Phase 0 result is provisional: it was computed on the current frozen encoder `h`, which Phase 0.5 may replace. Label-only results (linkage, recording route) are the exception.*
- *A new step, Phase 0.5, precedes Part 2: an encoder check, a provenance check, and Γ̂ by rotation, each with a decision rule.*
- *Part 2 changes four things: its driver is the setting at t+1; every method receives the real codes, including the no-route set A; it injects I48's effect vector; and it uses 500 permutations. Sections 4–8 are updated to match.*
- *The decision criteria (2.7, 2.8, 3.5) are unchanged. Where Phase 0 showed one to be ill-posed, a note says so. Earlier content is kept, and superseded passages are marked.*

*Revision 4 (25 Oct 2026). Brought in line with the downloaded data (week 0 done; code in `gonogo/part1/`). The linkage is a join on Ext-ICD's per-ECG encounter identifiers (2.1). Same-episode pairs, an ED visit and the admission it led to, are stable by construction and leave the analysis, which moves the stable fraction from 48.3% to 29.3%. The event threshold counts clean onsets. The coding system (ICD-9/ICD-10) joins `z`. The primary statistic becomes a cross-fitted R-learner gain with a conditional randomization test (2.3, 2.5). The negative-control pool and its three roles are fixed (2.2, approved 25 Oct). The recording route (2.4) and sanity check 1 (2.6) are measured, and new risks and method changes follow (Sections 5 and 7).*

*Revision 3 (22 Oct 2026). Data definitions corrected after checks against the pair index: cross-stay pairs only, labels of ECGs outside a stay treated as missing, cross-fitting instead of a single test fold. Adds the evidence already measured (1.1), two tests of the method's premises to Part 1 (context-dependent coding in 2.4, anchor relevance in 2.3), a power check to Part 2 (3.6), and the changes `method_section.tex` needs (Section 7). Revision 2 replaced the ECGSYN signal generator with a latent synthetic environment and a semi-synthetic test on MIMIC-IV-ECG.*

## 1. Context

The workshop model is an action-conditioned LeJEPA on MIMIC-IV-ECG. Its action is `a_t = y_{t+1} − y_t` over the 76 three-character ICD-10 codes I00–I99, taken from the stay-level discharge codes of MIMIC-IV-ECG-Ext-ICD. It showed three problems:

- On held-out pairs, the dynamics predictor does no better than shrinking `h_t` toward the population mean. It is not a copy of `h_t`: its predictions move away from `h_t` by 28% of the real change in squared distance. But the action effects it learns do not transfer. They explain +30% of the residual on training pairs and −18% on test pairs (ViT, k=1).
- In signal space, the action explains almost none of the change between consecutive ECGs. It accounts for 0.2% of the median beat's unexplained change and at most 3.1% of any clinical feature (heart rate), with nonlinear interactions included.
- Zeroing the action for every pair performs the same as the action-conditioned model (finetune 0.7391 vs 0.7409).

Many ICD codes have no ECG signature. The encoder instead learns patient features and "guesses" diseases typical of that patient profile.

### 1.1 Evidence already measured

All on k=1 cross-stay pairs of MIMIC-IV-ECG.

| Axis | Finding | Source |
|---|---|---|
| Weak | Chronic codes disappear at the next stay: I10 24%, I25 16%, I50 16%, I48 13% of the time (all cross-stay pairs; between episodes 33%, 23%, 24%, 19%). A code correlates 0.54 with itself across stays, while the ECG's evidence for it (supervised logit) correlates 0.75. | stay labels; `supervised_triage` logits, fold 18 |
| Weak, context-dependent (recording route, measured) | Most disappearances are the label source. An ED-only stay is labelled with its ED discharge codes, which omit chronic conditions: when the next stay is ED-only, I10, I25, I50 and I48 disappear 56%, 90%, 95% and 73% of the time, against 28%, 16%, 18% and 13% when it is an admission (adjusted OR 3.2, 44, 95, 22). Non-cardiac chronic codes are never recorded at ED-only stays. Crossing the ICD-9 to ICD-10 switch raises I48 and I50 onsets (OR 1.43 and 1.28). | `gonogo/part1/recording_route.py` (2.4) |
| Weak | 55% of ECGs from I48-coded stays have an irregular rhythm, against 15% otherwise. Of the I48 onsets, 33% were already irregular before the code appeared and 48% are irregular after. | median-beat features |
| Partially grounded | Within a patient, the change in a supervised encoder's logits detects new codes at AUROC 0.549 on average (43 codes). Codes with an ECG signature reach 0.61–0.69 (I21, I44, I47, I48); I10, I25 and I70 sit at about 0.50. Across patients the same logits reach 0.735. | `supervised_triage` logits, fold 18 |
| Partially grounded | Label effects on the change, in SDs of the stable pairs' change: I48 onset raises HR by +0.61 and RR irregularity by +0.31, I44 onset QRS width by +0.20, and I21 resolution lateral T waves by +0.25. | `anatomie/beat_action_ceiling.py` |
| Partially grounded (Part 1, cross-fitted, conditioned on `z`; provisional, encoder `h`) | On clean pairs between episodes, in SDs of the change: I44 and I45 onsets widen the machine QRS by +0.95, I48 removes the P wave (+1.39) and raises HR (+0.59), I47 raises HR (+0.57), I49 raises RR irregularity (+0.88, visible in `b` only). Every positive control passes sanity check 1, and conditioning on `z` changes their gains by about 10% or less. | `gonogo/part1/gain.py` (2.6) |
| Confounded (indirect only) | Most onsets (I21, I47, I49, I50, I25, I10, I48) share one signature, faster HR and smaller QRS amplitude, consistent with an acute context. Not yet tested against negative controls. | same |
| Noise floor | Two ECGs of the same stay, whose codes are identical by construction, change 80–84% as much as ECGs of consecutive stays (whole beat, gaps under 24 h). Same-episode pairs (2.1) are a second noise floor. | same |

Weak and partially grounded labels are established, and the recording route is measured. Confounding of the transition is still unverified; C1 tests it. *(Revision 5: C1 has been measured; see "Phase 0 results" after 2.8.)*

### Working hypothesis (H)

ICD transition labels are not actions. They are **instance-dependent noisy annotations of the transition**: `a = g(true latent transition, patient state, context, noise)`, and some coordinates have no effect on the observed signal (ungrounded).

The context can confound the transition by two routes:

- **Event route.** The context changes which conditions occur and also changes the ECG; an acute illness can trigger both.
- **Recording route.** The context changes which conditions get coded. An inpatient admission records more comorbidities than an ED visit, so the label noise depends on the context. The flicker of chronic codes (1.1) suggests this route is strong in MIMIC.

Either way, when code assignment correlates with an unobserved context that also changes the ECG (admission, acute illness), **ungrounded codes look grounded**. Existing methods are not designed for this regime:

- sparsity-based world models (SPARTAN, mechanism sparsity)
- action-mean centering (AD-JEPA)
- weakly supervised latent actions (LAOM)

### GO requires both conditions

- **C1.** Confounding of the transition is measurably present in MIMIC-IV-ECG (Part 1).
- **C2.** Existing baselines fail specifically on ungrounded-but-confounded codes when the confounder is a real one from MIMIC-IV-ECG, and the failure grows with confounding strength in a controlled simulation (Part 2).

If either fails, the "confounded" axis is dropped and the project is re-scoped (see Section 6).

---

## 2. Part 1: Is the transition confounded in MIMIC-IV-ECG?

**Question.** For each code `c`, does the onset `a_c` carry information about the transition beyond the current state `h_t`? And is that information in the ECG-measurable part of the signal or elsewhere?

### 2.1 Data

**Linkage.** MIMIC-IV-ECG-Ext-ICD (`records_w_diag_icd10.csv`) carries, per ECG, the ED stay, the admission that ED stay led to, the admission the ECG was taken in, age, sex, fold and the stay numbering. Joined on `study_id`, it reproduces the Lance table exactly (ECGs, subjects, folds, stay numbering, times to the second), and `all_diag_all` at three characters reproduces the 76 labels on every ECG, so it is their full-chapter version (`gonogo/part1/linkage.py`). A stay is the admission when the ECG was taken in hospital, otherwise the ED stay. An ED-only stay is labelled with its ED discharge codes and an admission with its hospital discharge codes, so the care setting decides which coding process produced a label (2.4).

**Pairs.** Use the cross-stay pairs of the pair index (`pairs.lance`, `pair_type = cross_stay`): 166,229 pairs of consecutive stays of the same patient, with both ECGs inside a stay. Keep only the 121,545 pairs **between episodes**; 29.3% of them are stable, and 85,906 change at least one code, 30,863 of them exactly one (35.9%). With usable `b` and `m` at both ECGs, 118,670 remain.

**Same-episode pairs are excluded.** An ED visit and the admission it led to are two stays of one episode, and both ECGs carry that admission's discharge codes. These 44,684 pairs (26.9%) are therefore stable by construction (100.00%). They are 99% of the pairs under 12 h apart and 65% of those at 1–7 days. They violate overlap like within-stay pairs and are used only as a noise floor. Revision 3's 48.3% stable fraction included them.

Do not use all consecutive ECGs per patient (638,683 pairs, revision 2's "about 639k"):

- **Label artefacts.** 52.6% of those pairs include an ECG recorded outside any stay. Such ECGs have no diagnosis codes (missing, stored as zeros), which creates 119,220 artefact transitions. Whether an ECG lies outside a stay is the care setting, a variable of `z`, so these artefacts would produce exactly the "gain that shrinks when `z` is controlled" that C1 looks for.
- **No transitions within a stay.** Codes are assigned per stay, so pairs within a stay never change codes (0 of 177,677). They carry no information about transitions and violate overlap. Use them only as a noise floor (2.9).

**Folds.** Cross-fit over the 20 patient-stratified folds: train on 19, predict the held-out fold, so that every pair gets an out-of-fold prediction. A single test fold leaves too few onsets (2.2).

**Clean subset.** Keep only:
- stable pairs (no cardiac code change), and
- single-code transitions (`|a_t|_1 = 1`).

This prevents co-occurring code changes from confounding each other. For a non-cardiac code `c`, the clean analogue is the stable pairs at risk for `c`, so its onset never co-occurs with a cardiac change; other non-cardiac codes may change. The clean subset is the primary analysis.

*Revision 7.*
- Restricting to single-code transitions conditions on the other codes at the value zero. The clean subset is therefore not free of collider bias, and it is not a collider check.
- The collider check (`method_section.tex`, after ass:wm-sufficiency) compares each decision with and without the other codes' transitions in the conditioning set, with `z` and the anchors kept in both. In this plan, that means the adjusted subset with and without `a_{−c}` as covariates.
- A decision that holds only with `a_{−c}` is flagged as possibly induced by a collider.
- The clean subset stays the primary analysis set of Part 1 and of the Phase 0.5 decisions. The adjusted subset (all at-risk pairs, with the other codes' transitions `a_{-c}` as covariates, as in the method's conditioning set) is a sensitivity analysis.

**Representations.** None may be trained with the action, and none may have seen labels from the folds it is scored on.

| Name | Source | Role |
|---|---|---|
| `m` | `machine_measurements.csv` (`gonogo/part1/measurements.py`, 12 dimensions) | ECG-observable, interpretable |
| `b` | median-beat features of `anatomie/beat_action_ceiling.py`: HR, RR irregularity, QRS width, per-lead QRS, ST and T | fixed by the signal |
| `h` | encoder trained without labels or actions, frozen: the no-action ViT (run `20261019/182223`) | captures what `m` and `b` miss |

A supervised encoder may be added only when scored on folds it was not trained on, because it memorizes its training labels (+0.021 probe gap between seen and unseen patients). The forward ViT encoders are the most visit-invariant representations measured so far: the action-conditioned one explains 62% of the next embedding from the current one. `h` may therefore have discarded within-patient changes, which `b` and `m` guard against (matrix row 4).

**Derived measurements**, all in ms, from `machine_measurements`:

| Measurement | Formula |
|---|---|
| HR | `60000 / rr_interval` |
| PR | `qrs_onset − p_onset` |
| QRS duration | `qrs_end − qrs_onset` |
| QT | `t_end − qrs_onset` |
| QTc | Bazett |
| Axes | `p_axis`, `qrs_axis`, `t_axis` |

Missing P-wave fields are kept as an indicator, since P absence is itself informative for I48. Drop physiologically impossible values. As implemented, missing values are sentinels (29999 for unmeasured fields, mostly P waves in 15% of ECGs, plus a few 32767 and 65535). The ranges are HR 24–250, PR 60–400, QRS 40–250, QT 200–700 and QTc 250–700. Axes enter as (cos, sin), so that their differences do not wrap. PR and the P axis are imputed where P is missing, and the indicator carries that information.

The file's `ecg_time` is wrong for 17,850 ECGs (2.2%), by whole days up to decades and 44% of them after the patient's death, so `m` is joined on `study_id` only. Its values belong to the right waveforms: machine HR against median-beat HR has r = 0.98 on all ECGs and 0.99 on those rows.

**Targets.** `Δm = m_{t+1} − m_t` and `Δb = b_{t+1} − b_t` (each dimension z-scored), and `Δh = h_{t+1} − h_t`. The state `s_t = [b_t, h_t, m_t]` holds every target's current value.

**Observed controls `z`.**

| Variable | Source |
|---|---|
| Δt between ECGs (log days) | `ecg_time`, stored as float64 seconds. `pd.to_datetime` reads it as nanoseconds, so convert explicitly. |
| Care setting at t and t+1 (ED only / ED then admitted / inpatient) | Ext-ICD encounter identifiers. Outpatient ECGs never occur in cross-stay pairs. |
| Coding system at t and t+1 (ICD-9 / ICD-10 / unknown) | `hosp/diagnoses_icd` per admission. ED-only stays are inferred from the patient's admissions on either side of the switch or from `anchor_year_group`; that rule agrees with `diagnoses_icd` on 99.3% of the admissions it decides. |
| Age, sex | Ext-ICD (`hosp/patients`) |
| HR at t | `m_t` and `b_t`, in the state |

Revision 2's "same admission" is constant on cross-stay pairs, and "same episode" turned out to determine the label (stable by construction), so same-episode pairs are excluded rather than controlled for. Ext-ICD converts ICD-9 codes to ICD-10, so a pair that crosses the switch (October 2015) changes coding process. 11,561 between-episode pairs (9.5%) do, and 7.2% involve a stay of unknown system. Candidate drivers for 2.9 are linked but kept out of `z`, because they can be consequences of the transition (an arrhythmia leading to admission or ICU): admission type, length of stay, ICU, ED disposition, triage acuity and temperature.

**External data, on disk since 25 Oct** (`/snfs2/josefernandes/datasets/physionet/<project>/<version>/`, fetched with `scripts/download_gonogo.sh`, checksums verified):
- `machine_measurements.csv` (MIMIC-IV-ECG 1.0; the local copy of that project's 176 MB manifest is truncated, so this file was checked by content as described above)
- `hosp/admissions`, `hosp/patients`, `hosp/transfers`, `hosp/diagnoses_icd`, `hosp/d_icd_diagnoses` (MIMIC-IV 2.2)
- `ed/edstays`, `ed/triage` (MIMIC-IV-ED 2.2)
- the full MIMIC-IV-ECG-Ext-ICD labels (1.0.1). They keep ICD-9 V-codes whose strings are also valid ICD-10 codes, for example V10.3 "history of breast cancer", which reads as a pedal-cycle accident in ICD-10. External-cause chapters V–Y are therefore excluded from the negative-control pool. Cardiac labels are unaffected: numeric ICD-9 codes were converted, and ICD-9-era admissions carry I-codes as often as the rest.

### 2.2 Code groups

| Group | Codes | Expected result |
|---|---|---|
| Positive controls (known ECG signature) | I44, I45, I47, I48, I49 (1,074–6,776 onsets over all cross-stay pairs; 129–777 clean onsets between episodes, so only I47 and I48 reach 250) | gain on `Δm`, `Δb` and `Δh` |
| Negative controls (no plausible ECG signature) | the 11 C1 controls of the approved split below; I82 and I87 as a secondary venous set | no causal gain |
| Ambiguous (the interesting ones) | the cardiac codes with ≥ 250 clean onsets between episodes: I10, I12, I25, I27, I50, I69, I73, I85, I95 | to be determined |
| Placebo | synthetic, one per real code (see 2.3) | null by construction |

**Venous codes are weak negative controls.** I80 and I82 (venous thrombosis) travel with I26 (pulmonary embolism), which has ECG signs, so they can carry a grounded event. I80 and I83 fall below the threshold; I82 and I87 pass.

**Disjoint roles.** Split the negative-control pool before looking at any result into three disjoint sets: controls for C1, anchors for the method, and held-out anchors for estimating residual bias (`method_section.tex`, identification guarantees). Using the same codes for all three makes the evidence circular.

**The pool and its roles (fixed 25 Oct, before any result on these codes).** None of revision 3's examples reaches the threshold on clean pairs between episodes: knee osteoarthritis has 167 onsets, femur fracture 185, other cataract 78 and age-related cataract fewer than 50. Like most chronic comorbidities, they are recorded almost only during admissions. The pool was built from label statistics alone (`gonogo/part1/controls.py`, `roles.py`; `gonogo/part1/control_roles.json`):
- **Candidates.** 151 non-cardiac codes have ≥ 250 clean onsets. 117 were excluded, each with a recorded reason: electrolytes, fluids or nutrition; drugs or chemotherapy with ECG effects; heart rate through fever, anemia, bleeding or hypoxia; endocrine or cardiovascular risk factors; pulmonary disease or sleep apnea; cardiac symptoms, devices or history; brain, stroke or seizure; liver disease; nonspecific or supplementary codes; external causes V–Y.
- **Separation.** One code failed the screen: DNR status (Z66). Its Mantel-Haenszel odds ratio of co-onset with grounded cardiac codes, stratified by setting, age and sex, is 1.60 against a threshold of 1.5; the median over the 151 codes is 1.16.
- **Categories, from the data.** "Both" codes have at least 25 onsets into ED-only stays, so they are recorded in either setting and overlap holds across settings. "Admission" codes are recorded essentially only during admissions, so under `z` they are informative only on pairs whose later stay is an admission; they are where the recording route acts.
- **Roles.** A seeded random split into thirds within each category, approved unchanged. *(Revision 6: the anchor and held-out-anchor roles of this approval are **superseded** by the frozen no-route set of 0.5d. That set is rotated as a whole, and the C1 controls are excluded from it. The record below is kept. The C1 controls keep their role as the fixed validation subset, but they were examined in Phase 0, so validation on them is no longer blind.)*

| Role | Codes |
|---|---|
| C1 controls | G43 migraine, J06 upper respiratory infection, K21 GERD, K22 oesophagus, K29 gastritis, K57 diverticular disease, K59 constipation, K64 haemorrhoids, K80 gallstones, N40 BPH, Z96 joint or lens implants |
| Anchors | B37 candidiasis, E55 vitamin D deficiency, M19 osteoarthritis, M25 joint pain, M81 osteoporosis, N39 urinary tract, R10 abdominal pain, R32 incontinence, R51 headache, Z23 immunization, Z90 absence of organs |
| Held-out anchors | G89 pain, K83 biliary, L03 cellulitis, L89 pressure ulcer, M47 spondylosis, M48 spinal stenosis, M54 back pain, M79 soft-tissue pain, R11 nausea, R33 urinary retention, Z80 family history of cancer |

**Event threshold.** Keep a code if it has at least 250 **clean** onsets between episodes over the cross-fitted folds. 13 of the 76 cardiac codes pass: I10, I12, I25, I27, I47, I48, I50, I69, I73, I82, I85, I87, I95. Counting all cross-stay onsets, as revision 3 did, passed 47 codes with a median of 1,306 onsets. Revision 2's threshold (200 onsets in train and 50 in one test fold) kept 29 codes and rejected I80 (19 test onsets) and I83 (5). Codes below threshold are reported as "insufficient data", not as ungrounded; a significant result below threshold is reported but kept out of the BH families.

### 2.3 Estimator

For each code `c` and each target `Δ ∈ {Δm, Δb, Δh}`, train two predictors with cross-fitting (2.1). Use a small MLP or gradient boosting, with the same capacity and tuning budget for both.

- **Base:** `Δ ~ f(h_t, Z)`
- **Full:** `Δ ~ f(h_t, Z, a_c)`

**Gain.** `G_c = L_base − L_full` on out-of-fold predictions. Use MSE normalized by the variance of each target dimension, averaged over dimensions. Also report per-dimension gains for `Δm` and `Δb`, e.g. QRS for I44 and RR irregularity / P absence for I48. `G_c` estimates the conditional mutual information `I(Δ; a_c | h_t, Z)`.

**Primary statistic (revision 4).** Two separately trained MLPs differ by about 1e-4 in out-of-fold loss from seed to seed on `Δb`, even with shared initialization and batches. That is above the gains of the small positive controls. The decision statistic is therefore a cross-fitted R-learner gain `G_R`: the out-of-fold gain of `base + (a_c − π̂) θ`, where `θ` is a constant effect per target dimension, fitted on the other folds from the base model's out-of-fold residuals, and `π̂` is the cross-fitted propensity. Like the method's code channels, it is silent where `a_c = 0`, up to `π̂`. The MLP gain above is kept as a secondary statistic that can capture state-dependent effects. The base model is a two-hidden-layer MLP, three seeds, with `s_t = [b_t, h_t, m_t]` (`gonogo/part1/gain.py`).

**Conditioning sets `Z`, for each target:**
1. none (`s_t` only)
2. Δt only (log gap)
3. `z`
4. `z` + anchor labels `a^{C0}_t`: the transitions of the 11 anchor codes (2.2), as onset and resolution indicators

Comparing (1) with (2) measures confounding by observed context. Comparing (2) with (3) measures the confounding that anchors proxy beyond `z`, which is the premise of the method. If (2) vs (3) is null, the method has nothing to work with even when C1 passes.

*Note (revision 5; the rule above is kept as written).*
- *In Phase 0, "(2) vs (3)" was null both with the 11 anchors and with the full no-route set A.*
- *Yet A recovers a substantial share of what the setting removes when it is withheld (Task 2, Phase 0 results; revision 6 wording).*
- *The comparison assumes `z` is observed. It therefore cannot measure the anchors' value as a substitute for missing encounter metadata, which is where the method could matter. In that sense the criterion is ill-posed for the question it was meant to answer.*
- *Phase 0.5b tests what that substitution consists of.*

**Placebo code.**
1. Fit a propensity model `P(a_c = 1 | h_t, z)` within each training fold.
2. Sample a fake onset `ã_c` from it, with the same prevalence.
3. Compute `G̃_c` identically.

`ã_c` shares the code's dependence on the current state but has no link to the transition. A real code only "carries information" if `G_c` clearly exceeds `G̃_c`.

**Secondary estimator (effect direction).** Propensity-match onset pairs to non-onset pairs on `(h_t, z)` and report the mean difference in `Δm`, `Δb` and `Δh` per dimension.
- The `Δm` and `Δb` differences give interpretable effects, e.g. "I44 onset → +X ms QRS".
- The `Δh` differences are effect vectors that Part 2 injects as realistic grounded effects.

### 2.4 Recording route: does the context change the coding?

For each chronic code (I10, I25, I50, I48, and chronic negative controls), model with logistic regression, adjusted for Δt, age and sex:
- the probability that a code present at stay t is absent at t+1 (today 13–24% overall), as a function of the care setting at t+1;
- the reverse, an onset, as a function of the care setting at t+1.

If chronic codes "resolve" much more often when the next stay is ED-only, the context drives the coding, and the label noise depends on it. This is direct evidence for the recording route of H. It also decides whether the method's label model must depend on the context (Section 7, item 1).

**Result (25 Oct; between-episode pairs; also adjusted for the setting at t and for a coding-system switch; patient-clustered CIs; `gonogo/part1/recording_route.py`, `recording_route.json`). Phase 0, label-only: it does not depend on the encoder.**

| Code | Resolution if t+1 is ED-only / an admission | OR ED-only vs inpatient | Onset into ED-only, OR | Coding switch, OR of onset |
|---|---|---|---|---|
| I10 | 56% / 28% | 3.2 (3.0–3.4) | 0.75 | 1.06 (0.99–1.14) |
| I25 | 90% / 16% | 44 (38–50) | 0.09 | 1.06 (0.97–1.15) |
| I50 | 95% / 18% | 95 (78–117) | 0.02 | 1.28 (1.18–1.38) |
| I48 | 73% / 13% | 22 (19–25) | 0.13 | 1.43 (1.31–1.56) |

Among the 33 pool codes, those recorded only during admissions "resolve" at every ED-only stay (median 100%, against 66% at an admission). Codes recorded in both settings resolve there 95% of the time (75% at an admission). Acute complaints go the other way: abdominal pain and headache onsets are 3.7 and 2.6 times as likely into an ED-only stay. The coding-system switch changes onset rates for 20 of the 37 codes.

The recording route is therefore present, and for chronic codes it is close to deterministic: it follows from how Ext-ICD builds labels (ED discharge codes vs hospital discharge codes). The setting at t+1 decides it and is observed, so it belongs in `z`. Whether `z` and the anchors also remove the confounding of the transition is what C1 tests.

### 2.5 Significance

- **Conditional randomization test (primary, revision 4).** Draw 1,000 fake onsets from the cross-fitted propensity at the real prevalence (the placebo of 2.3, repeated), and recompute `G_R`. This tests "Δ independent of `a_c` given the state and `Z`" and is exact when the propensity is right (logistic now).
- **Stratified permutation test.** Shuffle `a_c` within strata (Δt quintile × care setting at t+1 × age tertile) and recompute: 1,000 permutations for `G_R`, and optionally (100–500) for the MLP gain. The p-value is the fraction of permuted gains at or above the observed gain.
- **Multiple testing.** Benjamini-Hochberg across codes, q = 0.05, separately per target and condition, in two families: the positive controls, and the GO family (C1 controls, the venous set and ambiguous codes). Only codes above the event threshold enter.
- **Uncertainty.** Patient-level bootstrap (1,000 resamples) for confidence intervals on `G_c`.

### 2.6 Sanity checks (must pass before interpreting anything)

1. Every positive control with enough events shows significant gain on `Δm` or `Δb`. If not, the measurement or representation is broken; fix that first. **Passed (25 Oct; provisional, encoder `h`):** all five pass with `Z = z` (`p_crt` ≤ 0.002 on `Δm` for I44–I48, 0.001 on `Δb` for I49). On `Δh`, I44, I45 and I48 pass at 0.001 and I47 at 0.012; I49 is null on `Δh` and `Δm`, so its RR irregularity is visible only in `b`.
2. Placebo gains are centered at zero and pass the permutation null at the nominal rate. **Phase 0 (provisional):** MLP placebo gains lie within ±1.7×10⁻⁵ of zero over the 24 analyses of the permutation test (`task5_perm/summary.csv`). For the R-learner, the CRT is the placebo test itself, repeated 1,000 times. Its null mean is negative by construction (the out-of-fold cost of fitting θ), so gains are reported as excesses over it.
3. Gains do not depend on the random seed: run 3 seeds for the predictors. **Phase 0 (provisional):** the MLP gain does depend on the seed on `Δb` (spread ≈ 1×10⁻⁴, above small codes' gains), which is why the R-learner is primary (2.3). On `Δh` the spread is ≈ 3×10⁻⁶. Three seeds were used throughout.
4. Optional, as a figure for the paper: rerun one ambiguous code on all consecutive pairs, including ECGs outside stays, and show the gain that the missing-label artefacts create.

### 2.7 Interpretation matrix

| Gain on Δm / Δb | Gain on Δh | Changes with z or anchors? | Reading |
|---|---|---|---|
| yes | yes | no | **grounded** |
| no | yes | disappears with `z` | **confounded by observed context** |
| no | yes | persists with `z`, disappears with anchors | **confounded by context that anchors proxy** |
| no | yes | persists with both | unobserved confounder **or** real effect outside `m` and `b` |
| yes | no | — | encoder lost the information |
| no | no | — | ungrounded or undetectable |

The fourth row is resolved with negative controls. If non-cardiac or venous codes land there, it is confounding, because a real ECG effect is implausible for them.

### 2.8 Decision for C1

- **GO.** At least one negative control or ambiguous code shows a significant cross-fitted `G_c(Δh)` above its placebo, after Benjamini-Hochberg. That gain must also shrink substantially under `z` or under `z` + anchors. Report which conditioning set removes it:
  - Removed by `z`: observed confounding. Conditioning on `z` fixes it, and the paper must argue that such variables are often unrecorded.
  - Removed only by anchors: confounding by context that anchors proxy, the case the method is built for.

  The strongest GO combines a residual after `z` that anchors remove with a recording-route effect in 2.4.
- **NO-GO.** No code outside the positive group beats its placebo on `Δh`. A recording-route effect in 2.4 without any `Δh` gain means the labels are weak but do not confound the transition.
- **Inconclusive.** Positive controls fail the sanity check. Fix the representations and rerun; no decision is made.

**Status (25 Oct, preliminary; clean subset, R-learner with CRT; gains are excesses over the CRT null mean).** *Superseded by "Phase 0 results (provisional, encoder h)" below, which has the final numbers with CIs (for example, `z` removes 56–69% of the `Δb` gains listed here, not 58–68%). Kept for the record.*
- **Criterion 1 is met.** With Δt only, eight non-positive codes beat their null on `Δh` after BH: I10, I25, I50, I95, I85 (oesophageal varices) and three C1 controls (G43 migraine, K59 constipation, K80 gallstones).
- **Criterion 2 is met through `z`, not through the anchors.**
  - `z` removes 35–40% of the `Δh` gain of I10, I25, G43 and K59 (G43 loses significance), against 0–10% for the positive controls.
  - On `Δb`, `z` removes 58–68% of the gain of K21, K59, N40, I27 and I95, all of which lose significance.
  - Adding the anchors changes no gain (ratio 0.84–1.06 for the significant codes) and raises the propensity AUROC by at most 0.02.
- **Reading.** This is the plan's "observed confounding" outcome. The method's premise, a residual after `z` that the anchors remove, is not supported with this anchor set.
- **Drivers (2.9; `gonogo/part1/drivers.py`, `drivers.json`; drop-one shares of what `z` removes).**
  - Age and sex drive it for G43, I10, I25 and N40 on `Δh` (shares 0.75–1.0), and for N40 and I50 on `Δb`. These are patient characteristics that `h_t` encodes only partly.
  - The setting at t+1 (the recording route) drives it for K21 and K59 on `Δh` (0.60–0.65), and for K22, I95, K59 and I27 on `Δb` (0.35–0.70).
  - The driver for Part 2's semi-synthetic test is therefore either age or the setting at t+1, not only the setting.
- **Residual after `z` and anchors.** K59 and K80 keep a `Δh` gain with no `Δb` or `Δm` gain.
  - It is unchanged with an MLP propensity, although that model's AUROC is 0.02–0.07 lower than the logistic model's.
  - The R-learner is doubly robust (a null effect is biased only when both the propensity and the base model are wrong), so the residual is unlikely to be a propensity artefact. The MLP gain is less powerful, and its interval contains the residual.
  - It sits in the matrix's fourth row: context that neither `z` nor the anchors capture, or a small effect outside `m` and `b` (opioid-related constipation, acute biliary disease).
- **Adjusted subset (sensitivity: all at-risk pairs, other codes' transitions as covariates, 3–13 times more onsets).** The pattern holds.
  - With Δt only, 11 non-positive codes have significant `Δh` gains.
  - `z` removes 8–26% for the ambiguous codes and 40–74% for the C1 controls; G43 and Z96 lose significance.
  - Anchors change nothing (ratio 0.91–1.06 for the significant codes).
  - K59's residual after `z` and anchors replicates (0.7 vs 1.1 with Δt only); K80's does not.
- **Pending:** the MLP permutation test (secondary statistic).

### Phase 0 results (provisional, encoder h)

*Revision 5. All results in this section were computed with the current frozen encoder `h` (no-action ViT, run `20261019/182223`), in the state `s_t = [b_t, h_t, m_t]`. They are provisional until the encoder check of Phase 0.5. Label-only results (2.1, 2.4) do not depend on the encoder.*

*Revision 6. Results that use set A (Tables 4 and 5, and the decision labels based on them) are **exploratory**. Set A and the analyses that use it were defined after earlier Phase 0 results had been seen, and A's frequency criterion differed from the pool's. The Phase 0.5 reruns under the frozen anchor rule (0.5d) are **pre-specified**: their rules and anchor set were fixed before they ran. Where the two disagree, the Phase 0.5 result decides.*

*Revision 7. "Pre-specified" replaces revision 6's "confirmatory". The Phase 0.5 reruns use the same patients as Phase 0, so they protect against choices made after seeing results, but they do not confirm Phase 0 on independent data. Genuine confirmation needs new data. One option is MIMIC-CXR: a different modality on an overlapping cohort, with the same codes and encounters, so the linkage and the frozen rule carry over (Section 6). Another is ECGs from patients outside this dataset.*

*Conventions:*
- *Pairs are the clean subset between episodes unless stated.*
- *`E` is the R-learner gain in excess of its CRT null mean, ×10⁻⁵, with a patient-level bootstrap 95% CI. `E = 0` means as uninformative as a fake onset drawn from the propensity.*
- *`*` marks significance by the CRT (1,000 draws) after BH within its family (positive controls, or the GO family). Where no family applies, it marks CRT p < 0.05.*
- *Ratios use a paired patient-level bootstrap; `u` marks a ratio whose denominator's CI includes 0.*
- *Several gains are significant by the CRT although their bootstrap CI includes 0. The CRT conditions on the observed changes and tests for an association; the bootstrap adds the sampling variability of the effect's size.*
- *Paths are under `outputs/gonogo_part1/`.*

**1. C1 conditioning sets** (`gain/clean_bhm_<target>_<Z>_<code>.json`; "anchors" are the 11 approved anchors)

| Code | Group | Δh, Δt | Δh, z | Δh, z + anchors | Δb, Δt | Δb, z |
|---|---|---|---|---|---|---|
| I10 | ambiguous | 6.1 [2.5, 9.7]* | 4.0 [0.7, 7.2]* | 4.0 [0.7, 7.2]* | 2.9 [−2.6, 8.3]* | 2.8 [−2.6, 8.8] |
| I25 | ambiguous | 3.4 [0.7, 6.2]* | 2.1 [−0.4, 4.5]* | 2.2 [−0.3, 4.8]* | 0.8 [−3.0, 4.5] | 1.0 [−2.4, 4.5] |
| I50 | ambiguous | 3.1 [0.3, 5.7]* | 2.5 [−0.2, 5.1]* | 2.3 [−0.4, 4.8]* | 26.4 [10.3, 42.7]* | 20.9 [6.1, 35.7]* |
| I85 | ambiguous | 3.0 [−0.1, 6.2]* | 3.5 [0.5, 6.8]* | 3.5 [0.5, 6.7]* | 1.1 [−0.4, 2.6] | 1.8 [0.1, 3.7] |
| I95 | ambiguous | 2.3 [−0.2, 4.8]* | 1.8 [−0.4, 4.1]* | 1.8 [−0.5, 4.0]* | 8.0 [0.0, 15.6]* | 2.7 [−1.8, 6.9] |
| I27 | ambiguous | 0.3 [−1.6, 2.2] | 0.3 [−1.5, 2.1] | 0.5 [−1.4, 2.3] | 3.4 [−2.3, 9.4]* | 1.5 [−3.0, 6.0] |
| G43 | C1 control | 2.5 [−0.1, 5.0]* | 1.6 [−0.8, 4.1] | 1.4 [−1.1, 4.0] | 1.2 [−2.2, 4.4] | 0.7 [−2.7, 4.1] |
| K59 | C1 control | 2.9 [0.2, 5.5]* | 1.9 [−0.5, 4.1]* | 1.9 [−0.4, 4.2]* | 3.9 [−1.5, 8.9]* | 1.2 [−2.2, 4.6] |
| K80 | C1 control | 2.1 [−0.1, 4.4]* | 1.9 [−0.3, 4.2]* | 1.9 [−0.2, 4.2]* | −0.1 [−2.9, 2.5] | −0.6 [−2.5, 1.1] |
| K21 | C1 control | 1.3 [−1.4, 3.7] | −0.4 [−2.4, 1.6] | −0.4 [−2.4, 1.4] | 4.0 [−1.5, 9.1]* | 1.3 [−2.2, 4.6] |
| N40 | C1 control | 0.3 [−1.9, 2.5] | −0.5 [−2.2, 1.2] | −0.4 [−2.1, 1.3] | 6.5 [0.4, 12.6]* | 2.7 [−2.7, 9.0] |
| K22 | C1 control | −0.2 [−2.2, 1.7] | 0.0 [−2.0, 2.0] | 0.2 [−1.8, 2.2] | 9.3 [−0.4, 18.5]* | 4.7 [−2.0, 11.4]* |
| J06 | C1 control | 0.2 [−1.5, 1.9] | 0.3 [−1.5, 2.0] | 0.2 [−1.6, 1.8] | 6.5 [0.5, 12.8]* | 5.8 [0.0, 11.6]* |
| I48 | positive (reference) | 6.3 [2.7, 10.1]* | 5.7 [2.2, 9.2]* | 5.6 [2.1, 9.1]* | 157.7 [111.9, 202.0]* | 143.4 [99.0, 187.5]* |

- `z` removes 35–40% of the `Δh` gain of I10, I25, G43 and K59, and G43 loses significance.
- On `Δb`, `z` removes 56–69% of the gain of K21, K59, N40, I27 and I95, and all of them lose significance.
- The positive controls lose 0–10%.
- The anchors change no gain beyond `z` (ratio 0.84–1.06) and raise the propensity AUROC by at most 0.02. With no conditioning ("none"), the `Δh` gains match those with Δt within 0.2×10⁻⁵, except I10 (7.1 vs 6.1). The `Δb` gains differ by up to 1.9 (I50, 24.5 vs 26.4) and 3.6 for I48. The none column is in `gain/`.

**2. Drop-one drivers** (`drivers.json`, from the runs of `task2_metadata/`; five-group point shares in `drivers_dropone.json`). Shares are the part of what `z` removes that comes back when the variable alone is dropped. The table lists codes whose `z` effect has a CI excluding 0.

| Target, code | `z` effect (E[Δt] − E[z]) | Share age/sex | Share setting(t+1) | Driver |
|---|---|---|---|---|
| Δh I10 | 2.1 [0.8, 3.5] | 0.92 [0.55, 1.76] | −0.15 [−0.85, 0.10] | age/sex |
| Δh I25 | 1.4 [0.4, 2.3] | 0.75 [0.40, 1.69] | 0.23 [−0.33, 0.60] | age/sex |
| Δh K21 | 1.7 [0.3, 3.0] | 0.26 [−0.09, 0.77] | 0.65 [0.17, 1.02] | setting(t+1) |
| Δh K59 | 1.0 [0.2, 1.8] | 0.30 [−0.12, 0.99] | 0.60 [0.19, 1.47] | setting(t+1) |
| Δb I50 | 5.5 [1.1, 9.8] | 0.77 [0.26, 2.59] | 0.49 [0.10, 1.34] | both, age/sex first |
| Δb I95 | 5.3 [1.2, 9.1] | 0.02 [−0.25, 0.14] | 0.40 [0.11, 0.57] | setting(t+1) |
| Δb K22 | 4.6 [1.3, 7.6] | 0.00 [−0.26, 0.18] | 0.70 [0.42, 1.12] | setting(t+1) |
| Δb I27 | 2.0 [0.1, 4.0] | 0.33 [−0.18, 1.50] | 0.35 [−0.64, 0.91] | setting(t+1), weak |

G43's `z` effect on `Δh`, 0.9 [−0.1, 1.8], does not exclude 0, so no driver is assigned.

**3. Adjusted subset** (all at-risk pairs, other codes' transitions as covariates; `gain/adjusted_bhm_h_<Z>_<code>.json`; `Δh`)

| Code | Onsets | Δt | z | z + anchors |
|---|---|---|---|---|
| I10 | 13,423 | 2.7 [1.4, 3.9]* | 2.1 [1.0, 3.3]* | 2.0 [0.9, 3.2]* |
| I12 | 6,688 | 1.9 [0.9, 3.0]* | 1.5 [0.6, 2.5]* | 1.5 [0.6, 2.5]* |
| I25 | 8,268 | 3.9 [2.5, 5.4]* | 3.2 [1.9, 4.7]* | 3.3 [1.9, 4.7]* |
| I27 | 3,740 | 2.1 [1.1, 3.1]* | 1.8 [0.9, 2.8]* | 1.9 [0.9, 2.8]* |
| I50 | 8,944 | 10.4 [8.3, 12.5]* | 9.6 [7.5, 11.7]* | 9.6 [7.5, 11.8]* |
| I69 | 1,989 | 0.5 [−0.2, 1.1]* | 0.4 [−0.3, 1.0] | 0.4 [−0.3, 1.1] |
| I85 | 874 | 0.9 [0.1, 1.7]* | 1.1 [0.2, 1.9]* | 1.2 [0.3, 2.0]* |
| I95 | 7,554 | 1.4 [0.5, 2.3]* | 1.0 [0.2, 1.9]* | 1.0 [0.2, 1.8]* |
| G43 | 1,674 | 0.6 [−0.1, 1.4]* | 0.2 [−0.4, 0.8] | 0.1 [−0.5, 0.7] |
| K59 | 5,535 | 1.1 [0.4, 1.9]* | 0.7 [0.0, 1.3]* | 0.6 [−0.1, 1.3]* |
| K80 | 1,154 | 0.2 [−0.4, 0.8] | 0.2 [−0.4, 0.8] | 0.3 [−0.3, 0.9] |
| Z96 | 2,469 | 0.4 [−0.2, 1.0]* | 0.3 [−0.3, 0.9] | 0.2 [−0.3, 0.8] |

- The clean-subset pattern holds with 3–13 times more onsets.
- `z` removes 8–26% for the ambiguous codes and 40–74% for the C1 controls; G43 and Z96 lose significance.
- The anchors change nothing (ratio 0.91–1.06).
- K59's residual after `z` and anchors replicates; K80's does not.

**4. Task 1: label sets R and A** (`task1_labelsets/`; sets in `label_sets.json`)

The frequent non-cardiac codes are those with ≥ 200 transitions in folds ≤ 17. R holds the 297 with a plausible ECG route, each with a reason; A holds the other 208, including the whole approved pool. The table gives the fraction of the post-`z` gain each set removes.

| Code (target) | E after z | Removed by R | Removed by A | Removed by R + A |
|---|---|---|---|---|
| I10 (Δh) | 4.0 [0.7, 7.2]* | 0.03 [−0.54, 0.50] | −0.02 [−0.24, 0.15] | 0.15 [−0.29, 0.73] |
| I25 (Δh) | 2.1 [−0.4, 4.5]* | 0.23 [−0.46, 1.27] u | −0.07 [−0.54, 0.22] u | 0.20 [−0.42, 1.30] u |
| I50 (Δh) | 2.5 [−0.2, 5.1]* | 0.38 [−0.79, 1.44] u | 0.02 [−0.23, 0.24] u | 0.31 [−0.51, 1.22] u |
| I85 (Δh) | 3.5 [0.5, 6.8]* | 0.35 [0.07, 1.28] | 0.08 [−0.02, 0.35] | 0.37 [0.07, 1.35] |
| I95 (Δh) | 1.8 [−0.4, 4.1]* | 0.30 [−1.34, 2.38] u | 0.04 [−0.38, 0.83] u | 0.24 [−1.19, 1.90] u |
| K59 (Δh) | 1.9 [−0.5, 4.1]* | 0.30 [−1.88, 1.51] u | 0.15 [−1.01, 1.22] u | 0.29 [−1.52, 1.62] u |
| K80 (Δh) | 1.9 [−0.3, 4.2]* | 0.06 [−0.52, 0.57] u | 0.01 [−0.38, 0.36] u | 0.06 [−0.55, 0.66] u |
| I50 (Δb) | 20.9 [6.1, 35.7]* | 0.42 [0.23, 0.91] | 0.03 [−0.12, 0.18] | 0.39 [0.18, 0.87] |
| J06 (Δb) | 5.8 [0.0, 11.6]* | 0.10 [−0.28, 0.57] | 0.18 [−0.09, 0.69] | 0.19 [−0.18, 0.74] |
| K22 (Δb) | 4.7 [−2.0, 11.4]* | 0.75 [−1.71, 3.35] u | 0.12 [−0.83, 1.25] u | 0.72 [−1.52, 3.19] u |
| I82 (Δb) | 4.3 [−1.9, 10.4]* | 0.94 [−2.22, 3.93] u | 0.29 [−1.04, 1.66] u | 1.01 [−2.47, 4.55] u |
| I48 (Δh), reference | 5.7 [2.2, 9.2]* | 0.18 [0.06, 0.42] | −0.01 [−0.11, 0.07] | 0.22 [0.09, 0.50] |
| I48 (Δb), reference | 143.4 [99.0, 187.5]* | 0.22 [0.16, 0.30] | 0.02 [−0.02, 0.06] | 0.21 [0.14, 0.29] |
| I47 (Δb), reference | 48.6 [22.4, 74.7]* | 0.19 [0.08, 0.35] | 0.06 [−0.02, 0.15] | 0.22 [0.10, 0.40] |

- **A removes nothing after `z`.**
- **R removes part of the residual** for I85 and I50, and K59 loses CRT significance under R and R + A (1.3, not significant) while keeping it under A (1.6*).
- **R also removes 18–22% of the positive controls' effects.**

**5. Task 2: metadata substitution** (`task2_metadata/`)

"Removal" is what the withheld variable removes, E[z − v] − E[z]. The recovery columns give the fraction of it that a set restores when the variable is withheld. Rows are those whose removal has a CI excluding 0.

| Code (target) | Withheld | Removal | 11 anchors recover | Set A recovers |
|---|---|---|---|---|
| I10 (Δh) | age/sex | 1.94 [0.93, 2.92] | 0.07 [−0.16, 0.30] | 0.07 [−0.27, 0.41] |
| I25 (Δh) | age/sex | 1.03 [0.44, 1.66] | −0.03 [−0.40, 0.24] | 0.00 [−0.46, 0.45] |
| G43 (Δh) | age/sex | 0.87 [0.03, 1.70] | 0.14 [−0.31, 0.91] | −0.07 [−1.53, 0.76] |
| I50 (Δb) | age/sex | 4.21 [1.34, 7.02] | 0.45 [−0.04, 1.16] | 0.37 [−0.44, 1.35] |
| K21 (Δh) | setting(t+1) | 1.07 [0.06, 2.05] | 0.28 [−0.06, 1.32] | 0.73 [−0.05, 2.63] |
| K59 (Δh) | setting(t+1) | 0.61 [0.11, 1.14] | 0.01 [−0.79, 0.65] | 1.02 [−0.08, 3.49] |
| I50 (Δb) | setting(t+1) | 2.70 [0.47, 5.17] | 0.44 [−0.49, 1.60] | 0.42 [−1.24, 2.16] |
| I95 (Δb) | setting(t+1) | 2.12 [0.11, 4.05] | 0.50 [0.05, 1.29] | 1.30 [0.38, 2.83] |
| K22 (Δb) | setting(t+1) | 3.21 [0.98, 5.58] | 0.30 [−0.12, 0.69] | 0.69 [0.03, 1.42] |
| I10 (Δh) | all of z | 2.11 [0.82, 3.47] | 0.06 [−0.19, 0.37] | 0.17 [−0.27, 0.56] |
| I25 (Δh) | all of z | 1.37 [0.44, 2.31] | 0.02 [−0.29, 0.26] | 0.19 [−0.19, 0.56] |
| K21 (Δh) | all of z | 1.66 [0.29, 2.98] | 0.19 [−0.08, 0.64] | 0.68 [0.28, 1.95] |
| K59 (Δh) | all of z | 1.02 [0.21, 1.85] | 0.09 [−0.63, 0.48] | 0.80 [0.16, 2.34] |
| I50 (Δb) | all of z | 5.49 [1.12, 9.79] | 0.70 [0.31, 2.50] | 0.88 [0.34, 3.40] |
| I95 (Δb) | all of z | 5.27 [1.23, 9.12] | 0.34 [0.14, 0.62] | 1.00 [0.68, 1.53] |
| K22 (Δb) | all of z | 4.57 [1.33, 7.59] | 0.37 [0.11, 0.77] | 0.79 [0.28, 1.37] |

- **Set A recovers a substantial share of what the setting removes.** The primary reading is the setting-only withdrawal (`z` − setting(t+1) → `z` − setting(t+1) + A): point estimates 0.42–1.30, with CIs excluding 0 in 2 of 5 rows (I95, K22). With all of `z` withheld, point estimates are 0.68–1.00, with CIs excluding 0 for five of five setting-driven rows; this reading is secondary, because withholding all of `z` also withholds age/sex, which A never recovers. *(Revision 6: the earlier wording, "recovers most", overstated the primary reading. These results are exploratory.)*
- **The 11 anchors recover a third to two thirds** for three `Δb` rows and nothing on `Δh`.
- **Neither set recovers age/sex.**

**6. Task 3: laboratory and medication covariates** (`task3_diag/`; covariates in `pair_covariates.npz`; `Δh`)

Potassium, magnesium and TSH within ±24 h of each ECG, and opioid and laxative administration, at t and t+1. Coverage is K 97%, Mg 73–74%, TSH 11–12%, opioid 43% and laxative 53%. These may be post-event variables, so they are diagnostic only.

| Code | E after z (AUROC) | E after z + covariates (AUROC) | E after z + anchors + covariates (AUROC) | Removed by covariates | Removed by anchors + covariates |
|---|---|---|---|---|---|
| K59 | 1.9 [−0.5, 4.1]* (0.690) | 1.6 [−0.6, 4.0]* (0.788) | 1.6 [−0.6, 3.9]* (0.795) | 0.14 [−0.86, 1.23] u | 0.14 [−1.09, 1.23] u |
| K80 | 1.9 [−0.3, 4.2]* (0.666) | 1.6 [−0.6, 3.8]* (0.726) | 1.8 [−0.4, 4.1]* (0.729) | 0.15 [−0.45, 1.30] u | 0.06 [−0.42, 0.81] u |
| I48, reference | 5.7 [2.2, 9.2]* (0.793) | 5.0 [1.7, 8.4]* (0.810) | 5.1 [1.6, 8.4]* (0.812) | 0.12 [0.00, 0.31] | 0.11 [−0.02, 0.32] |

The covariates predict K59 strongly (propensity AUROC 0.69 → 0.79) but remove 14% of its residual, with a CI including 0. K59 and K80 stay significant.

**7. Task 4: propensity models** (`task4_gbm/`, `gain_pmlp/`; `Δh`, `z` + anchors)

| Code | Logistic: AUROC, E | MLP: AUROC, E | Gradient boosting: AUROC, E |
|---|---|---|---|
| K59 | 0.710, 1.9 [−0.4, 4.2]* | 0.687, 2.0 [−0.4, 4.3]* | 0.687, 1.8 [−0.6, 4.1]* |
| K80 | 0.672, 1.9 [−0.3, 4.2]* | 0.636, 1.8 [−0.3, 4.2]* | 0.636, 1.8 [−0.3, 4.2]* |

- **Neither flexible model beats the logistic propensity.** The agreement of their AUROCs at three decimals is a coincidence: they are 0.6874 and 0.6874 for K59, and 0.6356 and 0.6357 for K80.
- **The residual is the same under all three.**
- The MLP column used the anchors standardized with the state (`za`); the other two used 0/1 anchor columns (`z+anc`).

**8. Task 5: deliverables**

| Deliverable | Result | Path |
|---|---|---|
| Positive-control effect vectors in `h` | Debiased squared norms in z-units (split halves, folds 0–9 vs 10–19): I48 0.69 [0.14, 1.29], the only one whose CI excludes 0. I44 1.90 [−0.56, 4.80], I45 2.41 [−0.10, 5.48], I47 0.42 [−0.68, 1.56], I49 0.66 [−1.73, 2.99]. Plug-in norms are 1.6–4.8 times larger. | `effects_h.npz`, `effects_h.json`, `task5_effects/` |
| Noise floor (mean squared change relative to between-episode pairs) | `Δh`: stable 0.88, changed 1.05, same episode 0.72. `Δb`: 0.72, 1.12, 0.88, within stay 1.00. `Δm`: 0.68, 1.14, 0.94, within stay 0.99. | `noise_floor.json` |
| Variance decomposition (cross-fitted R², between-episode pairs) | `Δh`: gap 0.0001, codes beyond gap +0.0002, current state +0.13. `Δb`: 0.0024, +0.0041, +0.25. `Δm`: 0.0022, +0.0064, +0.21. Patient share of Δ ≈ 0; patient share of `h` 0.74. | `noise_floor.json` |
| MLP gain, stratified permutation test (`Δh`, 100 permutations) | After `z`: I50 p 0.01, I85 0.02, K59 0.04, K80 0.04, I95 0.05; I10, I25, G43 not significant. Positive controls 0.01–0.06. With 100 permutations the BH floor is q = 0.08 for 8 codes. | `task5_perm/summary.csv` |
| Drivers | Table 2 above | `drivers.json` |

**C1 outcome (provisional, encoder h).**
- **GO for confounding by observed context**, by the first branch of 2.8 ("removed by `z`: observed confounding").
  - Codes with no plausible ECG route, and ambiguous codes, have significant `Δh` gains that `z` partly removes, while the positive controls' gains do not move.
  - The drivers are age/sex and the care setting at t+1.
- **No code-label set removes the residual after `z`.** Not the 11 anchors, not the no-route set A, not the route set R (which only weakens it, while also removing part of real effects), and not laboratory or medication covariates.
- **The residual is unexplained.** It is robust for K59 only: in both subsets, under three propensity models, and after every covariate set. K80's residual does not replicate in the adjusted subset.
- **The strongest GO of 2.8 is not met.** That would be a residual after `z` that anchors remove, together with the recording-route effect of 2.4. The recording-route effect is present, but no anchor set removes the residual.

**Decision labels of the Phase 0 task (corrected).** The Phase 0 task defined four rows:
- (a) anchors recover a substantial share of withheld metadata: ambitious framing kept;
- (b) the residual after `z` is removed by ECG-route codes (R) only: co-condition confounding, modest framing;
- (c) the residual is removed by no-route codes (A): a larger anchor set is the fix, and the method survives;
- (d) nothing removes it: report an unexplained residual and re-scope per Section 6.

The outcomes:
- **Row (a) holds with the full no-route label set A as anchors, not with the 11 curated anchors.**
  - A recovers a substantial share of the withheld setting (point estimates 0.42–1.30; CIs exclude 0 in 2 of 5 rows). This is the primary reading, the setting-only withdrawal `z` − setting(t+1) → `z` − setting(t+1) + A (Table 5). With all of `z` withheld, the recovery is 0.68–1.00 (secondary). *(Revision 6 wording; exploratory. The setting-only withdrawal is pre-registered as primary for the 0.5a reruns.)*
  - The 11 anchors recover nothing on `Δh`, and a third to two thirds on three `Δb` rows.
- **Row (c) does not hold:** A removes nothing after `z` (Table 4).
- **Row (b) holds only in part.** R removes 0.35 [0.07, 1.28] of I85's `Δh` residual and 0.42 [0.23, 0.91] of I50's `Δb` gain. It also removes 0.18–0.22 of I48's real effect, so co-condition confounding and over-adjustment cannot be separated.
- **Row (d) applies to K59's residual**, robustly, and to K80's, which is not robust.
- **Age and sex are never recovered** by any label set, and they are always observed.

The earlier Phase 0 report assigned row (c) to the substitution result; that was a mislabeling, corrected here.

**Findings that change the method or Part 2.**
- **Conditioning on ECG-route codes (R) removes 18–22% of I48's effect.** The figures are `Δh` 0.18 [0.06, 0.42] and `Δb` 0.22 [0.16, 0.30], and I47 loses 0.19 [0.08, 0.35] on `Δb`. These data cannot distinguish over-adjustment, through mediators or co-consequences, from genuine confounding by those conditions. So effect magnitudes must be reported with and without route codes in the conditioning set.
- **Laboratory and medication covariates remove 12% [0, 31%] of I48's effect.**
- **Only I48's effect vector in `h` has a CI excluding 0.** Part 2 injects it, scaled, unless Phase 0.5 selects another representation.
- **Code transitions explain almost nothing of `Δh`.** The 152 cardiac code transitions explain R² ≈ 0.0002 of `Δh` beyond the gap, against 0.13 for the current state. The patient share of `h` is 0.74.
- **I48's gain is about 25 times larger on `Δb` than on `Δh`** per target dimension: 143 [99, 188] vs 5.7 [2.2, 9.2] after `z`. With the two findings above, this suggests the encoder suppresses within-patient change, which motivates the encoder check of Phase 0.5.
- **Anchor sets need to be broad, and they work as substitutes, not as removers.** A large, rule-based no-route set recovers the withheld setting, while a short curated list does not. Neither removes what remains after `z`. The method's anchors and its residual-bias margin are revised accordingly in `method_section.tex` (Section 7, item 10).

### 2.9 Outputs (inputs to Part 2)

- Per-code table: `G_c` in the three conditioning sets for each target, placebo gain, p-values, CIs, matrix row.
- Effect sizes for positive controls: per dimension in ms or SD for reporting, and as effect vectors in `h` (the matched `Δh` differences). Part 2 injects these vectors as grounded effects.
- **Confounding drivers.** For codes in the "confounded" rows, find which observed context variables account for the gain that disappears with `z`. Use a drop-one analysis: recompute `G_c` with each variable of `z` removed in turn. The strongest driver (e.g. ED-only vs inpatient) becomes the assignment mechanism for confounded codes in the semi-synthetic test.
- Recording-route results (2.4): flicker rates by care setting, per chronic code.
- Fraction of total `Δh` variance explained by patient identity versus Δt versus codes. Stable fraction on cross-stay pairs between episodes: 29.3% (revision 3's 48.3% counted the same-episode pairs). Noise floor from within-stay and same-episode pairs: their change relative to between-episode pairs. These set patient dominance, the stable fraction and the noise level in the latent generator.

**Estimated cost:** 5–7 days. This includes the downloads and the linkage of ECGs to stays and care settings; the analyses on `b` and `h` with Δt can start immediately.

---

## Phase 0.5: checks before Part 2 (revision 5)

Phase 0 leaves three questions that Part 2 depends on:
- whether the representation carries grounded effects well enough to inject and detect them;
- what the no-route set A recovers when encounter metadata are withheld;
- how large the unexplained residual is in the units of the grounding test.

Each check below has a decision rule fixed before it runs. The rules were revised once, in revision 6 (3 Nov), before any Phase 0.5 result existed, and they are not revised once a Phase 0.5 statistic has been computed. Every rerun uses 500 permutations where a permutation test is involved.

**Order (revision 6).**
1. 0.5d: the anchor rule is frozen before any Phase 0.5 statistic.
2. 0.5b statistic 1, which is label-only.
3. 0.5a: the encoder candidates, then the encoder check.
4. On the chosen representation: the reruns of 0.5a and statistic 2 of 0.5b.
5. 0.5c.

Each run gets its own folder under `outputs/phase05/`, and `outputs/phase05/RECOVERY.md` records progress so that work can resume across short allocations.

**Status of evidence (revision 6, wording revised in 7).** Phase 0 results that use set A are exploratory. The Phase 0.5 reruns are pre-specified, not confirmatory: they use the same patients as Phase 0, and genuine confirmation would need new data, for example MIMIC-CXR (see the header notes of "Phase 0 results").

**Estimated cost:** 3–4 days, most of it training the encoder of 0.5a.

### 0.5a Encoder check

**Candidates (revision 7).** Only representations that satisfy the method's encoder requirement are eligible (§7 item 10; `method_section.tex`, §wm-setting).
- **Eligible:**
  - **`g`**, an encoder trained on single recordings without same-patient positives (details below);
  - **`[g; b]`**, its concatenation with the median-beat features;
  - **`b`**, the median-beat features alone.
- **Reference only, not eligible:** the current `h` (the no-action ViT, which predicts a patient's next recording from the current one, an objective that rewards invariance across the patient's recordings) and `[h; b]`.
  - Both are computed alongside the eligible candidates. They serve as the empirical check of the method's exclusion of same-patient objectives.
  - If `h` beats `g` on folds 0–9, that is reported as a finding that contradicts the method's encoder requirement. `h` is not chosen in either case.
- **The encoder `g`, as implemented** (candidate (ii) of revision 5):
  - It is the ViT of run `20261019/182223`, retrained with only the objective changed. Backbone, data, folds, epochs, optimizer and seed are the same.
  - The objective is SIGReg on single ECGs: the positive pair is two views of one recording, never two recordings of one patient (`scripts/pretrain.py ++objective=single`; run `gorgonoid/dynamics_single.sbatch`; handoff in `outputs/phase05/3b_encoder_single/HANDOFF.md`).
  - **Views (revision 7):** one global amplitude factor for all leads within ±5% (not per lead, and not ±20% as on 3 Nov), a time shift of up to ±0.5 s, Gaussian noise at 5% of each lead's SD, and one 0.5 s mask. There is no time stretch or resampling, which would make the encoder invariant to heart rate.
  - **Why:** SIGReg with a view-prediction loss learns invariance to whatever the views vary, and amplitude and rate changes are part of the signal here (voltage, heart rate), so the views vary them as little as possible. The two runs of 3 Nov used ±20% and are invalid.
  - If a public pretrained model is used instead, check whether its pretraining data include our patients and report the overlap (under cross-fitting, every patient is a test patient for some fold). Exclude models pretrained with any MIMIC labels.

*Revision 6 listed the current `h` and `[h; b]` as candidates and ranked `[h; b]` first among ties, which conflicted with the method's exclusion of same-patient objectives. Superseded by the eligible set above and kept for the record.*

**Sanity rule for candidate encoders (revision 9; declared on 9 Nov 2026 before any new number on g was looked at).**
- **Entry.** A candidate encoder enters the 0.5a comparison only if its diagnostic probe AUROC is within 0.05 of the reference h. The probe is the same script, labels, folds and protocol used for h's reported AUROC. The diagnosis identifies the protocol and checks it on h before it is applied to g.
- **Failure label.** An encoder that fails this check is recorded as "training failed", not as "lost the comparison".
- **Retrain.** The training run is allowed one retrain, and only if the diagnosis identifies a concrete cause. The retrain's objective is fixed in the plan before it is launched.
- **Second failure.** If the retrained encoder also fails, b is the chosen representation, and the encoder ablation is reported as a limitation.
- **Checkpoint.** Any encoder uses the last checkpoint of training, not the val-loss-selected one. The exception is when the diagnosis shows the last checkpoint is itself degenerate; that is reported and the analysis stops.
  - *Note:* h's existing export (`h_zeroactions.npz`, used in Phase 0 and in the 0.5a references) is from h's val-loss-selected checkpoint (epoch 339 of 405). Applying this rule to h too would mean re-exporting h from its last checkpoint and recomputing the references. This is an open question for the user, and nothing has been re-exported.
- **Pipeline bugs.** If the diagnosis finds a pipeline bug rather than a training failure, the bug is fixed, g is re-exported and re-probed, and this rule is applied. No retrain is used in that case.

*Clarification (revision 9, 9 Nov, recorded after the diagnosis of g and before any result of the intra retrain):*
- **Checkpoint scope.** "Last checkpoint" applies to the candidate encoders trained for the 0.5a comparison (g, and the intra retrain). h remains the frozen Phase 0 reference at its existing epoch-339 export (`h_zeroactions.npz`) and is not re-exported. This settles the note above.
- **The probe** is the offline clean linear probe of the diagnosis (`gonogo/phase05/diag_g.py probe`: test fold 19, macro AUROC).
- **The bar is frozen at 0.529:** h's offline probe, 0.579, minus 0.05. The decision uses the point estimate, not the confidence interval.
- **What the probe can detect.** The random-init floor (0.608) exceeds h (0.579) on this probe, so the probe screens for destructive training only. It cannot show that an encoder learned something useful. The decisive comparison remains ψ̂² in 0.5a.

**Statistic.** For each positive control, the debiased squared effect norm ψ̂²: the split-half inner product, as in `gonogo/part1/effects.py`, with a patient-level bootstrap CI. It is measured in units of the noise floor, the within-patient covariance of the change on stable between-episode pairs (Σ of `method_section.tex`), so that representations of different dimension are comparable.

**Metric (revision 8, declared before any result on g).**
- **Primary.** Σ is estimated with Ledoit–Wolf shrinkage: the analytical intensity, toward a scaled identity (`sklearn.covariance.LedoitWolf`). It is applied to the stable-pair changes after standardizing each dimension by its stable-pair SD, so in raw units the target is the diagonal of the stable-pair variances.
  - Standardizing first makes the metric independent of each feature's units (b mixes bpm, ms and mV). A scaled identity in raw units would not be.
  - ψ̂² uses the inverse of this estimate. There is no tuning parameter, and the intensity is recorded per representation.
- **Where Σ is estimated.** Separately for each representation, cross-covariances of a concatenation included, on the stable usable between-episode pairs of the folds the statistic uses. That is folds 0–9 for selection and folds 10–19 for reporting; for an all-fold statistic (0.5c), all folds.
- **Sensitivity, always reported next to the primary:** the pseudo-inverse of the sample covariance on its principal subspace holding 99.9% of the stable-pair variance (revision 7's metric), and the full inverse.
- **Decisions use the primary metric only.** This covers qualification and choice in 0.5a, and the margin and comparisons in 0.5c. If the primary and the pseudo-inverse disagree on which candidates qualify, the disagreement is reported and the primary decides.
- **The same metric everywhere Σ appears:** ψ̂² of the positive controls, ψ̂² of the withheld codes and Γ̂ (0.5c), and the tested codes.
- **Concatenation.** In population, the Mahalanobis norm of a concatenation is at least that of each part, because the extra block adds a non-negative Schur-complement term. So ψ²([g; b]) ≥ ψ²(g) and ψ²(b) by construction. The choosing rule therefore weighs the information a concatenation adds against the estimation noise its extra dimensions add, which widens its intervals. A concatenation can lose to a part only through that noise.
- **States (revision 8).**
  - g and [g; b] use the state [b, g, m].
  - b uses [b, m], without h.
  - The references h and [h; b] use [b, h, m].
  - Z = z throughout.
  - Every state is that of the candidate's own representations, so the excluded encoder h never enters an eligible candidate's estimate.

*Revision 7 had made Σ⁻¹ the 99.9% principal-subspace pseudo-inverse, estimated over all folds. That choice was made after the folds-0–9 references had been seen, and it changed whether b qualified (2 → 3 of 5). Superseded by the metric above (it remains a sensitivity analysis) and kept for the record.*

*Revision 6:* it is computed on the clean subset (primary) and, as a power check, on the adjusted subset. Within each half of the folds used below, the two independent estimates come from quarters of the folds: 0–4 vs 5–9 for selection, 10–14 vs 15–19 for reporting.

*Candidates after the retrain (revision 9, 9 Nov; recorded before any intra result).*
- g failed the sanity rule ("training failed") and leaves the 0.5a comparison.
- The intra-recording encoder (`++objective=intra`) replaces it. The eligible candidates are [intra; b], intra and b. Ties are broken in that order: **[intra; b], then intra, then b**. This is the order of the binding decision rule below, with the concatenation first.
- Wherever this section names g or [g; b], read intra or [intra; b]. Their state is [b, intra, m].
- intra enters the comparison only if it passes the sanity rule (offline probe ≥ 0.529, point estimate). If it fails, b is the chosen representation.
- The references h and [h; b] are unchanged.
- *Correction (9 Nov, before any intra result).* An earlier edit of this block set the order to intra, then [intra; b], then b, reversing the binding rule by mistake. That reversed order was introduced and withdrawn before any intra result existed. The binding order stays concatenation first, as in the decision rule below, the "b no longer depends on g" note and §7 item 12.

**Decision rule (revision 6; candidates and tie-break revised in revision 7).**
- **Choosing, on folds 0–9 only.** Among the eligible candidates (`[g; b]`, `g`, `b`) on which at least three of the five positive controls have a lower bound above zero, choose the one with the largest median ψ̂². When the paired patient-bootstrap CI of the difference in median ψ̂² between the leader and another qualifying candidate includes 0, they are tied. Ties are broken in this order: `[g; b]`, then `g`, then `b`. The references `h` and `[h; b]` are reported on the same folds with the same statistic, and never chosen.
- **Reporting, on folds 10–19.** The positive controls of the chosen candidate are reported on folds 10–19, which the choice did not use. Positive-control results on the selection folds, or pooled over all folds, are checks of the choice, not evidence for it. `b` stays the validation readout: whatever is chosen, the positive controls are also reported on `Δb`, where their effects were established (2.6).
- **If none qualifies.** If no eligible candidate reaches three on folds 0–9, the rule of 3.6 applies: the method cannot be demonstrated on real codes, and Part 2 becomes the main evidence. This holds even if a reference reaches three.
- **Rerun.** Then rerun short versions of Tasks 1 and 2 on the chosen representation, using the frozen set of 0.5d. The codes are those significant after `z` in Phase 0 (Tables 1, 4 and 5), together with any code significant after `z` on the chosen representation.
  - Task 1 under `z`, `z+R` and `z+A`.
  - Task 2 under Δt, Δt+A, `z` − setting(t+1), `z` − setting(t+1) + A, and `z`. **Pre-registered primary reading (revision 6):** the setting-only withdrawal, `z` − setting(t+1) → `z` − setting(t+1) + A. Δt → Δt + A and the all-of-`z` withdrawal are secondary.
  - Use 3 seeds and 1,000 CRT draws.
- **Revision.** If those conclusions change, revise the Phase 0 decision labels before Part 2.

*Revision 5 had a different choosing rule: no fold split, and ties broken toward "the simpler candidate", `b`, then `[h; b]`, then a new encoder. It is superseded and kept for the record.*

### 0.5b Provenance check

**Statistics (revision 6).**
1. **Label-only.** The cross-fitted AUROC of predicting the setting at t+1 (ED-only vs admission) from the transitions of set A alone, and, for comparison, from the 11 anchors of the 25 Oct approval. Features are onset and resolution indicators; the model is logistic, cross-fitted over the 20 folds, with a patient-level bootstrap CI. It does not depend on the encoder.
2. **Absolute gain reduction by A.** D_A = E[Δt] − E[Δt + A], the excess gain with Δt only minus that with Δt and A. It is computed within pairs of constant setting (both stays admissions, and both ED-only where onsets allow) and over all pairs, on the chosen representation. The codes are the setting-driven ones: Task 2 rows of the 0.5a rerun whose setting-only removal has a CI excluding 0, or the Phase 0 rows (K21 and K59 on `Δh`; I50, I95 and K22 on `Δb`) if the rerun has none. CIs come from a paired patient-level bootstrap.

*Revision 5 recomputed Task 2's recovery ratio within constant-setting pairs. Its denominator is what withholding the setting removes, which is zero by construction when the setting does not vary, so the ratio was undefined. It is superseded and kept for the record.*

**Decision rule (revision 6).**
- **Provenance only.** A predicts the setting at t+1 at AUROC ≥ 0.9, and D_A within constant-setting pairs has a CI including 0 or a point estimate below a quarter of D_A over all pairs. Then A recovers label provenance only, and the method's claim narrows to "the label vector recovers label provenance when encounter metadata are missing", the modest framing.
- **Beyond provenance.** D_A within constant-setting pairs has a CI excluding 0 and is at least a quarter of D_A over all pairs. Then A carries context beyond provenance, and the ambitious framing is kept.
- **Aggregation.** Each branch is applied to each setting-driven code.
  - The modest framing follows when "provenance only" holds for every such code.
  - The ambitious framing follows when "beyond provenance" holds for a majority of them.
  - Anything else, including an AUROC below 0.9 with neither branch met, is reported per code as mixed, and the modest framing is used.

### 0.5c Γ̂ by rotation

**Procedure (revision 6).**
1. On the chosen representation, use the five rotation groups of the frozen no-route set (`rotation_groups` in the frozen file of 0.5d). They were drawn at freeze time with seed 0, stratified by clean onsets. The C1 controls are excluded and keep their validation role.
2. For each group, deconfound with `z` plus the other groups, and give the withheld codes code channels. The code under test is never among the inputs of the deconfounding.
3. Compute each withheld code's residual excess gain and debiased ψ̂²_n in noise-floor units, together with its patient-bootstrap variance v_n.
4. **Random-effects margin.**
   - Model each estimate as ψ̂²_n = ψ²_n + e_n. Here ψ²_n is code n's true residual, e_n is estimation noise with variance v_n, and the true residuals vary across codes with mean μ and between-code variance τ².
   - Estimate μ and τ² with DerSimonian–Laird. The fixed-effect weights are w_n = 1/v_n and Q = Σ w_n (ψ̂²_n − μ̂_FE)². Then τ̂² = max{0, (Q − (N − 1)) / (Σ w_n − Σ w_n² / Σ w_n)}, and μ̂ is the weighted mean with weights 1/(v_n + τ̂²).
   - Set Γ̂² = μ̂ + 1.645 τ̂, the 95th percentile of the true residuals under a normal random-effects model. When τ̂ = 0, set Γ̂² to the upper 95% bound of μ̂, μ̂ + 1.645 SE(μ̂).
   - **Why.** A quantile of the point estimates (revision 5's 95th percentile of ψ̂²_n) mixes the spread of the true residuals with each withheld code's estimation noise. At a few hundred onsets per code, the noise dominates. The tested code's lower confidence bound already accounts for that code's own estimation noise, so the margin should describe true residuals only; otherwise noise is counted twice and the test loses power for no gain in validity. The cost is a distributional assumption: the margin protects codes whose residual lies within it, not the worst case (`method_section.tex`, prop:wm-sensitivity).
5. **Secondary margins by recording profile.** Repeat step 4 within the "admission" and the "both" category, using 2.2's rule ("both" if at least 25 onsets go into ED-only stays). Compare each tested code also with the margin of its own recording profile. Decisions use the pooled margin.
6. Compare every tested code against the pooled margin: the positive controls, the ambiguous codes and the C1 controls. A code clears it when its ψ̂² lower bound exceeds Γ̂². Positive controls are compared on folds 10–19, which 0.5a did not use for the choice, and are also reported on all folds as a check. The other codes are compared on all folds.
   - *Power note (revision 7).* On folds 10–19 the positive controls' ψ̂² are quarter-fold estimates (10–14 vs 15–19). They rest on half the onsets of an all-fold estimate: 20–190 per quarter for I44–I49 on the clean subset, judging from folds 0–9. Their intervals are therefore wide, and "too little power" is a likely outcome for the smaller controls. The adjusted subset (3–13 times more onsets) is reported alongside as a power check. Decisions stay on the clean subset.
   - All ψ̂² here, for withheld and tested codes alike, use the primary metric of 0.5a: Ledoit–Wolf shrinkage of the standardized stable-pair changes (revision 8; revision 7 had the principal-subspace pseudo-inverse). Σ is estimated on the stable pairs of the folds each statistic uses, so the margin and the tested codes are in the same units. The two sensitivity metrics of 0.5a are reported alongside, and decisions use the primary.

*Revision 5 set Γ̂² to the 95th percentile of the withheld codes' ψ̂² over all rotations. That rule is superseded and kept for the record.*

**Decision rule.**
- **Usable.** If at least two positive controls clear the margin and no C1 control does, the grounding test is usable on real codes.
- **Too little power.** If no positive control clears it, the margin absorbs the effects at current power. Real codes are then reported as inconclusive, and the paper rests on the benchmark (3.6).
- **Margin too small.** If a C1 control clears it, Γ̂ underestimates the residual, because withheld and tested codes are not comparable. Revisit the anchor rule before Part 2.

### 0.5d Frozen anchor rule (revision 6; runs before every Phase 0.5 statistic)

**Frozen 3 Nov 2026, 15:16 (−03), before any Phase 0.5 statistic.**
- **File:** `outputs/phase05/freeze/label_sets.json`, SHA-256 `c1feffa3508d9070af4994544c08d62efcc78d0a1e48732e1716b3816cf4bce4` (also in `label_sets.sha256`). It was written by `gonogo/phase05/freeze_label_sets.py`, which refuses to overwrite it.
- **Inputs:** label statistics only, all computed on 25 Oct before any gain. These are `label_sets.json` of Phase 0 (frequency and route rule) and `control_pool.csv` (clean onsets, ED-only onsets, separation odds ratio).

**Rule.**
1. **Frequency (one criterion):** ≥ 200 transitions (onsets plus resolutions) on the usable between-episode pairs of folds ≤ 17. This is Phase 0's criterion for A, unchanged. It was chosen over "≥ 250 clean onsets", which would leave 38 no-route codes and about 26 to rotate:
   - The anchors work as many weak proxies jointly, and breadth is what separated A from the 11 curated anchors in Phase 0. Under the clean-onset criterion, each rotation would deconfound with about 21 codes, close to the 11-anchor regime that failed.
   - The context estimator receives onset and resolution indicators, so transitions in both directions are the right count for a proxy. Clean onsets are the right count for a tested code.
   - The pre-specified reruns then test the set whose exploratory results motivated them, not a different, smaller one. (The frozen file's justification text says "confirmatory reruns"; read it as pre-specified, per revision 7. The file is frozen and not edited.)
   - The low precision of rare withheld codes is absorbed by the random-effects margin of 0.5c.
   - Cost: most rotated codes have fewer than 100 clean onsets, so their ψ̂² are imprecise.
2. **Route:** `label_sets.assign` (the route reasons and chapters of Phase 0).
3. **Separation screen on every candidate:** the Mantel–Haenszel OR of a grounded cardiac onset (I44, I45, I47, I48, I49, I21, I26) on pairs with the candidate's onset, stratified by setting(t+1) × age tertile × sex, over all at-risk between-episode pairs. A candidate fails if OR > 1.5, and a failing code moves to R.
4. **Recording profile:** "both" if at least 25 clean onsets go into ED-only stays, otherwise "admission".
5. **Roles:** the 11 C1 controls are the validation subset and never an input. Every other code is rotated.
6. **Rotation groups:** five groups, stratified by clean onsets (blocks of five, seeded permutation, seed 0).

**Result of the rule** (label statistics, not a Phase 0.5 result).
- 208 candidates. 15 fail the screen: B00, B02, D18, D32, H90, J38, J39, K02, L76, N99, S37, T17, Z00, Z53, Z78, with OR 1.52–2.62.
- C0 = 193 codes: 25 "both" and 168 "admission".
- 11 validation codes and 182 rotated codes, in groups of 37, 36, 36, 36 and 37.
- R = 312 codes: Phase 0's 297 plus the 15 screen failures.
- The screen's OR is noisy for rare codes. 15 of 208 candidates fail it, against 1 of the 38 frequent ones (Z78), so some fail by chance; failing only removes a candidate.
- **A for every Phase 0.5 rerun** is C0 without the validation subset, and without the code under test. Phase 0's A included the C1 controls and the 15 screen failures, so the reruns differ from Phase 0 in those 26 codes.
- `gonogo/method_appendix_anchors.tex` (`sec:wm-app-anchors`) is generated from this file by `gonogo/phase05/make_anchor_appendix.py`.

**Supersedes** the anchor and held-out-anchor roles of the 25 Oct approval (2.2). That record is kept in `control_roles.json`, and the old lists are also stored in the frozen file under `superseded_2026_10_25`. The 11 old anchors are used once more, only as the comparison of 0.5b statistic 1.

### Phase 0.5 results (pre-specified, same patients as Phase 0; filled as they come, paths under `outputs/phase05/`)

**0.5b statistic 1, label-only (3 Nov 15:21, after the freeze; `3a_provenance_auroc/`, script `gonogo/phase05/provenance_auroc.py`).**
- Pairs: 118,670 usable between-episode pairs, 19.5% with an ED-only stay at t+1, 42,832 patients.
- Method: L2 logistic regression on onset and resolution indicators, cross-fitted over 20 folds, with a patient bootstrap (1,000 resamples).

| Set | Codes | AUROC, setting(t+1) ED-only vs admission | Pairs with any transition of the set |
|---|---|---|---|
| A (frozen C0 without the validation subset) | 182 | 0.817 [0.814, 0.820] | 80% |
| C0 (validation subset included) | 193 | 0.852 [0.849, 0.854] | 86% |
| 11 anchors of the 25 Oct approval | 11 | 0.654 [0.649, 0.658] | 43% |

- **A does not reach AUROC ≥ 0.9.** The "provenance only" branch of the 0.5b rule therefore cannot hold, whatever statistic 2 shows. The framing is decided by statistic 2 on the chosen representation: "beyond provenance" for a majority of setting-driven codes keeps the ambitious framing, and anything else is mixed, with the modest framing.
- The label vector reveals the setting only in part. The validation codes, which are frequent and recorded in both settings, add 0.035.

**0.5b statistic 2, power pre-check (revision 7; information only, the rule is unchanged; `4a_power_da/`, script `gonogo/phase05/power_da.py`).**
- **Method.** Uses the Phase 0 per-pair gains of Task 2 (Δt and Δt+A, current state, Phase 0's A). It computes only CI half-widths of D_A and onsets within constant-setting pairs, never point estimates there. Units are excess gain ×10⁵ with a patient bootstrap. "Expected D_A" is Phase 0's all-pairs D_A scaled by the onset density of the subset, which is what it would be if A's reduction persisted in full.

| Code (target) | All pairs: onsets, ±half-width | Phase 0 D_A, all pairs | Both admissions: onsets, ±half-width | Both admissions: expected D_A, power | Both ED-only: onsets, ±half-width |
|---|---|---|---|---|---|
| K21 (Δh) | 2,717, ±0.78 | 1.12 | 1,979, ±1.12 | 1.42, 0.65 | 70, ±0.72 (power 0.06) |
| K59 (Δh) | 1,384, ±0.70 | 0.81 | 1,152, ±1.07 | 1.12, 0.54 | 42, ±0.67 (0.06) |
| I50 (Δb) | 758, ±3.18 | 4.81 | 676, ±5.04 | 7.20, 0.79 | 12, not estimable |
| I95 (Δb) | 1,656, ±3.97 | 5.29 | 1,452, ±5.99 | 7.36, 0.67 | 25, ±3.20 (0.05) |
| K22 (Δb) | 405, ±3.08 | 3.63 | 345, ±4.35 | 5.08, 0.60 | 3, not estimable |

- **Reading.** "Beyond provenance" for a majority (3 of 5) is attainable only if A's reduction largely persists within constant setting: the chance is 0.77 at full persistence and 0.07 at half. Power comes from the both-admissions pairs; the both-ED-only pairs are too sparse. A partial persistence will most likely end as "mixed" and the modest framing.

**Diagnosis of g (revision 9; `5_diag_g/`, script `gonogo/phase05/diag_g.py`; sequential, no ψ̂² on g).**

*Step 1a, probe identity (from the run logs; no new computation).*
- **The probe that flagged g** is the online linear probe of `scripts/pretrain.py` (`eval/auroc__MultilabelAUROC_epoch`: 76 codes, LARS head, train folds ≤ 17, scored on fold 18).
  - For g it peaked at 0.557 at epoch 2 and ended at 0.542 (epoch 404); it was 0.536 at epoch 134 and 0.524 at epoch 269.
  - For h the same callback reached 0.633 at epoch 370 and ended at 0.632, rising from 0.597 at epoch 134. This reproduces h's reported online-probe value (0.633, appendix table of the no-action ablation).
- **h's "about 0.74"** is its full fine-tuning test AUROC (0.7391; `finetune.py` with `freeze_encoder=false`, then `evaluate.py`). That is a different protocol, so the comparison "g < 0.55 against h ≈ 0.74" mixed protocols.
- **Not like for like even on the same callback.** With `++objective=single`, the online probe was fed embeddings of augmented views, in training and in validation, whereas h's probe saw clean recordings. So g's 0.54 against h's 0.63 is not an equal comparison either. This is a pipeline difference in the diagnostic, not in g's training: the probe head is detached from the encoder.
- **Fix:** one offline frozen linear probe on clean embeddings, run identically for every encoder (`diag_g.py probe`). The results follow.
- **Export of g.** g's last checkpoint (epoch 404) was exported through `embed_h.py` (`h_single_last.npz`, 221,413 ECGs).
  - Its built-in check gives a relative distance between the two ECGs of a pair of 0.70 (stable pairs) and 0.75 (changed pairs), against 0.18 and 0.25 for h.
  - The values are equal across training, validation and test folds, so g does not memorize its training pairs.
  - g places a patient's recordings about four times farther apart than h does, as expected from an objective without same-patient positives.

*Step 1, common clean probe* (test fold 19 macro AUROC over 68 codes; patient bootstrap 95% CI; `5_diag_g/probe.json`). This is now the probe of the sanity rule: the same script, labels, folds and protocol for every encoder, so the bar is h − 0.05 = 0.529.

| Encoder | Test AUROC [95% CI] | Validation AUROC |
|---|---|---|
| Random-init ViT, g's architecture (the "learned nothing" floor) | 0.608 [0.584, 0.625] | 0.608 |
| h (val-loss checkpoint, epoch 339) | 0.579 [0.563, 0.598] | 0.586 |
| g, last checkpoint (epoch 404) | 0.516 [0.502, 0.537] | 0.534 |

- **1c.** All three g checkpoints load under `strict=True`, with no missing or unexpected keys. Their hyperparameters equal h's apart from the objective and the checkpoint names. The export path is the one h used: same data and ECG rows, 10 s at 500 Hz with no crop, the same lead order, and no normalization in either the training or the export path. Both encoders' weights moved far from initialization (median relative change: h 1.31, g 1.08).
- **1d.** The two views of every recording in a training batch differ as intended: shifts within ±0.5 s, amplitude factors within 0.95–1.05, noise at 5% of the lead SD, and different 0.5 s masks. The logged parameters reproduce `_single_views` exactly.
- **Step 1 finds no pipeline bug in g's training or export.** Fixing the probe protocol does not change g's outcome.

*Step 2, checkpoints* (same probe). Intermediate checkpoints were not saved; the losses at epochs 134 and 269 come from the training log.

| Epoch | Checkpoint | Train loss | Val loss | Online probe (augmented inputs) | Clean probe |
|---|---|---|---|---|---|
| 2 | online-probe best | 9.35 | 8.74 | 0.557 | 0.558 [0.538, 0.577] |
| 134 | none | 1.41 | 1.42 | 0.535 | — |
| 269 | none | 1.25 | 1.24 | 0.524 | — |
| 374 | val-loss selected | 1.23 | 1.20 | 0.544 | 0.517 [0.503, 0.538] |
| 404 | last | 1.23 | 1.22 | 0.542 | 0.516 [0.502, 0.537] |

- The last checkpoint is not degenerate: it scores the same as the val-loss checkpoint, and it is not collapsed (Step 3d). The rule's last checkpoint stands.
- The diagnostic content falls from epoch 2 on, and the same-patient pair distance rises from 0.36 (epoch 2) to 0.70 (epoch 404).

*Step 3, shortcut and collapse* (1,000 held-out fold-19 recordings or patients; `5_diag_g/shortcut.json`, `losses.json`).

| | g (last) | h |
|---|---|---|
| Recording-ID retrieval, top-1 (two augmented views; chance 0.001) | 0.995 | 0.998 |
| Same-patient retrieval, top-1 (two recordings of different stays) | 0.083 | 0.725 |
| Effective rank (of 256) | 20.5 | 43.4 |
| Per-dimension std, min / median / max | 0.62 / 1.04 / 1.40 | 0.79 / 1.01 / 1.26 |
| Prediction loss at the end (val) | 0.027 | 0.415 |

- **Loss trajectory.** g's view-prediction loss peaks at 0.11 (epoch 10), once SIGReg has spread the initially collapsed embedding, and then falls steadily to 0.027. With a per-dimension variance of about 1, that is a near-perfect prediction (about R² 0.97). The loss does not reach its floor early; the task becomes ever easier to solve.

**Diagnosis: a waveform-fingerprint shortcut.**
- The view-prediction task is solved by recording-specific features that survive the mild views. These features carry little patient or diagnostic information.
- g identifies the recording but not the patient across recordings (0.083 against h's 0.725), and its diagnostic AUROC is near chance and below even random features.
- It is not collapse: no dimension is near zero, and the effective rank is 20.
- Recording-ID retrieval is near perfect for h too. So the signature is low same-patient retrieval with low diagnostic AUROC, together with the training dynamics, not recording-ID retrieval alone.

**Sanity outcome.** g fails the rule as trained (0.516 against a bar of 0.529) and is recorded as **"training failed"**, not as having lost the comparison. No pipeline bug was found, so the single permitted retrain applies.

**Retrain objective, fixed before launch** (`++objective=intra`, `gorgonoid/dynamics_intra.sbatch`; submitted by the user).
- **Unchanged from h:** the predictive objective, the predictor (the same Dyn MLP with Proj(0), no action conditioning), the loss (1 − λ)·MSE + λ·SIGReg on both views, λ = 0.05, the architecture, and every hyperparameter.
- **Changed: the target.** Instead of the patient's next recording, the model predicts the embedding of a disjoint segment of the same recording from the rest of it.
  - **Revised before launch** (9 Nov, second pass): I-JEPA style with dropped patch tokens rather than zero-filled samples (`_intra_patches`, `ViT1d.forward_kept`).
  - The target is one random contiguous window of 80 patches (4 s). Its embedding is the pooled embedding of those patch tokens alone.
  - The context encoder sees only the tokens outside the window and outside 4 patches (0.2 s) on each side, each at its own RoPE position: 112 of the 200 tokens. The window starts where both gaps fit.
  - Context and target share no patch and no boundary. There is no amplitude, noise or shift augmentation.
  - The full-token input used at export is a superset of the training inputs, not an unseen zero-filled input. `forward_kept` with every token equals the full forward pass exactly.
  - *(The first version zero-filled samples on full-length inputs. It was superseded before any launch.)*
- **The contrast with h.** The retrain replaces h's cross-visit target with a disjoint segment of the same recording. That is not the only difference: each encoder pass also sees part of a recording rather than a whole one.
- **Online probe.** For both single-recording objectives it now receives the clean recording's embedding (no gradient), as h's did. The retrain's online probe is therefore comparable, but the sanity rule still uses the common clean probe above on the last checkpoint.
- **If the retrain also fails, b is the chosen representation** and the encoder ablation is reported as a limitation.

**Post-training report of intra** (10 Nov; `6_intra_report/`, `diag_g.py report`; last checkpoint, not checkpoint selection).
- **Run:** SLURM 84972 on gorgona5, 6 h 23 min. Run dir `runs/runs/20261109/231920/498bd4dbf3a1`; last checkpoint at epoch 404, global step 118,260.

| Check | intra (last) | h (frozen reference) | g (training failed) |
|---|---|---|---|
| Offline clean probe, test fold 19 | 0.528 [0.514, 0.547] | 0.579 | 0.516 |
| Recording-ID retrieval top-1 (chance 0.001) | 1.000 | 0.998 | 0.995 |
| Same-patient retrieval top-1 | 0.159 | 0.725 | 0.083 |
| Effective rank (of 256) | 35.8 | 43.4 | 20.5 |
| Pair distance of a patient's two recordings (stable / changed) | 0.62 / 0.66 | 0.18 / 0.25 | 0.70 / 0.75 |
| Online clean probe, max (epoch) / last | 0.570 (202) / 0.555 | 0.633 / 0.632 | — |

- The retrieval items are the same 1,000 fold-19 items as the diagnosis of g; h reproduces those values exactly.
- The random-init floor on the offline probe is 0.608. The pair distance of the random network is 0.32 / 0.41.

**Sanity outcome: intra fails the frozen rule ("training failed").**
- Its offline probe point estimate, 0.5281, is below the bar of 0.5290 (h 0.5790 − 0.05) by 0.0009, well inside its own confidence interval.
- The decision uses the point estimate, as fixed before the run. By the rule declared on 9 Nov, **b is the chosen representation**, and the encoder ablation is reported as a limitation.
- intra improves on g on every check: probe, same-patient retrieval, effective rank, and an online probe that rises instead of falling. It still places a patient's recordings far apart, farther than a random network does, and its same-patient retrieval is a fifth of h's.
- 0.5a, 3d and 3e have not been run.

*Status of the encoder ablation (revision 9, 10 Nov; recorded before any new number was computed).*
- **The 0.5a choice is fixed on b.** g and intra both failed the sanity rule ("training failed").
- **ψ̂² for intra and [intra; b] is exploratory only.** It is computed for the encoder-ablation section, with the same protocol as the references (state [b, intra, m]).
- **It cannot change the 0.5a choice**, whatever it shows.
- **The checks that follow on b:** 0.5a reporting on folds 10–19, then 3d and 3e.

**0.5a on b (revision 9, 10 Nov; `10_0.5a_b/`; Ledoit–Wolf primary; `gonogo/phase05/psi2.py`).** b was chosen on folds 0–9 (3 of 5 positive controls with a lower bound above 0, median ψ̂² 0.71; revision-8 table below). The evidence is folds 10–19 (10–14 against 15–19, Σ from the stable pairs of folds 10–19). b's state is [b, m], the references' state is [b, h, m], and Z = z.

| Code | Clean onsets (10–14 / 15–19) | b (chosen) | h (reference) | [h; b] (reference) |
|---|---|---|---|---|
| I44 | 48 / 42 | −0.05 [−1.30, 1.23] | 0.16 [−0.60, 0.97] | 0.08 [−1.58, 1.61] |
| I45 | 48 / 52 | −0.39 [−1.09, 0.30] | 0.69 [−0.54, 2.72] | 0.40 [−1.00, 2.52] |
| I47 | 78 / 74 | 0.49 [−0.16, 1.18] | 0.10 [−0.31, 0.52] | 0.51 [−0.25, 1.35] |
| I48 | 204 / 210 | 0.90 [0.59, 1.24]* | 0.10 [−0.07, 0.25] | 0.97 [0.60, 1.36]* |
| I49 | 34 / 43 | 0.59 [−0.37, 1.86] | 0.59 [−0.26, 1.55] | 1.21 [−0.11, 2.82] |
| Lower bound > 0, clean | | 1 of 5 | 0 of 5 | 1 of 5 |
| Lower bound > 0, adjusted (power check) | | 5 of 5 | 3 of 5 | 4 of 5 |

- **b's 3 of 5 on the selection folds did not replicate on folds 10–19.** On the clean subset, 1 of 5 positive controls (I48) has a lower bound above 0; I44 fell from 1.70 to −0.05. On the adjusted subset, all five do, but that is the power check, not the evidence.
- **Pooled 20 folds, sensitivity check only** (halves 0–9 against 10–19; `10_0.5a_b/folds_all_halves/`). b has 3 of 5 on clean (I47, I48, I49), with median ψ̂² 0.51; h has 2 and [h; b] 2. On adjusted: 5, 4 and 5. These folds include the selection folds, so this is not evidence for the choice.
- **Reading (revision 9, 10 Nov).** The per-feature Δb readouts below confirm the expected direction of every positive control on folds 10–19, with lower bounds above 0 for I47, I48 and I49. The failure to replicate is therefore a power limitation of the omnibus ψ̂² for effects concentrated in a few features. The omnibus statistic spends its degrees of freedom on 39 dimensions while the signal sits in one or two of them.
- **The narrowed encoder requirement does not change the 0.5a choice** (revision 9, 10 Nov; `method_section.tex`, §wm-setting).
  - The requirement now excludes only objectives that treat a patient's recordings as positives for invariance (contrastive, such as CLOCS or PCLR). Predictive cross-visit objectives, whose predictor can carry the change, are allowed.
  - h would therefore have been eligible. It does not change the choice, because h did not beat b, and neither h (1 of 5) nor [h; b] (2 of 5) qualifies on the clean selection folds 0–9 under the primary metric. b remains the chosen representation.
- **Consequence for the go/no-go.** On real codes, ψ̂² cannot show that the method recovers effects at these onset counts. **Part 2 becomes the main evidence that the method recovers effects**, with a sparse secondary statistic pre-declared for it (§3.4, revision 9).
- **Validation readout on Δb (folds 10–19, state [b, m]).** Signature effects in SDs of the feature's stable-pair change, clean / adjusted:

| Code | Feature | Clean | Adjusted |
|---|---|---|---|
| I44 | QRS width | +0.21 [−0.14, +0.54] | +0.36 [+0.22, +0.49] |
| I45 | QRS width | +0.21 [+0.00, +0.46] | +0.08 [−0.02, +0.20] |
| I47 | HR | +0.61 [+0.36, +0.85] | +0.39 [+0.31, +0.47] |
| I48 | P absence (Δm) | +1.21 [+1.01, +1.41] | +1.05 [+0.98, +1.13] |
| I48 | HR | +0.62 [+0.48, +0.75] | +0.50 [+0.45, +0.55] |
| I49 | RR irregularity | +0.91 [+0.47, +1.43] | +0.51 [+0.35, +0.67] |

- Every positive control keeps its signature in the expected direction. On the clean subset, the lower bound is above 0 for I47, I48 and I49 and at 0 for I45.
- **Does h beat b?** No (`h_vs_b.json`; the differences in median ψ̂² are paired bootstrap CIs).
  - Clean subset, h − b: −0.53 [−0.94, +0.18] on folds 0–9 and −0.32 [−0.77, +0.51] on folds 10–19.
  - Adjusted subset, h − b: −0.20 [−0.44, −0.01] and −0.17 [−0.37, −0.09].
  - h is never above b. On the adjusted subset it is below b with a CI excluding 0, and it has fewer positive controls with a lower bound above 0 everywhere.
  - So there is **no finding against the encoder requirement**.
  - [h; b] − b is +0.08 [−0.20, +0.81] (0–9) and +0.03 [−0.24, +0.93] (10–19) on clean, and +0.34 [+0.08, +0.49] and +0.46 [+0.12, +0.53] on adjusted. h adds information to b only where power is high, mainly for the conduction codes I44 and I45.
- **Changed-to-stable distance ratio** (`distance_ratio.md`). This is the mean squared distance between a pair's two ECGs on pairs whose cardiac codes change, divided by that on stable pairs. It uses the 1,000 fold-19 items of the diagnosis (997 with b at both ECGs), every representation standardized per dimension, and bootstrap 95% CIs.

| Representation | All items (544 stable, 453 changed): means / medians | Between-episode items (210 stable, 453 changed): means / medians |
|---|---|---|
| h | 1.40 [1.27, 1.54] / 1.46 [1.28, 1.62] | 1.31 [1.15, 1.49] / 1.30 [1.09, 1.48] |
| intra | 1.07 [0.99, 1.14] / 1.03 [0.95, 1.13] | 0.95 [0.88, 1.03] / 0.89 [0.80, 1.02] |
| g | 1.09 [1.01, 1.16] / 1.10 [1.02, 1.21] | 0.96 [0.88, 1.03] / 0.96 [0.87, 1.06] |
| b | 1.22 [0.84, 1.77] / 1.80 [1.31, 2.43] | 1.53 [0.99, 2.71] / 1.74 [1.22, 2.68] |

  - b and h move more when the codes change. intra and g do not between episodes.
  - b's features are heavy-tailed, so its mean ratio is imprecise, and the median ratio is the robust reading.
  - Same-episode pairs are stable by construction and close in time, which inflates every "all items" ratio. The between-episode columns are the fair comparison.
- **Exploratory: intra and [intra; b]** (state [b, intra, m]; `7_intra_gain/`, `10_0.5a_b/intra_folds{0_9,10_19}/`). They cannot change the choice. Ledoit–Wolf, lower bounds above 0 out of 5 and median ψ̂²:

| Folds, subset | intra | [intra; b] | b (chosen) | h (reference) | [h; b] (reference) |
|---|---|---|---|---|---|
| 0–9, clean | 0 (0.07) | 2 (0.83) | 3 (0.71) | 1 (0.19) | 2 (0.79) |
| 0–9, adjusted | 3 (0.02) | 5 (0.48) | 5 (0.34) | 3 (0.14) | 5 (0.67) |
| 10–19, clean | 0 (0.07) | 1 (0.38) | 1 (0.49) | 0 (0.16) | 1 (0.51) |
| 10–19, adjusted | 1 (0.03) | 5 (0.30) | 5 (0.24) | 3 (0.08) | 4 (0.71) |

  - intra alone carries almost none of the positive controls' effects. Its largest clean ψ̂² is 0.21 (I49), and its adjusted lower bounds above 0 come with medians near 0.02–0.03.
  - [intra; b] is roughly b. intra therefore adds little or nothing to b, while h adds information to b on the adjusted subset.
  - The pseudo-inverse disagrees with the primary on qualification for [intra; b] (clean, folds 0–9: 3 of 5 against 2 of 5) and for intra (adjusted, folds 0–9: 2 against 3).
  - **Reading for the ablation.** Without cross-visit pairing, neither single-recording objective (g, intra) learned a representation that carries the positive controls' effects. The cross-visit objective (h) carries some, mainly the conduction codes, but less than the fixed beat features b. The requirement "no same-patient positives" is therefore not shown to cost anything: no alternative that meets it beats h. Nor is it shown to help, because b, which needs no encoder, does best. This is a limitation of the encoder ablation, not a finding for or against the requirement.

**3d on b: Task 1–2 reruns and 0.5b statistic 2** (10 Nov; `8_3d_b/tables.md`, `task1.json`, `task2.json`, `da.json`; scripts `gonogo/phase05/run_b.py`, `tasks_b.py`).
- **Setup.** Clean subset, state [b, m], target Δb. Frozen sets of 0.5d (A = C0 without the validation subset, R = frozen R; SHA c1feffa3…). 3 seeds and 1,000 CRT draws. E is the R-learner excess gain ×10⁻⁵ with a paired patient bootstrap. `*` marks CRT significance after BH within family and condition.
- **Codes.** The 12 Phase 0 codes that were significant after z (or in Task 2): I10, I25, I50, I85, I95, K59, K80, K22, J06, I82, K21 and G43. On b, the codes significant after z (BH in the GO family) are I50 and J06, both already included (`codes_3d.json`).

*Task 1 on b* (fraction of the post-z gain removed; `u` marks a denominator whose CI includes 0):

| Code | Group | E after z | Removed by R | Removed by A |
|---|---|---|---|---|
| I50 | ambiguous | 24.8 [9.5, 40.5]* | 0.44 [0.27, 0.89] | 0.06 [−0.05, 0.21] |
| J06 | C1 control | 7.3 [1.4, 13.4]* | 0.12 [−0.22, 0.47] | 0.17 [−0.03, 0.54] |
| I85 | ambiguous | 1.9 [0.4, 3.6] | −0.14 [−4.16, 2.00] | 0.20 [−0.24, 1.25] |
| K59 | C1 control | 1.4 [−1.9, 4.6] | −0.47 u | 0.15 u |
| K80 | C1 control | −0.5 [−2.8, 1.7] | u | u |
| I48 (reference) | positive | 149.1 [102.1, 194.2]* | 0.22 [0.15, 0.30] | −0.01 [−0.05, 0.04] |
| I47 (reference) | positive | 46.6 [21.3, 72.6]* | 0.16 [0.02, 0.32] | 0.04 [−0.04, 0.13] |

- The other codes (I10, I25, I95, I82, K21, K22, G43) have no post-z gain on b whose CI excludes 0.
- **A removes nothing after z.** **R removes 44% of I50's gain, but also 22% and 16% of the real I48 and I47 effects** (over-adjustment and co-condition confounding are again inseparable).
- **Residual signatures.** J06's post-z residual is the acute-context signature: HR up 0.16 SD, lateral T waves flatter, right-precordial QRS smaller. I50's is lateral T-wave inversion, a plausible real repolarization effect.

*Task 2 on b* (setting-only withdrawal = the pre-registered primary reading). Rows whose setting-only removal CI excludes 0:

| Code | Removal by setting(t+1) | A recovers (setting withheld) | Removal by all of z | A recovers (all of z) |
|---|---|---|---|---|
| I50 | 2.79 [0.31, 5.73] | 0.79 [−0.54, 3.89] | 4.27 [−0.00, 8.75] | 0.86 u |
| K22 | 2.96 [0.88, 5.18] | 1.05 [0.34, 1.89] | 4.08 [1.12, 7.04] | 0.94 [0.37, 1.68] |

- Secondary (all-of-z removal CI excluding 0): I95 5.18 [1.23, 9.03], recovered 1.01 [0.56, 1.52]; I82 1.94 [0.05, 3.82], recovered 1.48 [0.12, 4.95].
- **On the primary reading, A recovers a substantial share of the withheld setting** (0.79–1.05; CI excludes 0 for 1 of 2 rows). This is the same conclusion as in Phase 0 (0.42–1.30; 2 of 5).

*0.5b statistic 2 on b.* D_A = E[Δt] − E[Δt + A], ×10⁻⁵. The setting-driven codes are the Task 2 rows above. The null-mean difference is 0.02–0.04 and negligible.

| Code | All pairs | Both admissions | Both ED-only | Beyond provenance |
|---|---|---|---|---|
| I50 | 3.68 [0.34, 7.05] (758 onsets) | 1.53 [−3.84, 6.99] (676) | not estimable (12 onsets) | no |
| K22 | 3.82 [0.86, 6.82] (405) | 3.32 [−0.90, 7.70] (345) | not estimable (3) | no |

- **0.5b outcome (rule unchanged): mixed, so the modest framing applies.**
  - "Provenance only" is excluded because A's AUROC for the setting is 0.817 < 0.9.
  - "Beyond provenance" holds for 0 of 2 codes: within pairs whose stays are both admissions, D_A keeps 42% (I50) and 87% (K22) of its all-pairs value, but neither CI excludes 0.
  - The power pre-check (4a) predicted this: power at full persistence was 0.60–0.79 per code.
  - The method claim narrows to "the label vector recovers label provenance when encounter metadata are missing". A contribution beyond provenance is not established.

*Phase 0 decision labels, revised on b (the chosen representation; pre-specified reruns, same patients).*
- **(a) holds, unchanged.** A recovers a substantial share of the withheld setting on the primary reading. The 11 curated anchors were not rerun (superseded).
- **(b) holds only in part, unchanged.** R removes 0.44 of I50's post-z gain, while removing 0.16–0.22 of real effects. I85's R-removal from Phase 0 (on Δh) does not appear on b.
- **(c) does not hold, unchanged.** A removes nothing after z.
- **(d) changes code.** On b, the unexplained post-z residual is **J06** (upper respiratory infection, a C1 control): significant after z, z+R and z+A, and removed by neither set. **K59's robust residual was specific to h**: on b, K59 shows no post-z gain, as it showed none on Δb in Phase 0. K80 shows none either. J06's residual already existed on Δb in Phase 0 (Table 1), but the Phase 0 labels were read on Δh.
- **Reading of (d).** A C1 control with a residual after z and every label set falls in row 4 of the interpretation matrix: confounding by context that nothing recorded captures, or a real effect. Its signature (faster HR, flatter T waves, smaller QRS) matches the acute-context signature of §1.1. A route through fever is possible, which the exclusion rule did not list for J06. In both readings it is the case the 0.5c margin is meant to bound.

**3e on b: final result** (10 Nov 13:15; `9_3e_b/gamma.json`, `tables.md`; 182 of 182 withheld codes, none dropped, no run failed).

| Margin (Ledoit–Wolf, DerSimonian–Laird) | N | μ̂ [SE] | τ̂ | Γ̂² |
|---|---|---|---|---|
| **Pooled** | 182 | 0.008 [0.003] | 0 | **0.014** (μ̂ + 1.645 SE, since τ̂ = 0) |
| Profile "both" | 20 | 0.007 [0.003] | 0 | 0.013 |
| Profile "admission" | 162 | 0.016 [0.008] | 0 | 0.029 |
| Groups 0–4 (check) | 37 / 36 / 36 / 36 / 37 | 0.016 / 0.015 / 0.004 / 0.007 / 0.008 | 0 | 0.031 / 0.030 / 0.014 / 0.019 / 0.017 |

- **Withheld codes.** The point estimates spread widely (5th–95th percentile −0.30 to 0.32, maximum 1.22 for M11 with 12 and 19 onsets). That spread is estimation noise: τ̂ = 0, so the margin is the upper bound of μ̂.
- **Group margins** agree within their uncertainty, so the rotation does not falsify the anchor rule.
- **Positive controls (folds 10–19, z + A).**
  - Only **I48 clears**: ψ̂² 0.93 [0.61, 1.27], which also clears its profile margin.
  - I47 0.46 [−0.19, 1.19], I49 0.56 [−0.41, 1.88], I44 −0.02 [−1.30, 1.21] and I45 −0.39 [−1.13, 0.31] do not.
- **C1 controls (all folds): none clears.** The largest lower bound is J06's, −0.012 (ψ̂² 0.060 [−0.012, 0.131]).
- **Ambiguous codes: none clears.** The largest is I85, 0.092 [−0.072, 0.257].
- **0.5c outcome: too little power.** Exactly one positive control clears and no C1 control does, which the user's reading fixed before this result treats as too little power.
  - Real-code grounding results are therefore reported as inconclusive, and the paper rests on the benchmark (3.6; Part 2 is the main evidence).
  - The margin itself is small and stable. The lack of power comes from the positive controls' quarter-fold estimates, not from a large residual.

**Margin sensitivity** (revision 10, step 2a; reported only, no decision depends on it; `9_3e_b/margin_sensitivity.{md,json}`, script `gonogo/phase05/margin_sensitivity.py`).
- **τ estimators.** On the 182 withheld codes, DerSimonian–Laird, Paule–Mandel and REML all give τ̂² = 0, μ̂ 0.0083 [SE 0.0032] and Γ̂² = 0.0135.
- **The 0.5c outcome does not depend on the margin.** Against a zero margin, every positive control except I48 still fails (lower bounds on folds 10–19: I44 −1.30, I45 −1.13, I47 −0.19, I49 −0.41; I48 +0.61).
- **SE calibration check, on permuted labels.**
  - Each withheld code's residualized labels were permuted within folds, which makes every true residual 0, and ψ̂² and its bootstrap variance were recomputed (3 replicates).
  - Cochran's Q/df is 0.29, 0.31 and 0.40 instead of about 1. The null z-scores have SD 0.54–0.63 instead of 1. On the real withheld codes, Q/df is 0.39.
  - **So the patient bootstrap overstates the variance of the split-half ψ̂² by about 3× near zero.** This is what theory predicts. A bootstrap draw perturbs both halves, θ̂*_A = θ̂_A + η_A and θ̂*_B = θ̂_B + η_B. Under the null, the variance of θ̂*_A′Mθ̂*_B then has three terms of equal expected size (θ̂_A′M S_B M θ̂_A, θ̂_B′M S_A M θ̂_B, and tr(M S_A M S_B)), while the true variance is the last term alone.
  - **Consequence:** every lower-bound decision of Phase 0.5 (0.5a qualification, 0.5c clearing) has been conservative near zero, with CIs about √3 too wide.
- **With permutation-calibrated variances** (variances × 0.333; normal-approximation CIs; *sensitivity only, the pre-specified decision is unchanged*):
  - On the real withheld codes, Q/df becomes 1.16, and τ̂² becomes 0.00011 (DL), 0.00066 (PM) or 0.00001 (REML). Γ̂² is then 0.029, 0.058 or 0.013.
  - The positive controls clearing would be **I47 and I48** under all three. Calibrated lower bounds: I47 0.070, I48 0.731; I44, I45 and I49 still fail.
  - **J06** (C1 control, calibrated lower bound 0.018) would clear the REML margin but not the DL or PM margins.
  - So the calibrated outcome would be **"usable" under DL and PM, and "margin too small" under REML.** The pre-specified outcome is "too little power". The 0.5c outcome is therefore fragile to the variance estimate, and the bootstrap's conservativeness is what made it "too little power".
**Real-data failure table on b** (revision 10, step 2b; reported only; `11_failure_table/table.{md,json}`, script `gonogo/phase05/failure_table.py`).
- **Test.** The same ψ̂² test (Ledoit–Wolf, patient bootstrap) on folds 10–19 (quarters 10–14 against 15–19), clean subset, state [b, m], Δb.
- **Count.** The number of codes whose lower bound clears 0. In the last row it is also counted against Γ̂² = 0.0135.

| Condition | C1 controls (false grounding) | Ambiguous | Positive controls (detection) |
|---|---|---|---|
| naive (no z, no anchors) | 1 / 11 (K21) | 1 / 9 (I50) | 1 / 5 (I48) |
| z | 0 / 11 | 0 / 9 | 1 / 5 (I48) |
| z − setting(t+1) | 1 / 11 (K21) | 0 / 9 | 1 / 5 (I48) |
| z − setting(t+1) + A | 1 / 11 (K21); against Γ̂²: 0 | 0 / 9; against Γ̂²: 0 | 1 / 5 (I48); against Γ̂²: 1 (I48) |

- **The pattern is the one the method addresses, but on a single code.**
  - The naive analysis falsely grounds one no-route code, K21 (GERD, setting-driven in Phase 0).
  - z removes the false grounding, and withholding the setting brings it back.
  - **The anchors did not remove K21's false grounding; only the margin did.** Adding A to z − setting(t+1) leaves K21's lower bound above 0, and only the comparison against Γ̂² removes it. I48 survives every condition.
- **Two limits.** With 1 of 11 C1 controls, this is an illustration, not evidence. And the conservative bootstrap (margin sensitivity) lowers every count.

- **Open for the user, before the semi-synthetic test is generated:** whether Part 2's ψ̂² inference keeps the percentile bootstrap (as pre-specified) or uses a permutation-calibrated variance. The power check below uses the pre-specified bootstrap; its rule (c) uses point estimates only and is not affected.

*The text below was written while the rotation was running and is kept for the record.*

**3e on b: Γ̂ by rotation** (`9_3e_b/`, scripts `run_b.py rotation` and `gamma_b.py`). *In progress; no 0.5c decision yet.*
- **Withheld codes.** Each of the 182 rotation codes is run under z + the other four groups (`z+rot<g>`), on Δb, clean, state [b, m], with residuals. ψ̂²_n uses folds 0–9 against 10–19 and Ledoit–Wolf Σ from all folds; v_n is its bootstrap variance.
- **Margin.** DerSimonian–Laird pooled, plus margins by profile ("both" and "admission") and per group as a check.
- **Tested codes,** under z + A: positive controls on folds 10–19 (quarters), ambiguous codes and C1 controls on all folds.
- **Status.** The rotation stopped when allocation SLURM 85229 ended, with roughly 105 of 182 codes done. It resumes with `python -I gonogo/phase05/run_b.py rotation`, which skips finished codes, followed by `python -I gonogo/phase05/gamma_b.py`.
- **A gap in the rule as written.** The 0.5c rule covers "usable" (≥ 2 positive controls clear and no C1 control does), "too little power" (none clears) and "margin too small" (a C1 control clears). If exactly one positive control clears and no C1 control does, the outcome is not covered. A partial preview suggests this is the likely case (only I48 clears). The rule is not changed here because results now exist; how to read that case is for the user to decide.
  - **Reading fixed by the user (10 Nov 12:50).** This was decided after the 90-code preview of 9:02 and before the final result. "Exactly one positive control clears the margin and no C1 control does" is read as **"too little power"**. It fails the "usable" threshold (≥ 2) without meeting "margin too small" (no C1 control clears). The rule's three outcomes are otherwise unchanged.

**0.5a references on folds 0–9, revision 8 (9 Nov; `4b_rev8/`, script `gonogo/phase05/psi2.py --preset references-rev8`).**
- **Inputs.**
  - h and [h; b] use the 3 Nov `gain.py` residual runs (`4b_references/gain_{clean,adjusted}/`, state [b, h, m]).
  - b uses new runs with state [b, m], without h (`4b_rev8/gain_bm_{clean,adjusted}/`, targets Δb and Δm). They needed `gain.py --state bm`.
  - Z = z throughout.
- **Estimate.** θ from folds 0–4 against folds 5–9, patient bootstrap 95% CI. Σ is estimated on the 17,264 stable pairs of folds 0–9. Folds 10–19 were not read, and the choosing rule is not applied.
- **Metric diagnostics:**

| Representation | Role | Dims | Ledoit–Wolf intensity | Dims holding 99.9% | Condition number, sample (Ledoit–Wolf, standardized) |
|---|---|---|---|---|---|
| h | reference | 256 | 0.004 | 42 | 6.1×10⁷ (2,540) |
| [h; b] | reference | 295 | 0.005 | 64 | 1.1×10¹⁰ (2,400) |
| b | eligible | 39 | 0.003 | 23 | 1.4×10⁵ (133) |

Clean subset, primary metric (Ledoit–Wolf), ψ̂² [95% CI]; `*` marks a lower bound above 0:

| Code | Onsets (0–4 / 5–9) | h (reference) | [h; b] (reference) | b (eligible) |
|---|---|---|---|---|
| I44 | 42 / 53 | 2.45 [0.51, 5.29]* | 3.58 [0.83, 7.47]* | 1.70 [0.03, 3.70]* |
| I45 | 43 / 61 | 0.75 [−0.33, 2.22] | 0.79 [−0.37, 2.53] | 0.15 [−0.31, 0.64] |
| I47 | 69 / 81 | 0.11 [−0.26, 0.48] | 0.71 [−0.05, 1.56] | 0.71 [0.08, 1.45]* |
| I48 | 175 / 188 | 0.19 [−0.01, 0.40] | 0.84 [0.45, 1.27]* | 0.85 [0.52, 1.22]* |
| I49 | 20 / 32 | −0.03 [−1.08, 1.06] | 0.77 [−0.78, 2.68] | 0.58 [−0.48, 1.98] |
| Median ψ̂² | | 0.19 | 0.79 | 0.71 |

Lower bounds above 0, out of 5 positive controls, by metric (primary in bold):

| Subset | Metric | h | [h; b] | b |
|---|---|---|---|---|
| clean | **Ledoit–Wolf** | **1** | **2** | **3** |
| clean | 99.9% pseudo-inverse | 2 | 3 | 3 |
| clean | full inverse | 0 | 0 | 3 |
| adjusted (power check) | **Ledoit–Wolf** | **3** | **5** | **5** |
| adjusted | 99.9% pseudo-inverse | 3 | 5 | 4 |
| adjusted | full inverse | 0 | 0 | 5 |

Per-feature readout on Δb, the validation readout. It is the standardized effect in SDs of the feature's stable-pair change, on folds 0–9 pooled, with state [b, m]. P absence exists only in `m`, so I48's comes from Δm.

| Code | Feature | Clean | Adjusted |
|---|---|---|---|
| I44 | QRS width (Δb) | +0.53 [+0.14, +0.95] | +0.47 [+0.34, +0.61] |
| I45 | QRS width (Δb) | +0.19 [−0.07, +0.44] | +0.12 [+0.01, +0.23] |
| I47 | HR (Δb) | +0.52 [+0.30, +0.73] | +0.32 [+0.24, +0.39] |
| I48 | P absence (Δm) | +1.21 [+1.01, +1.43] | +1.11 [+1.04, +1.19] |
| I48 | HR (Δb) | +0.62 [+0.47, +0.77] | +0.47 [+0.42, +0.52] |
| I49 | RR irregularity (Δb) | +0.83 [+0.35, +1.37] | +0.54 [+0.38, +0.70] |

- **Qualification under the primary metric** (≥ 3 of 5 lower bounds above 0, clean): b qualifies. The references h and [h; b] do not.
- **Disagreement with the pseudo-inverse:** only for the reference [h; b], which qualifies under it (3 of 5). The primary decides.
- **Compared with revision 7** (pseudo-inverse, b with state [b, h, m], Σ over all folds: h 2, [h; b] 3, b 3): the eligible b's qualification is unchanged. Only the references move.
- **b no longer depends on g.** b's folds-0–9 evaluation is complete, with b qualifying, median ψ̂² 0.71. So the "none qualifies" branch of 0.5a (rule 3.6) can no longer apply. The 0.5a choice will be among the qualifying ones of [g; b], g and b, by median ψ̂², ties broken in that order.
- **All signatures point the expected way.** I45's QRS widening has a lower bound above 0 only on the adjusted subset.

*The revision-7 table below used the 99.9% pseudo-inverse, Σ over all folds, and b with h in its state. It is superseded by the tables above and kept for the record.*

**0.5a references on folds 0–9 (revision 7; `4b_references/`, script `gonogo/phase05/psi2_references.py`).**
- **Inputs.** Fresh `gain.py` residual runs from 3 Nov 17:00 on gorgona10. Z = z, state [b, h, m]; positive controls; Δh and Δb.
- **Estimate.** ψ̂² with θ from folds 0–4 against folds 5–9, using the principal-subspace metric (99.9% of stable-pair variance: h 42 of 256 directions, b 23 of 39, [h; b] 64 of 295). Patient bootstrap 95% CI. Folds 10–19 were not read, and the choosing rule is not applied until g exists.

| Code | Clean onsets (0–4 / 5–9) | h (reference) | [h; b] (reference) | b (eligible) |
|---|---|---|---|---|
| I44 | 42 / 53 | 2.23 [0.38, 4.99]* | 3.14 [0.82, 6.66]* | 1.17 [0.27, 2.31]* |
| I45 | 43 / 61 | 0.78 [−0.23, 2.25] | 0.89 [−0.18, 2.47] | 0.10 [−0.27, 0.51] |
| I47 | 69 / 81 | 0.10 [−0.21, 0.39] | 0.58 [0.03, 1.23]* | 0.60 [0.13, 1.21]* |
| I48 | 175 / 188 | 0.19 [0.02, 0.37]* | 0.51 [0.26, 0.81]* | 0.42 [0.23, 0.66]* |
| I49 | 20 / 32 | 0.22 [−0.62, 1.06] | 0.31 [−0.75, 1.48] | 0.08 [−0.59, 0.68] |
| Lower bound > 0 (clean) | | 2 of 5 | 3 of 5 | 3 of 5 |
| Median ψ̂² (clean) | | 0.22 | 0.58 | 0.42 |
| Lower bound > 0 (adjusted, power check; 235–1,625 onsets per quarter) | | 3 of 5 | 5 of 5 | 4 of 5 |
| Median ψ̂² (adjusted) | | 0.11 | 0.37 | 0.16 |

- **The metric matters.** With the full inverse as written, h and [h; b] are noise-dominated (|ψ̂²| up to about 400, CIs about ±1,000). b changes too: 2 of 5 on clean, median 0.80; 5 of 5 on adjusted, median 0.30. Full tables are in `4b_references/table.md`.
- **Readings for later (no decision now).**
  - b alone already reaches three positive controls on clean folds 0–9 under the proposed metric.
  - h adds to b mainly for the conduction codes I44 and I45.
  - The reference h by itself is the weakest of the three.

---

## 3. Part 2: Do existing methods fail on confounded, ungrounded codes?

**Question.** With known ground truth, do existing methods falsely declare ungrounded-but-confounded codes as grounded? Two complementary environments answer it.

| Environment | Role | Confounder | Strength control |
|---|---|---|---|
| Latent synthetic (3.1) | controlled sweep; shows the failure mechanism and where it begins | simulated | full, from γ = 0 to strong confounding |
| Semi-synthetic MIMIC-IV-ECG (3.2) | **main test**; real noise, patient dominance and confounding | real (strongest driver from 2.9); revision 5: the care setting at t+1 | mixing knob `γ_mix`, from 0 to the real strength |

**Why no signal-level generator.** All methods operate on the frozen representation `h`, so injecting effects in `h` tests exactly what the methods do. Whether the encoder captures real ECG effects is already checked by the positive controls of Part 1. A signal-level generator (e.g. ECGSYN) would cost several days, invite realism objections, and require inventing the confounder. It is kept only as an optional follow-up (Section 5). *(Revision 5: in Phase 0 only I48's effect vector in `h` has a CI excluding 0, and the encoder may be replaced in Phase 0.5a. The positive-control check is therefore rerun on the chosen representation before injection.)*

**Revision 5, the confounding driver.** The driver is the care setting at t+1, which drives the confounding of setting-driven codes (Phase 0 results, Table 2). Age drives other codes, but it is observed in every deployment and partly encoded in `h_t` (patient share 0.74), so it would not act as a hidden driver.

### Revision 10 (10 Nov 2026): Part 2 after Phase 0.5, recorded before any Part 2 data are generated

**(a) Representation.**
- **b is primary, and C2 is read on b.**
- h is a secondary environment. It is now eligible under the narrowed encoder requirement (§7 item 14).
- Part 2 has known ground truth, so running both involves no selection.

**(b) Injected effect vectors.**
- **On b:** the R-learner effect vectors of I47, I48 and I49 under z, estimated on all 20 folds (`8_3d_b/gain/clean_bm_b_z_<code>.resid.npz`). They are simulation inputs, not evidence.
- **Pre-declared scenarios kept alongside them (§3.4, revision 9):**
  - one-feature: QRS width, at I44's all-fold per-feature effect;
  - two-feature: HR and RR irregularity, at I48's all-fold per-feature effects;
  - diffuse: a random direction with the median squared Mahalanobis norm of the three real vectors.
- **On h:** I48's vector (`effects_h.npz`).

**(c) Effect scale for C2.**
- The sweep is 0.5×, 1×, 2× and **4×**.
- **C2 is read at the smallest scale at which the oracle reaches a G-vs-U AUROC ≥ 0.9 at γ_mix = 0.** This uses the oracle only, not the compared methods. If no scale reaches it, C2 cannot be read: that is recorded and Part 2 stops.
- **Operationalization** (execution order (g) puts the scale choice before the semi-synthetic test is generated):
  - The rule is applied in the b-calibrated latent environment at γ = 0, the latent analogue of γ_mix = 0: no link between labels and the hidden context.
  - The oracle for this step is the Part 1 estimator given the hidden context u in its base model and its propensity. That is the cross-fitted R-learner with the debiased ψ̂² of 0.5a and Ledoit–Wolf Σ. The baselines do not exist yet and are not used.
  - The score is ψ̂²; the AUROC separates the G codes of a draw from its U codes and is averaged over the code draws.
  - **Folds.** ψ̂² uses halves (folds 0–9 against 10–19) on all patients, the protocol of the all-fold tested codes of 0.5c. The quarter-fold reporting protocol (10–14 against 15–19) is reported next to it.
  - **Nuisance models.** In the latent environment, the Part 1 estimator's base model is cross-fitted ridge regression and its propensity cross-fitted logistic regression, instead of gain.py's MLP. The generator's conditional mean of the change is linear in the current state, so the linear base model is correctly specified. The R-learner, its θ and the ψ̂² statistic are otherwise those of gain.py and psi2.py.
  - **Detection** of a G code: ψ̂² lower bound above 0, and above the real margin Γ̂² = 0.014 (3e), both reported. The sparse statistic of §3.4 is not part of the power check.
  - **Draws.** 8 independent draws per scale. Each draw has 6 G codes (the I47, I48, I49, one-feature, two-feature and diffuse vectors) and 6 U codes with matched onset counts.
  - **Correction after the first power-check run (10 Nov 13:46), before any result was used.**
    - The first run followed §3.1 literally: the onset risk was logistic in a projection of the unobserved patient vector p. That makes U codes non-null for any estimator conditioning on x_t. Given x_t, a high-risk patient has a lower state deviation s = x − p, so a label that reveals p predicts the regression to the patient's own mean.
    - U codes were then "detected" in 98–100% of draws. That fails the environment's validation checkpoint (timeline: "finds no gain for U codes"), so **the first run is invalid** and its outputs are kept as `power_b_invalid_risk_on_p.*`.
    - The risk score now depends on the observed state x_t, as §3.2's "risk profile" does. Onset pairs are drawn from candidate pairs with weights exp(α·w′x_t), where α is set for a propensity AUROC near the real ~0.7. Given x_t, the label is then independent of the change, and U codes are null.
    - The patient-identity route is a real confounding mechanism (Phase 0's age/sex drivers are a part of it, and z conditions on age/sex). It is noted for the semi-synthetic test, where onset risk already depends only on observed age and h_t.

**(d) Latent environment calibrated to b** (`gonogo/part2/latent.py`; measured on b's usable between-episode pairs).
- dim = dim(b) = 39, in units of b's cross-sectional SD.
- Patient share of variance: one-way random-effects ANOVA over patients with ≥ 2 ECGs, as in `noise_floor.py`.
- The share of the change explained by the current state.
- The stable-pair change covariance (the noise floor).
- The stable fraction.
- The setting(t+1) shift of the change: hidden context u_{t+1} = "the next stay is ED-only", with its real frequency and the real mean shift of Δb on stable pairs.
- Clean onset counts: G and U codes at the real counts of the positive controls and of the median passing cardiac code.

**(e) Weak labels for the injected G and GC codes.**
- A true event is recorded with probability κ < 1, which is lower when the stay at t+1 is ED-only. This mimics the real recording route: κ_admission = 0.9, and κ_ED-only = 0.9 × 0.13, using I48's odds ratio of onset into ED-only stays (2.4).
- An unrecorded event leaves its pair labelled stable, effect included.
- UC codes, whose labels depend on setting(t+1), already cover provenance confounding.

**(f) The method in the same harness.**
- It runs with the same interface as the baselines.
- **Baselines are scored at the oracle F1 threshold** of 3.4. The method is scored by its own "grounded" class, without an oracle threshold.
- This asymmetry favours the baselines and is reported with the results.

**(g) Execution order.**
1. The 3.6 power check in the b-calibrated latent environment.
2. The scale choice by (c).
3. The semi-synthetic test.
4. Baselines 1–3.
5. Baselines 4–6.
6. The method.

### Revision 11 (11 Nov 2026): user decisions, recorded before any semi-synthetic data are generated

**1. ψ̂² inference for Part 2.** The Phase 0.5 outcomes stay as pre-specified. The calibrated versions in the margin-sensitivity paragraph are post hoc sensitivity analyses.
- **Why the inference changes.** The split-half ψ̂² is a degenerate U-statistic at θ = 0. There, the patient bootstrap of T = θ̂_A′ M θ̂_B overstates its variance about 3× (margin sensitivity, revision 10). M = Σ̂⁻¹ is the Ledoit–Wolf metric.
- **Primary test against 0: the stratified permutation null.** The residualized labels (a − π̂) are permuted within strata and T is recomputed. On real data the strata are those of 2.5. In the latent environment there is no gap or age, so the strata are fold × quintile of π̂.
- **Primary intervals.**
  - V_A and V_B (the covariances of θ̂_A and θ̂_B) are each estimated by a patient bootstrap within their own half.
  - The variance is the bias-corrected plug-in Var(T) = [θ̂_B′ M V_A M θ̂_B − tr(M V_A M V_B)]₊ + [θ̂_A′ M V_B M θ̂_A − tr(M V_A M V_B)]₊ + tr(M V_A M V_B). That is the user's formula in the Σ⁻¹ metric.
  - The interval is T ± 1.96 √Var(T), and comparisons against Γ̂² use it.
- **Secondary:** the percentile bootstrap of T.
- **Validation before the semi-synthetic test**, in the b-calibrated latent environment:
  - the coverage of the new intervals at every scale, including θ = 0 (target ≈ 0.95);
  - Q/df ≈ 1 on permuted labels.
  - *If coverage at θ = 0 is below 0.93, stop and report.*
  - Design: so that the true θ is known exactly, G codes are recorded without weak labels (κ = 1). Given x_t, the labels are then independent of the change and of u, so θ = scale × Δ_c for both estimators. Truth: ψ² = (scale Δ_c)′ M (scale Δ_c); for U codes, 0.
  - 20 draws × (6 G + 12 U) codes, with the Part 1 estimator, whose statistic the intervals are for. That gives 240 null codes for the coverage at θ = 0 (SE ≈ 0.014) and 120 G codes per scale. 400 bootstrap draws per half and 200 permutations per code.

**Validation result** (11 Nov 05:30; `outputs/part2/latent/validate_rev11.{md,json}`, rows `validate_rev11_rows.csv`; `latent.py validate --draws 20`).
- **Design.** κ = 1, risk on the observed state, shrunk vectors, Part 1 estimator.

| θ | Codes | Coverage, new interval | Coverage, percentile bootstrap (secondary) | Permutation test rejects (p < 0.05) | Mean truth ψ² / mean T |
|---|---|---|---|---|---|
| **0 (U codes)** | 240 | **0.950** | 1.000 | 0.046 | 0 / −0.002 |
| 0.5× | 120 | 0.967 | 0.992 | 0.800 | 0.169 / 0.168 |
| 1× | 120 | 0.950 | 0.967 | 0.942 | 0.677 / 0.675 |
| 2× | 120 | 0.942 | 0.950 | 1.000 | 2.709 / 2.705 |
| 4× | 120 | 0.950 | 0.942 | 1.000 | 10.837 / 10.827 |

- **Q/df on permuted labels, new variance:** U codes 0.92 (z SD 0.96, N 240); G codes at 1× 0.77 (z SD 0.87); all codes 0.82.
- **By vector,** coverage is 0.90–1.00 at every scale, with 20 codes per cell.
- **Verdict: pass** (coverage at θ = 0 is 0.950 ≥ 0.93).
  - The new intervals are calibrated at the null and at every scale, and the permutation test has its nominal size.
  - The percentile bootstrap over-covers at the null (1.000), confirming the degeneracy it was replaced for.
  - On permuted labels of G codes, the new variance is somewhat conservative: the injected effect stays in the residuals.

**2. Scale choice and injected vectors.**
- **Rule (c) is applied in the semi-synthetic test, as written** (γ_mix is the semi-synthetic knob). The oracle is the Part 1 estimator plus the withheld driver, setting(t+1).
  - Generate the semi-synthetic G and U codes at γ_mix = 0 first, apply rule (c) there with the new inference, and record the chosen scale. **Only then** generate UC and GC codes and the γ_mix sweep.
  - The latent-environment result (0.5×) is recorded as **preliminary**.
- **Shrink every injected vector to its debiased norm, in both environments.**
  - Each vector is multiplied by √(ψ̂²_debiased / ψ²_plug-in).
  - For I47, I48 and I49, ψ̂²_debiased is the split product of the 20-fold halves (0–9 against 10–19).
  - For the one- and two-feature vectors, it is the same split product restricted to their coordinates.
  - The diffuse vector takes the median debiased norm of the three real vectors.
- **Latent noise stays Gaussian.** A t-noise variant is optional and report-only.
- **Realism check (report-only).** The detection rate at 1× in the semi-synthetic test, against the real folds 10–19 result (1 of 5 positive controls). It is computed with the same protocol as the real result (quarters 10–14 against 15–19, lower bound > 0 of the percentile bootstrap) and also with the new intervals.

**Semi-synthetic test on b, operational details** (fixed here, before generation):
- **Base data.** The 35,009 real usable stable between-episode pairs (the clean pool of b), with real b_t, b_{t+1}, z and labels.
- **Injected onsets.** Pairs are sampled with weights exp(α·score), where the score is the standardized mean of a random projection of b_t and age (the 3.2 risk profile) and α = 0.75 (propensity AUROC near 0.7). The onset counts are those of the latent environment.
- **G codes.** scale × Δ_c (shrunk) is added to b_{t+1} on event pairs. Events are recorded with κ = 0.9 at an admission and 0.9 × 0.13 at an ED-only next stay (revision 10 e); an unrecorded event keeps its effect and label 0.
- **U codes.** The same risk profile, no effect.
- **The Part 1 estimator** conditions on b_t and on z without setting(t+1): log gap, setting(t), age, sex and the coding systems. It also conditions on the transitions of the frozen set A (the real codes, as every method receives them, revision 5). The base model is cross-fitted ridge, the propensity cross-fitted logistic, on the real patient folds.
- **The oracle** adds setting(t+1).
- **Propensity features and permutation strata.** The propensity uses [b_t, z without setting(t+1)], plus setting(t+1) for the oracle. It does not use the A transitions, because the injected labels depend only on b_t, age and, for recording, setting(t+1). The permutation strata are fold × gap quintile × age tertile, × setting(t+1) for the oracle; they stay within folds, so the halves are preserved.
- **Scoring.** ψ̂² on halves 0–9 against 10–19. The score for rule (c) is the ψ̂² point estimate. Detection is the new interval's lower bound above 0, and above Γ̂². 8 draws × (6 G + 6 U).

**Semi-synthetic test on b at γ_mix = 0: result and scale choice** (11 Nov 05:42; `outputs/part2/semisynth/gamma0.{md,json}`, rows `gamma0_rows.csv`; `gonogo/part2/semisynth.py --draws 8`).
- **Design.** 35,009 real stable pairs; 8 draws × (6 G + 6 U); κ̄ = 0.663; propensity AUROC median 0.725. ψ̂² on halves, new inference.

| Estimator | Scale | G-vs-U AUROC, mean (min) over draws | G: new lower bound > 0 | > Γ̂² | permutation p < 0.05 |
|---|---|---|---|---|---|
| oracle | 0.5× | 0.854 (0.667) | 0.58 | 0.56 | 0.71 |
| oracle | **1×** | **0.972** (0.917) | 0.83 | 0.83 | 0.92 |
| oracle | 2× | 1.000 | 1.00 | 1.00 | 1.00 |
| oracle | 4× | 1.000 | 1.00 | 1.00 | 1.00 |
| Part 1 | 0.5× / 1× / 2× / 4× | 0.854 / 0.979 / 1.000 / 1.000 | 0.58 / 0.83 / 1 / 1 | 0.54 / 0.83 / 1 / 1 | 0.73 / 0.92 / 1 / 1 |
| U codes (θ = 0) | — | — | oracle 0.02, Part 1 0.02 | 0.00 | oracle 0.10, Part 1 0.12 |

- **Rule (c): the oracle's mean G-vs-U AUROC reaches 0.9 first at 1× (0.972; 0.854 at 0.5×). C2 is read at 1×.** This supersedes the preliminary latent result (0.5×).
- **By vector** (oracle, lower bound above Γ̂², at 1×): I47, I48, I49, two-feature and diffuse 1.00. The one-feature vector is 0.00, detected only from 2×.
- **The null on real pairs.** The new interval's lower bound clears 0 for 2% of U codes, near the nominal 2.5%. The permutation test rejects 10–12% of U codes (5–6 of 48 per estimator, SE ≈ 3%), above its nominal 5% and its latent-environment size (4.6%). This may be anti-conservative on real data and is flagged; no decision here uses it.
- **Realism check (report-only)** at 1×, with the real protocol (quarters 10–14 against 15–19, percentile lower bound > 0):
  - 71% of G codes are detected (oracle) and 73% (Part 1); 75–79% with the new interval. The real data detected 1 of 5 positive controls.
  - By vector: I48 1.00 (real I48 also clears); two-feature and diffuse 1.00; I47 0.75 (real fails); I49 0.50–0.62 (real fails); one-feature 0.00 (real I44 fails).
  - **The semi-synthetic test remains more optimistic than the real data.** Real onset pairs have 1.7–2.6 times the residual variance of stable pairs (summed over b's 39 standardized dimensions: I44 2.63, I45 1.74, I47 2.36, I48 1.92, I49 1.90).
  - That excess is far larger than the effects' own squared norms: real onsets come with acute, heterogeneous changes. The semi-synthetic test injects onsets into stable pairs, which have stable-pair noise plus a constant effect, so θ̂ is about twice as precise as on real data.
  - This is reported, not corrected. A heteroskedastic variant would add noise to injected event pairs to match the real onset-pair variance, and is a decision for the user before the UC/GC generation.
- **Next, per decision 2:** generate UC and GC codes and the γ_mix sweep at 1×, which has not been started.

**3. Risk on the unobserved patient vector p.** The correction (risk on the observed state) is kept for G and U codes. The p-driven version is retained as an exploratory stress scenario, **"patient-substrate confounding"**: reported, never used for C2. Its first results are the invalid run of revision 10 (`outputs/part2/latent/*invalid_risk_on_p*`).

**4. Failure table.** The anchors did not remove K21's false grounding; only the margin did (text corrected above).

### Revision 13 (11 Nov 2026): user decisions, recorded before running anything below

**1. Input-variant decomposition (pre-declared secondary; no decision depends on it).**
- **Purpose:** to separate "the anchors remove the confounding" from "there is little confounding to remove".
- **Data:** the rev-12 sweep draws at γ_mix = 1 and γ_mix = 0.5: 6 UC + 3 GC codes × 8 draws each, from the saved event and recorded-onset indices. The R-learner is the Part 1 estimator, with four input variants:
  - **naive:** base model and propensity on b_t only (no z, no A);
  - **z − s:** both on [b_t, z without setting(t+1)];
  - **z − s + A:** the current run (the base model adds the A transitions; propensity on [b_t, z without setting(t+1)]);
  - **oracle:** the current run plus setting(t+1) in both.
- **Reported per variant:** the UC false-grounding rate under three readings (CRT p < 0.05; new lower bound > 0; new lower bound > Γ̂²) and the GC effect bias (median √max(T, 0)/√ψ²_true).
- **Operational detail.** The sweep did not save its noise draws, only the indices. The noise is therefore redrawn (seed 1213) once per code and shared by all four variants. Its scale c is solved, as in rev 12, on the z − s + A residuals. The z − s + A and oracle variants thus repeat the sweep's numbers with fresh noise, which also serves as a check.

**2. Semi-synthetic test on h (the pre-declared secondary environment), same protocol as b.**
- **Representation.** h is the Phase 0.5 reference h, the `zeroactions` export, with 256 dimensions. The base pool is the same 35,009 real stable pairs.
- **Estimators.** As on b, with h_t in place of b_t: the representation's own state, z without setting(t+1), A in the base model, and setting(t+1) added for the oracle. M is Ledoit–Wolf on the standardized stable-pair changes of h. The risk profile is a random projection of standardized h_t, plus age.
- **Vector.** The I48 θ of the real run `clean_bhm_h_z_I48` (state [b, h, m], target Δh, z, clean; 4b references), shrunk to its debiased norm (split product, folds 0–9 vs 10–19).
  - The shrink uses h's Ledoit–Wolf metric, because h's full inverse is ill-conditioned (condition number 6e7).
  - **If the debiased norm is ≤ 0, stop and report.**
- **Codes per draw.** G and GC codes all carry the I48 vector, each with its own risk profile. Counts:
  - G and U: 6 codes each, with b's counts (302, 777, 129, 185, 204, 351), so the G-vs-U AUROC has the same design as on b;
  - GC: 3 codes, with counts 302, 777 and 129;
  - UC: 6 codes.
- **κ** as on b.
- **Noise targets, measured on h** with the rev-12 procedure. G codes: the real I48 onset-pair ratio on h. U, UC and GC: the median ratio of the 11 C1 controls on h.
  - These come from new gain.py runs of the C1 controls on h (state [b, h, m], target Δh, z, clean, residuals saved), because the Phase 0 h runs did not save residuals.
- **Γ̂² for h.** No rotation exists on h, so it is computed with the 3e protocol:
  - each of the 182 rotation codes under z + rot<g>, with state [b, h, m], target Δh, the clean subset and the frozen sets;
  - the pooled DerSimonian–Laird margin over the Ledoit–Wolf ψ̂² (folds 0–9 vs 10–19), with 1,000 patient-bootstrap variances, i.e. `gamma_b.py`'s estimator applied to h. Output in `outputs/phase05/12_h_rotation/`.
- **Rule (c) on h** is applied at γ_mix = 0 and recorded separately.
  - The γ_mix sweep and the input-variant decomposition (step 1) then run on h at h's chosen scale. If no scale reaches 0.9 on h, they run at 4×, and this is reported.
  - **C2 is still read on b; h is reported alongside.**

**3. No stronger semi-synthetic driver is added.** The latent γ sweep is the pre-declared place to show where failures begin.

**4. Tex.** In the grounding-test paragraph, "permutation null" is replaced by a conditional randomization test, with labels drawn from the cross-fitted propensity model. The degenerate-U-statistic sentence and the bias-corrected interval are kept.

**Stop after steps 1–2 and report, before the baselines.**

**Revision 13, result 1: input-variant decomposition on b** (`outputs/part2/semisynth/decompose.*`). The data are the rev-12 sweep's draws at 1×: 48 UC and 24 GC codes per γ_mix, with noise redrawn and shared by the variants.

| γ_mix | Variant | UC: CRT p < 0.05 [95% CI] | UC: lower bound > 0 | UC: lower bound > Γ̂² | UC: median T | UC: propensity AUROC | GC bias, median (10–90%) |
|---|---|---|---|---|---|---|---|
| 1 | naive (b_t) | 0.29 [0.17, 0.44] | 0.13 | 0.02 | 0.026 | 0.54 | 0.94 (0.88–1.06) |
| 1 | z − s | 0.10 [0.04, 0.23] | 0.04 | 0 | 0.008 | 0.66 | 0.95 (0.89–1.06) |
| 1 | z − s + A (Part 1) | 0.10 [0.04, 0.23] | 0.02 | 0 | 0.004 | 0.66 | 0.94 (0.90–1.06) |
| 1 | oracle | 0.10 [0.04, 0.23] | 0.02 | 0 | 0.000 | 0.75 | 0.94 (0.90–1.07) |
| 0.5 | naive (b_t) | 0.08 [0.02, 0.20] | 0.02 | 0 | 0.001 | 0.56 | 1.02 (0.89–1.12) |
| 0.5 | z − s | 0.06 [0.01, 0.17] | 0 | 0 | −0.007 | 0.60 | 0.99 (0.87–1.08) |
| 0.5 | z − s + A (Part 1) | 0.06 [0.01, 0.17] | 0 | 0 | −0.008 | 0.60 | 0.98 (0.87–1.08) |
| 0.5 | oracle | 0.04 [0.01, 0.14] | 0.02 | 0 | −0.009 | 0.66 | 0.98 (0.87–1.08) |

GC codes clear Γ̂² in 0.96 of cases under every variant.

**Reading (descriptive; no decision depends on it).**
- **At γ_mix = 1 there is confounding to remove.** Without z (naive), the CRT rejects 29% of UC codes and the median T is 0.026, above Γ̂².
- **z without setting(t+1) removes most of it:** CRT 10%, median T 0.008.
- **A removes a little more of the median T** (0.004); the oracle takes it to 0.
- **Even the naive variant reaches the margin in only 2% of UC codes.** The lower bound rarely exceeds Γ̂² because a single code's interval is wider than the confounded T.
- **At γ_mix = 0.5 the confounding is small under every variant.**
- **So on b, both statements hold in part.** There is confounding, and z (not A) is what removes it. What is left is too small to reach the margin, even before adjustment.
- The GC effect is not biased by the confounding under any variant (medians 0.94–1.02).
- Check: with fresh noise, the Part 1 and oracle rows reproduce the sweep within sampling error (CRT 0.10 vs 0.15 and 0.06).

**Revision 13, result 2a: calibration on h** (`outputs/part2/semisynth_h/calibration_h.*`).
- **I48 on h.** Plug-in ψ̂² 0.374, debiased 0.156 (halves), so the shrink factor is 0.65. The debiased norm is > 0, so the test proceeds.
- **Noise targets on h,** measured with the rev-12 procedure:
  - G codes: I48's onset-pair ratio, 1.20 (b: 1.92).
  - U, UC and GC codes: the C1 median, 1.023 (b: 1.111). The 11 C1 ratios range from 0.90 (J06) to 1.09 (K22).
  - So the real onset pairs of h are less over-dispersed than those of b.
- **Operational change on h (no change of method).** The cross-fitted logistic propensity (same IRLS, ridge, start and stopping rule) runs on the GPU in float64 for h's 256-dimensional state. On a test problem its propensities match the CPU version to 4e-16. The first h run on the CPU was stopped unfinished, before any output: a fit took minutes.

**Revision 13, result 2e: Γ̂² on h** (`outputs/phase05/12_h_rotation/gamma_h.*`; rotation runs in `gain/`, C1 runs in `gain_c1/`). The rotation used the 3e protocol on h: 182 withheld codes, none dropped.
- **Pooled DerSimonian–Laird estimate:** μ̂ 0.0006 [SE 0.0063], τ̂ 0, so **Γ̂² on h = 0.0110** (μ̂ + 1.645 SE).
- **Comparison with b:** μ̂ 0.0083 [SE 0.0032], Γ̂² 0.0135.
- **Overstated bootstrap variance.** Q/df is 0.41 on h, as on b (0.39), so the bootstrap variances overstate the null variance here too (rotation note of revision 10).
- **Relative to the effect,** the margin is larger on h. I48's injected ψ² at 1× is 14× Γ̂² on h, against 65× on b (0.882 / 0.0135).
- **Preview.** Before the end, `gamma_h.py --preview` was run once on the 96 finished codes, as a code test: Γ̂² 0.028. It does not enter any result.

**Revision 13, result 2b: γ_mix = 0 on h and rule (c)** (`outputs/part2/semisynth_h/gamma0_rev12.*`). The design is 8 draws × (6 G + 6 U), all G codes carrying the I48 vector of h, with the median propensity AUROC 0.713.

| Estimator | Scale | G-vs-U AUROC, mean (min) | G: new lower bound > 0 | G: lower bound > Γ̂² (0.0110) | G: CRT p < 0.05 |
|---|---|---|---|---|---|
| Oracle | 0.5× | 0.653 (0.500) | 0.15 | 0.06 | 0.33 |
| Oracle | 1× | 0.924 (0.833) | 0.58 | 0.56 | 0.79 |
| Oracle | 2× | 1.000 | 1.00 | 1.00 | 1.00 |
| Oracle | 4× | 1.000 | 1.00 | 1.00 | 1.00 |
| Part 1 | 0.5× | 0.663 (0.528) | 0.15 | 0.08 | 0.35 |
| Part 1 | 1× | 0.924 (0.833) | 0.58 | 0.56 | 0.81 |
| Part 1 | 2× | 1.000 | 1.00 | 1.00 | 1.00 |
| Part 1 | 4× | 1.000 | 1.00 | 1.00 | 1.00 |

- **Rule (c) on h: 1×**, the same scale as on b. The margin between 0.5× and 1× is wider than on b.
- **U codes:** the CRT rejects 3/48 (oracle) and 1/48 (Part 1); the new lower bound is > 0 and > Γ̂² in 0/48.
- **Noise:** before any noise, the event-pair ratios on h were already ≈ 1.00 (b: 1.2–1.3). After noise they sit at the targets: 1.20 for G, 1.02 for U.
- **Realism at 1× (report-only; real data: 0 of 5 on h).** Quarters, percentile lower bound > 0: 0.19 of G codes for both estimators.
  - By onset count: 777 → 0.88, 351 → 0.12, 302 → 0.12, and 0 for the counts 129, 185 and 204.
  - On h the environment is close to the real result, unlike on b (0.60–0.67 vs 1 of 5).
- **Margin reading.** With Γ̂² on h (result 2e; recomputed from the saved rows with `--summary-only`), G codes at 1× clear the margin in 0.56 for both estimators; on b it was 0.85 (Part 1) and 0.88 (oracle).

**Revision 13, result 2c: the γ_mix sweep on h at 1×** (`outputs/part2/semisynth_h/sweep.*`). The design is 8 draws × (6 G + 6 U + 5 × (6 UC + 3 GC)), all carrying the I48 vector; ψ²_true is 0.156.
- **Baseline codes.** The G codes are detected by the CRT in 0.75 (Part 1) and 0.73 (oracle), and lie above the F1 threshold in 0.90. The U codes: CRT 0.04, F1 0.04.

| γ_mix | UC, Part 1: F1 threshold / CRT p < 0.05 / new lower bound > 0 / median T | UC, oracle: same | GC bias, Part 1: median (10–90%) | GC: CRT p < 0.05, Part 1 |
|---|---|---|---|---|
| 0 | 0.02 / 0.04 / 0.04 / −0.002 | 0.04 / 0.04 / 0.02 / −0.003 | 1.02 (0.92–1.25) | 0.92 |
| 0.25 | 0 / 0.02 / 0.02 / −0.004 | 0.02 / 0.02 / 0.02 / −0.006 | 1.01 (0.86–1.28) | 0.88 |
| 0.5 | 0.06 / 0.02 / 0 / 0.002 | 0.06 / 0 / 0 / −0.002 | 1.02 (0.76–1.13) | 0.83 |
| 0.75 | 0.10 / 0.06 / 0 / 0.008 | 0.08 / 0.04 / 0 / 0.004 | 0.93 (0.52–1.08) | 0.67 |
| 1 | 0.04 / 0.02 / 0 / 0.001 | 0.04 / 0.02 / 0 / −0.007 | 0.98 (0.79–1.24) | 0.96 |

- **On h, false grounding of UC codes stays at the U level at every γ_mix** for both estimators: CRT ≤ 6%, F1 reading ≤ 10%. The CRT's rise seen on b at high γ_mix (15%) does not appear on h.
- **GC bias stays ≈ 1** (medians 0.93–1.02). It is noisier than on b (10–90% range down to 0.52 at γ_mix = 0.75), because I48's effect on h is small relative to h's noise.
- **Noise.** The ratio of UC codes before noise falls from 1.00 to 0.90 as γ_mix rises, because ED-only pairs are less variable. c rises accordingly, from 0.15 to 0.33.
- **Decision-rule reading (lower bound > Γ̂² = 0.0110).**
  - UC false grounding is 1/48 at γ_mix = 0 and 0/48 at every other γ_mix, for both estimators.
  - G codes clear the margin in 0.52, U codes in 0.
  - GC codes clear it in 0.79 / 0.67 / 0.54 / 0.50 / 0.67 (Part 1, γ_mix 0 to 1); oracle 0.75 / 0.67 / 0.54 / 0.50 / 0.67.

**Revision 13, result 2d: input-variant decomposition on h** (`outputs/part2/semisynth_h/decompose.*`).
- **Data.** The h sweep's draws at γ_mix 1 and 0.5, with noise redrawn and shared by the variants. One random stream per (draw, γ_mix): seed [1213, draw, 100 γ_mix]. On b it was one stream per draw.

| γ_mix | Variant | UC: CRT p < 0.05 [95% CI] | UC: lower bound > 0 | UC: median T | UC: propensity AUROC | GC bias, median (10–90%) |
|---|---|---|---|---|---|---|
| 1 | naive (h_t) | 0.04 [0.01, 0.14] | 0.02 | 0.004 | 0.53 | 0.98 (0.52–1.30) |
| 1 | z − s | 0 [0, 0.07] | 0 | −0.002 | 0.64 | 0.98 (0.54–1.30) |
| 1 | z − s + A (Part 1) | 0 [0, 0.07] | 0 | −0.001 | 0.64 | 0.98 (0.56–1.28) |
| 1 | oracle | 0.02 [0.00, 0.11] | 0 | −0.006 | 0.74 | 0.98 (0.58–1.28) |
| 0.5 | naive (h_t) | 0.02 [0.00, 0.11] | 0 | 0.003 | 0.53 | 1.06 (0.87–1.24) |
| 0.5 | z − s | 0.02 [0.00, 0.11] | 0 | −0.004 | 0.59 | 1.05 (0.85–1.20) |
| 0.5 | z − s + A (Part 1) | 0.02 [0.00, 0.11] | 0 | −0.005 | 0.59 | 1.04 (0.80–1.18) |
| 0.5 | oracle | 0.02 [0.00, 0.11] | 0 | −0.005 | 0.62 | 1.03 (0.79–1.18) |

- **Reading (descriptive): on h there is little confounding to remove.** Even with naive inputs, the CRT rejects 4% of UC codes at γ_mix = 1 and the median T is 0.004.
- **On b, the same driver gives 29% and 0.026 without z.** The recording route's driver shows up in the hand-crafted features (b), and z removes it. In h it is nearly invisible to begin with.
- **With Γ̂² on h = 0.0110** (result 2e), UC codes clear the margin in 0/48 under every variant at both γ_mix values. GC codes clear it in 0.63–0.75.

### Revision 12 (11 Nov 2026): user decisions, recorded before any UC/GC data are generated

**1. Null calibration on real pairs.**
- **Why.** On real b at γ_mix = 0, the stratified permutation test rejected 10–12% of U codes. The UC false-grounding rate is C2's primary metric, so this is fixed before the sweep.
- **Fix: a conditional randomization test (CRT) replaces the permutation test.**
  - Null labels ã ~ Bernoulli(π̂(x)) are drawn from the fitted, cross-fitted propensity, 500 draws.
  - For each draw, T is recomputed with the residuals held fixed. The p-value is (1 + #{T_k ≥ T}) / 501.
  - The likely cause of the failure: the injected labels depend on b_t, so they are not exchangeable within the permutation strata, whereas the CRT respects that dependence.
- **Validation.** U codes of the γ_mix = 0 generator (no added noise, as in that run), 96 per estimator: 16 draws × 6.
  - The original run did not store its label vectors, so its 48 U codes cannot be re-tested. Fresh draws from the identical generator replace them, and the stratified permutation test is recomputed on them for comparison.
  - The rejection rate is reported with a Clopper–Pearson 95% CI. **If the CI excludes 0.05 from above, stop and report.**
- **The bias-corrected interval is kept as is.**

**2. Realistic noise on injected event pairs, applied identically to G, U, UC and GC codes.**
- **(a)** κ < 1 (revision 10 e) was already used for G codes in the γ_mix = 0 run (`semisynth.py`: events recorded with κ = 0.9 / 0.117 by setting(t+1); n_events = n_recorded / κ̄). U codes carry labels only, so κ does not apply. It applies to G and GC in every run below.
- **(b) Matching the real noise of onset pairs.**
  - The residual-variance ratio is the mean squared base-model residual, summed over b's 39 dimensions in z-units, on recorded-onset pairs divided by the same on the other pairs.
  - After the κ mixture it is measured for each injected code, with the Part 1 estimator's residuals as the reference. Patient-bootstrapped residuals are then added to the code's event pairs, scaled by c with c² = max(target − ratio before noise, 0). The residuals come from the real stable pairs' Part 1 base model: a random patient, then one of that patient's pairs.
  - The same noise is given to both estimators, because it is part of the data. For G and GC, c is set per scale, since the effect itself adds to the ratio. The achieved ratio is recorded.
  - **Targets.** G codes take the real ratio of their source positive control: I47 2.36, I48 1.92, I49 1.90. The synthetic sparsity patterns (one-feature, two-feature, diffuse) take the median, 1.92. U, UC and GC codes take the median ratio on real onset pairs of the 11 C1 controls, **measured first: 1.11** (J06 0.80, K29 0.99, K59 1.01, G43 1.04, K80 1.10, K22 1.11, K21 1.13, Z96 1.14, K64 1.17, K57 1.21, N40 1.31).
- **(c)** γ_mix = 0 is rerun with (a), (b) and the CRT, and rule (c) is reapplied. The new scale is recorded even if it differs from 1×.
- **(d) Realism check (report-only),** at 1× with the real protocol. The environment is not tuned further to match the real result.

**3. UC/GC codes and the γ_mix sweep, at the chosen scale** (Part 1 estimator and oracle only; stop before the baselines).
- **UC and GC onsets.** Each onset follows the real driver with probability γ_mix, otherwise the risk profile of G and U.
  - The driver mechanism samples pairs with weight 10 when the next stay is ED-only and 1 otherwise ("high onset rate when the driver is on, low when off"). Odds ratio 10 lies within the real recording-route odds ratios of 2.4 (3–95).
  - γ_mix ∈ {0, 0.25, 0.5, 0.75, 1}.
  - GC codes add the effect, with κ < 1 like G codes. Their vectors are I47, I48 and I49 (3 GC codes per draw). UC codes have no effect (6 per draw, counts matched to the G codes).
- **G and U codes** are generated once per draw; they do not depend on γ_mix.
- **Metrics, both estimators.**
  - **False grounding of UC codes:** (i) per 3.4, the fraction of UC codes whose T exceeds the threshold that maximizes the G-vs-U F1 in the same draw; (ii) the decision rule, new lower bound > Γ̂² = 0.0135; (iii) CRT p < 0.05.
  - **GC effect bias:** the ratio of the estimated to the injected effect norm, √max(T, 0) / √ψ²_true; also GC detection.
- **Draws:** 8 (5–10 in 3.2).
- **Operational details** (fixed with the code, before any UC/GC data).
  - For GC codes, the number of events is n_recorded / E[κ], where E[κ] mixes the driver-weighted mean of κ (γ_mix) and the overall mean (1 − γ_mix). The expected number of recorded onsets is therefore the same at every γ_mix, so power does not change with γ_mix through the counts.
  - The noise donors are the Part 1 base-model residuals of the real stable pairs. c solves the ratio equation exactly, cross terms included. The CRT does not refit the propensity for each null draw.
  - Caveat, stated in advance: the added noise makes the event pairs' residual variance depend on the label. The CRT tests full conditional independence, so it may reject null codes somewhat more often once the noise is on. The bias-corrected interval does not test independence, so the decision rule (lower bound > Γ̂²) is not affected in the same way.
  - Code: `gonogo/part2/semisynth.py` (`crt-validate`, `gamma0`, `sweep`). Seeds 1210, 1211 and 1212, one stream per draw. Label vectors are saved.

**Revision 12, result 1: CRT validation passes** (`outputs/part2/semisynth/crt_validation.*`). The test set is 96 U codes per estimator from the γ_mix = 0 generator (16 draws × 6), without added noise; rejection is at p < 0.05.

| Estimator | CRT rate [95% Clopper–Pearson CI] | KS p (CRT p-values ~ U(0,1)) | Permutation, same codes |
|---|---|---|---|
| Part 1 | 4/96 = 0.042 [0.011, 0.103] | 0.94 | 4/96 = 0.042 [0.011, 0.103] |
| Oracle | 4/96 = 0.042 [0.011, 0.103] | 0.66 | 5/96 = 0.052 [0.017, 0.117] |

- Neither CI excludes 0.05 from above, so the run continues.
- **The permutation test did not over-reject on these fresh codes.**
  - Revision 11's 10–12% was 6/48 [0.047, 0.252] for Part 1 and 5/48 [0.035, 0.227] for the oracle, and both CIs include 0.05. That flag is therefore compatible with chance at 48 codes per estimator.
  - Pooling both runs, the permutation test rejected 10 of 144 codes per estimator (0.069).
  - The CRT replaces it as decided, and is no worse on these codes.
- The new interval's lower bound is above 0 for 1% of the U codes.

**Revision 12, result 2: γ_mix = 0 rerun and rule (c)** (`outputs/part2/semisynth/gamma0_rev12.*`). The rerun has κ < 1 on G codes, realistic noise and the CRT: 8 draws × (6 G + 6 U), κ̄ 0.663, median propensity AUROC 0.733.

| Estimator | Scale | G-vs-U AUROC, mean (min) | G: lower bound > Γ̂² | G: CRT p < 0.05 |
|---|---|---|---|---|
| Oracle | 0.5× | 0.899 (0.722) | 0.38 | 0.79 |
| Oracle | 1× | 0.976 (0.861) | 0.88 | 0.96 |
| Oracle | 2× | 1.000 | 0.98 | 1.00 |
| Oracle | 4× | 1.000 | 1.00 | 1.00 |
| Part 1 | 0.5× | 0.910 (0.722) | 0.38 | 0.77 |
| Part 1 | 1× | 0.976 (0.861) | 0.85 | 0.96 |
| Part 1 | 2× | 1.000 | 0.98 | 0.98 |
| Part 1 | 4× | 1.000 | 1.00 | 1.00 |

- **Rule (c): C2 is read at 1×, unchanged.** The oracle's mean AUROC at 0.5× is 0.899, just under 0.9.
- **U codes (θ = 0).** The CRT rejects 3/48 per estimator [0.013, 0.172]. The new lower bound is above 0 in 2% and above Γ̂² in 0%.
- **Noise: the ratio of U codes was above its target before any noise.**
  - The Part 1 residual-variance ratio of U codes was 1.18–1.30 (median per vector) before noise, above the 1.111 target, so c = 0 for almost all U codes.
  - The risk profile already selects pairs with larger residual changes than the real C1 onset pairs have. Noise cannot lower a ratio, so for U codes the injected pairs remain somewhat noisier than real null onsets. The same holds for UC codes at low γ_mix. This is reported, not tuned.
  - G codes before noise: 1.17–1.46 at 1×. After noise they sit at their targets, except I48 at 4× (2.38 > 1.92 before noise).
- **Realism check at 1× (report-only; real data: 1 of 5).** Quarters 10–14 vs 15–19, percentile lower bound > 0:
  - Oracle: 0.67 of G codes. By vector: I47 0.62, I48 1.00, I49 0.50, one-feature 0.00, two-feature 1.00, diffuse 0.88.
  - Part 1: 0.60. By vector: I47 0.62, I48 1.00, I49 0.38, one-feature 0.00, two-feature 1.00, diffuse 0.62.
  - Before the noise (revision 11) these were 0.71–0.73. The gap to the real 1 of 5 narrows but remains. As decided, the environment is not tuned further.

**Revision 12, result 3: the γ_mix sweep at 1×** (`outputs/part2/semisynth/sweep.*`; Part 1 estimator and oracle only; baselines not run).
- **Design.** 8 draws × (6 G + 6 U + 5 γ_mix × (6 UC + 3 GC)). G and U are at γ_mix = 0. G is detected by the margin in 0.83 of codes for both estimators; U codes are falsely grounded in 0 of 48.
- **Driver strength.** 81% of the driver's onsets fall at an ED-only next stay, against 30% of the pool. The mean κ is 0.26 under the driver and 0.66 under the risk profile. Recorded GC onsets stay at 418–431 at every γ_mix, by construction.

| γ_mix | UC false grounding, Part 1: F1 threshold / lower bound > Γ̂² / CRT p < 0.05 | Oracle: same three | GC effect bias, Part 1: median √T₊/√ψ² (10–90%) | GC detected (lower bound > Γ̂²), Part 1 |
|---|---|---|---|---|
| 0 | 0.02 / 0/48 / 0.04 | 0.02 / 0/48 / 0.04 | 1.01 (0.94–1.08) | 1.00 |
| 0.25 | 0.02 / 0/48 / 0.06 | 0.04 / 0/48 / 0.06 | 0.98 (0.89–1.04) | 1.00 |
| 0.5 | 0.06 / 0/48 / 0.02 | 0.06 / 0/48 / 0.02 | 1.00 (0.87–1.09) | 0.96 |
| 0.75 | 0.04 / 1/48 / 0.15 | 0.04 / 1/48 / 0.10 | 0.97 (0.85–1.08) | 1.00 |
| 1 | 0.04 / 0/48 / 0.15 | 0.02 / 0/48 / 0.06 | 0.98 (0.90–1.10) | 1.00 |

- **UC false grounding.**
  - By the decision rule, it stays at 0–2% at every γ_mix for both estimators.
  - The F1-threshold reading stays at 2–6%.
  - Only the CRT moves. For Part 1 it rises from 4% to 15% at γ_mix ≥ 0.75, against 6–10% for the oracle. This is the direction confounding predicts, but well under the margin. Part of it may come from the label-dependent noise (caveat above): c rises to 0.40–0.52 at high γ_mix, because ED-only pairs have 0.72× the residual variance of the other pairs.
- **GC.** The effect bias is ≈ 1 at every γ_mix (medians 0.97–1.01), and the Part 1 estimator and the oracle agree within 0.01.
- **Why the confounding channel is weak here** (diagnostic, `driver_as_label.json`).
  - The driver itself, used as a label, has a Part 1 T of 0.032 [0.023, 0.040]: above Γ̂², and CRT p 0.002. Without A in the base model it is 0.055.
  - A UC code inherits only part of that: 81% vs 30% ED-only, a rare label (≈ 325 onsets), and the residualization on setting(t), which predicts setting(t+1).
  - At this driver strength (odds ratio 10), confounding by the recording route therefore does not reach the margin. The sweep does not show what a stronger driver, or a driver that also moves the state's change, would do.
- **Status.** Stopped here, before the baselines (Section 3.3), as instructed.

### 3.1 Latent synthetic environment

A small numpy generator (about 200 lines) that produces pairs `(h_t, h_{t+1})` directly in representation space; no signal is rendered.

| Component | Definition |
|---|---|
| Patient identity `p` | `p ~ N(0, σ_p² I)` in `R^d`; `σ_p` set so the identity share of variance matches 2.9 (Phase 0: 0.74 in `h`, provisional) |
| State | `h_t = p + s_t`, with `s_{t+1} = ρ s_t + drift(Δt) + context shift + code effects + noise` |
| Hidden context `u_t` | binary Markov "acute/admitted" state; when on, shifts the transition by a fixed vector `β` |
| Time gap Δt | sampled from the empirical MIMIC Δt distribution of cross-stay pairs |
| Stable fraction | matched to 2.9: 29% on cross-stay pairs between episodes. Revision 2's 73% and revision 3's 48% included within-stay and same-episode pairs, which are stable by construction. |
| Dimension | `d = 32` for sweeps; spot-check at `d = 256`, the size of the real `h` |

**Code types (ground truth known).**

| Type | Onset probability depends on | Effect on `h_{t+1}` |
|---|---|---|
| **G**, grounded | patient risk score (logistic in a projection of `p`) | adds a fixed effect vector `Δ_c` |
| **U**, ungrounded | patient risk score | none |
| **UC**, ungrounded, confounded | patient risk score + `γ · u_{t+1}` (logit scale) | none; the transition still moves through `β` when `u` is on |
| **GC**, grounded, confounded (secondary) | patient risk score + `γ · u_{t+1}` | adds `Δ_c` |

Use 4–6 codes each of G, U and UC, and 2–3 GC codes, multi-label with sparse co-occurrence.

**Annotation noise.**
- Flip each label with probability `ε + ε_u · u_{t+1}`. The context-dependent part `ε_u` implements the recording route: coding intensity depends on the context, for every code type.
- Delay an onset by one step with probability `p_delay`.

**Sweep.**

| Knob | Values |
|---|---|
| Confounding strength γ (logit scale) | 0, 0.5, 1, 2, 4 |
| Effect size of G and GC codes | 0.5×, 1×, 2× the median positive-control effect norm from 2.9 (measured effects are small: 0.2–0.6 SD of the stable pairs' change on the affected feature). Revision 5: 0.5×, 1×, 2× the debiased norm of I48's effect vector, the only one whose CI excludes 0 in Phase 0, or of the representation chosen in Phase 0.5 |
| Label noise ε | 0, 0.05, 0.15 |
| Context-dependent noise `ε_u` | 0, 0.1 |
| Delay `p_delay` | 0, 0.2 |
| Patient dominance | value from 2.9 (Phase 0: 0.74), and 0.5× |

**Information given to methods.** `h_t`, `h_{t+1}`, Δt and the labels. The context `u` is hidden from every method except the oracle.

### 3.2 Semi-synthetic test on MIMIC-IV-ECG (main test)

Real pairs, real representations and real confounding; only the injected codes are artificial.

**Base data.** The Part 1 clean subset of cross-stay pairs, with frozen `h`, real Δt and the real code vector.

**Injected codes.** Draw the full set of injected codes 5–10 times independently and average results over draws.

| Type | Onset assignment | Injected effect |
|---|---|---|
| **G** | logistic in age and a random projection of `h_t` (a "risk profile", the shortcut seen in the workshop model) | `h_{t+1} ← h_{t+1} + Δ_c`, with `Δ_c` taken from the positive-control effect vectors of 2.9, scaled 0.5×, 1×, 2×. Revision 5: `Δ_c` is I48's vector (`effects_h.npz`), scaled, or the positive-control vector of the representation chosen in Phase 0.5 |
| **U** | same risk profile | none |
| **UC** | with probability `γ_mix`, follows the real driver from 2.9 (high onset rate when the driver is on, low when off); otherwise the risk profile. Revision 5: the driver is the care setting at t+1 | none |
| **GC** (secondary) | as UC | as G |

Onset rates match the median onset rate of real codes that pass the Part 1 event threshold. Pairs are treated independently, so labels and effects are applied per pair; persistence across consecutive pairs is not modeled in the go/no-go.

**Strength control.** `γ_mix ∈ {0, 0.25, 0.5, 0.75, 1}`. At `γ_mix = 0`, UC codes follow the same mechanism as U codes (sanity check). At `γ_mix = 1`, they are exactly as confounded as the real driver.

**Information given to methods.** `h_t`, `h_{t+1}`, Δt and the code vector (real and injected). The driver is withheld; the oracle additionally receives it. Two input variants are run for every method:
- with and without the real codes, which are themselves proxies of the context;
- with and without the observed variables of `z` other than the driver, since a natural baseline would condition on them.

**Revision 5, inputs.** Every method, baselines included, receives the real codes as inputs, including the no-route set A. Phase 0 showed that the label vector recovers a substantial share of the withheld setting (exploratory; rerun in 0.5a and 0.5b under the frozen set of 0.5d), so a baseline denied the codes would fail for a reason a practitioner would not face. The variants above without real codes, and with the observed `z`, are kept as secondary analyses. The driver stays withheld from every method but the oracle.

### 3.3 Baselines

All operate on the same frozen `h`, with the same training budget and 3 seeds.

1. **Monolithic action-conditioned predictor**: the workshop dynamics head (`Dyn_φ` with `Proj_ω`), trained on frozen `h`.
2. **AD-JEPA centering**: action-mean-centered predictor, trained-in and post hoc.
3. **Sparsity**: learned binary gates or L1 on the action→latent mapping (mechanism-sparsity style); SPARTAN-style sparse attention if feasible.
4. **LAOM-style**: latent action inferred from `(h_t, h_{t+1})`, codes used as supervision.
5. **CAI threshold**: per-code conditional mutual information from a probabilistic dynamics model.
6. **Oracle**: the best of the above, also given the hidden context (`u` in 3.1, the driver in 3.2). Upper bound.

Each method must output a **per-code grounding score**: gate value, edge weight, action-channel norm, or CMI.

### 3.4 Metrics

| Metric | Definition | Main use |
|---|---|---|
| **False grounding rate (UC)** | fraction of UC codes scored above the threshold that maximizes G-vs-U F1, averaged over code draws | **primary** |
| False ungrounding rate (G) | fraction of G codes scored below that threshold. Penalized and equivalence-type tests can declare weak grounded codes ungrounded through shrinkage alone | the opposite error; reported next to the primary |
| Grounding AUROC | separate G from U ∪ UC | overall detection |
| Effect recovery | R² between estimated and injected effect vectors of G codes | counterfactual quality |
| Effect bias (GC) | ratio of estimated to injected effect norm of GC codes | over-attribution under confounding |
| Leakage | ridge-probe R² of the action channel against the hidden context | should be ≈ 0 |
| Predictive gain | improvement over the copy baseline on changed pairs | sanity |

**Revision 5, significance.** Every permutation test in Part 2 uses 500 permutations, as does any rerun of the Part 1 MLP gain. Phase 0's 100 permutations left a BH floor of q = 0.08 over eight codes.

**Revision 9, pre-declared before any Part 2 data are generated (10 Nov).**
- **Part 2 is the main evidence that the method recovers effects.** On real codes, the omnibus ψ̂² did not replicate b's positive controls on folds 10–19 (0.5a results), although the per-feature readouts confirmed their directions.
- **Representation.** b is the representation chosen in 0.5a. The semi-synthetic injections of 3.2 therefore use b's space and b's positive-control effect vectors (the revision-5 note "or the positive-control vector of the representation chosen in Phase 0.5").
- **Secondary sparse statistic.** For each code c, T_c = max_k |θ̂_ck / ŝe_k|.
  - θ̂_ck is the cross-fitted per-feature contrast of c on feature k, in SDs of that feature's stable-pair change: the R-learner's per-dimension effect, or the code channel's contrast for a world model.
  - ŝe_k is its patient-bootstrap standard error.
  - **Null.** Its own stratified permutation null: c's labels are shuffled within strata of gap quintile × setting(t+1) × age tertile, 500 permutations, recomputing T_c each time. The maximum over features is recomputed in every permutation, so the null controls the family-wise error over features.
  - **Decision.** p_T = (1 + #{T_perm ≥ T_c}) / 501. A code is "detected by the sparse statistic" if p_T < 0.05, with BH across codes within each code type. It is reported next to ψ̂²'s decision and does not replace it.
- **Concentrated-effect scenarios.** Added to the effect-size knob of 3.1 and to the injected effects of 3.2.
  - **One feature:** Δ_c has a single non-zero coordinate, for example b's QRS width, like a conduction block.
  - **Two features:** two non-zero coordinates, for example HR and RR irregularity, like atrial fibrillation.
  - **Spread:** the same squared Mahalanobis norm spread over all features (the existing setting).
  - **Effect sizes:** the per-feature readouts of the positive controls, 0.2–0.6 SD per feature, at 0.5×, 1× and 2×.
  - **Counts:** onsets at the real clean counts of 3.6 (129–777).
- **Reported in each scenario,** for G codes: the detection rate of ψ̂² (lower bound above 0, and lower bound above Γ̂²) and that of the sparse statistic. For U and UC codes, the rate of each statistic. This characterizes where ψ̂² loses power to the sparse statistic. The C2 decision rule (3.5) is unchanged.

### 3.5 Decision for C2

- **GO.** In the semi-synthetic test at `γ_mix = 1`, all baselines separate G from U well (AUROC ≥ 0.9), but every non-oracle baseline has false grounding on UC ≥ 0.5, while the oracle stays ≤ 0.2. In the latent sweep, false grounding rises with γ.
  - Key figures: **false grounding rate vs `γ_mix`** on MIMIC, with real confounding at the right end, and **false grounding rate vs γ** in the latent sweep. One curve per method, oracle flat.
- **NO-GO.** At least one non-oracle baseline keeps false grounding on UC ≤ 0.2 at `γ_mix = 1`; existing tools handle the real confounding. If the latent sweep shows failures at larger γ, the problem exists in principle but not at the strength observed in MIMIC.
- **Partial.** The best non-oracle baseline lies between 0.2 and 0.5 at `γ_mix = 1`. The problem is real but moderate; decide from the effect bias on GC codes and from whether unrecorded contexts are plausibly stronger than the chosen driver.

### 3.6 Power at real event counts

Grounded effects in this data are small: positive controls reach a within-patient AUROC of 0.61–0.69, and no code explains more than 3.1% of any feature's change (1.1). Before the method is built, check that grounding is detectable at all at real event counts.

- **Setup.** In the latent environment, set onset counts to the real distribution of clean onsets between episodes: a median of 351 over the 13 cardiac codes that pass the threshold (252–4,605), and 129–777 for the positive controls. Revision 3's figures (median 1,306 over 47 codes, 1,074–6,776 for the positive controls) counted all cross-stay onsets. Set effect sizes to the measured ones.
- **Report.** The fraction of G codes that the oracle, each baseline, and the Part 1 estimator detect.
- **Decision rule.** If the oracle detects few G codes at these counts, the method cannot be demonstrated on real codes. The paper must then rest on the benchmark and on the semi-synthetic test.

**Result of the power check, b-calibrated latent environment** (revision 10, 10 Nov 13:58; `outputs/part2/latent/power_b.{md,json}`, rows in `power_b_rows.csv`; script `gonogo/part2/latent.py`; after the correction of the risk profile recorded in revision 10 (c)).
- **Calibration.** dim 39; patient share 0.485; R² of the change from the state 0.254; stable fraction 0.295; ED-only stay at t+1 on 30% of stable pairs; 1 − ρ = 0.79; κ̄ = 0.66; propensity AUROC median 0.67.
- **Design.** 8 draws × (6 G + 6 U codes) at the real clean onset counts, with 500 bootstrap draws.
- **Validation passes.** U codes are never "detected" (0.00 at both thresholds). Under the conservative bootstrap (margin sensitivity) this is below the nominal rate.

| Estimator | Scale | G-vs-U AUROC (mean over draws) | G detected: lower bound > 0 | > Γ̂² = 0.014 | quarters (10–19), lower bound > 0 |
|---|---|---|---|---|---|
| oracle | 0.5× | **0.969** | 0.67 | 0.60 | 0.29 |
| oracle | 1× | 1.000 | 0.85 | 0.85 | 0.81 |
| oracle | 2× | 1.000 | 1.00 | 1.00 | 1.00 |
| oracle | 4× | 1.000 | 1.00 | 1.00 | 1.00 |
| Part 1 estimator | 0.5× | 0.969 | 0.69 | 0.62 | 0.29 |
| Part 1 estimator | 1× | 1.000 | 0.85 | 0.85 | 0.81 |
| Part 1 estimator | 2× | 1.000 | 1.00 | 1.00 | 1.00 |
| Part 1 estimator | 4× | 1.000 | 1.00 | 1.00 | 1.00 |

- **By vector** (Part 1 estimator, lower bound above Γ̂²):

| Vector | 0.5× | 1× |
|---|---|---|
| I48 | 1.00 | 1.00 |
| two-feature | 1.00 | 1.00 |
| diffuse | 1.00 | 1.00 |
| I47 | 0.50 | 1.00 |
| I49 | 0.25 | 1.00 |
| one-feature (QRS width, ψ² 0.15 at 1×) | 0.00 | 0.12 |

  Every vector is detected at 2×.
- **Rule (c): the oracle reaches AUROC ≥ 0.9 already at 0.5×, so C2 is read at 0.5×.**
  - The oracle and the Part 1 estimator agree at γ = 0, because the hidden context shifts the change only slightly (|β| = 0.17).
  - The rule uses point estimates (AUROC of ψ̂²). At 0.5×, 60–62% of G codes clear the margin, and the single-feature effect is never detected.
- **3.6 rule.** At the measured effect sizes (1×), the oracle detects 85% of G codes at real counts, which is not "few". The method can be demonstrated in principle.
- **Caveat: the environment is more optimistic than the real data.**
  - At 1× with quarter folds, 81% of simulated G codes clear 0. On real folds 10–19, 1 of 5 positive controls did.
  - Two sources are known.
    - The injected 1× vectors are the all-fold plug-in θ, whose norms are 1.07–1.6 times the debiased ψ̂² (I47 0.75 against 0.51; I49 0.98 against 0.63).
    - The generator is Gaussian, while b's changes are heavy-tailed (the distance table): the real noise is larger than the Gaussian Σ_b suggests for these statistics.
  - Neither changes rule (c), which was applied as declared.

**Estimated cost:** about 1.5 weeks (latent generator 1–2 days, semi-synthetic injection 1–2 days reusing the Part 1 pipeline, baselines 4–5 days, sweeps, power check and analysis 2–3 days).

---

## 4. Timeline

| Week | Work | Checkpoint |
|---|---|---|
| 0 | credentialed downloads and linkage of ECGs to stays and care settings; meanwhile, Part 1 on `b` and `h` with Δt only | data ready (done 25 Oct) |
| 1 | Part 1: full `z`, anchors, recording route (2.4), estimator, permutation tests | **C1 decision** (Phase 0, done 27 Oct: GO for observed confounding, provisional) |
| 1.5 | Phase 0.5 (revision 5): encoder check (0.5a), provenance check (0.5b), Γ̂ by rotation (0.5c), with short Task 1–2 reruns on the chosen representation | representation chosen; framing (0.5b) and margin (0.5c) decided |
| 2 | Part 2: latent generator, power check (3.6), semi-synthetic injection, baselines 1–3 | environments validated: the Part 1 estimator recovers injected G codes, finds no gain for U codes, and treats UC like U at γ = 0 and `γ_mix = 0` |
| 3 | Part 2: baselines 4–6, sweeps, figures | **C2 decision** |

*Revision 5: Part 2 starts after Phase 0.5, about 3–4 days later than first planned. Its driver, inputs, injected vectors and permutation count follow the revision-5 notes in Section 3.*

*Revision 10 timeline (from 10 Nov; Phase 0.5 done on b). Replaces weeks 2–3 above; order of 3, revision 10 (g):*

| Step | Work | Checkpoint |
|---|---|---|
| 2.0 | b-calibrated latent environment and the 3.6 power check (oracle and Part 1 estimator, scales 0.5–4×) | detection rates; **effect scale chosen by rule (c)**, or C2 cannot be read |
| 2.1 | semi-synthetic test on b (primary) and on h (secondary), at the chosen scale and the sweep | environments validated at γ_mix = 0 |
| 2.2 | baselines 1–3 | — |
| 2.3 | baselines 4–6 | — |
| 2.4 | the method, scored by its own classes | **C2 decision** on b (3.5) |

If C1 is NO-GO at the end of week 1, Part 2 still runs, but without UC and GC codes (the real driver would not confound anything), and it only serves the re-scoped question in Section 6.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Label artefacts: 52.6% of consecutive ECG pairs include an ECG outside any stay, whose codes are missing | cross-stay pairs only; treat outside-stay labels as missing (2.1) |
| Too few onsets for most codes | cross-fitting over all folds (47 codes pass); pool rare codes into clinically related groups; report insufficient-data codes explicitly |
| Low power: grounded effects are small | power check in 3.6 before building the method |
| Coding depends on the context, which violates the method's non-differential label noise | measured in 2.4; method change in Section 7, item 1 |
| Anchors invalid (venous thrombosis travels with pulmonary embolism) or reused across roles | non-cardiac anchors from the full Ext-ICD labels; disjoint roles fixed in advance (2.2) |
| External data not yet downloaded | start Part 1 on `b` and `h`; downloads in week 0 |
| `m` too coarse to detect real effects | add `b` (per-lead amplitudes, RR irregularity); rely on `Δh` with negative controls to disambiguate |
| Confounding fully explained by observed `z` | conditioning on `z` then solves the real problem; the semi-synthetic test withholds the driver from methods, and the paper must argue that real deployments miss such contexts |
| Injected effects live in `h`, so the semi-synthetic test does not check whether the encoder captures signal-level effects | Part 1 positive controls show that the frozen encoder captures real effects (I44, I48); signal-level injection (e.g. widening the QRS of real ECGs, or an ECGSYN generator) stays an optional follow-up if reviewers ask |
| The chosen driver is a weak stand-in for the real context | report results for each candidate driver from 2.9; use the latent sweep to show where failures begin |
| Real codes partially deconfound baselines by acting as proxies of the context | run every method with and without real codes as inputs; report both |
| Fast-moving field (ECG-WM, AD-JEPA) | position explicitly against AD-JEPA, LAOM, SPARTAN and ECGTwin in the related work from day one |
| Same-episode pairs (an ED visit and the admission it led to) are stable by construction, 26.9% of cross-stay pairs | excluded and used as a noise floor (2.1) |
| The label source switches with the setting: ED-only stays carry ED discharge codes, which omit chronic conditions | setting at t+1 in `z`; recording route measured (2.4); chronic codes are informative only on pairs whose later stay is an admission |
| The coding system changes between stays (ICD-9 converted to ICD-10 by Ext-ICD) and shifts onset rates (I48 OR 1.43) | coding system at t and t+1 in `z` (2.1) |
| Ext-ICD keeps ICD-9 V-codes whose strings are also valid ICD-10 codes | external-cause chapters V–Y excluded from the pool; cardiac labels unaffected (2.1) |
| `machine_measurements.csv` has wrong dates for 2.2% of ECGs | joined on `study_id` only; values checked against the waveform (HR r = 0.98) |
| The plan's example negative controls (cataract, osteoarthritis, fractures) are too rare on clean pairs | pool rebuilt from the data, roles fixed in advance (2.2) |
| Two separately trained predictors differ by more than the small codes' gains | cross-fitted R-learner gain with a conditional randomization test as the primary statistic (2.3, 2.5) |
| (Revision 5) The encoder `h` suppresses within-patient change: patient share 0.74, code transitions explain R² ≈ 0.0002 of `Δh`, and I48's gain is 25 times larger on `Δb` | encoder check (Phase 0.5a); encoders trained with same-patient positives are excluded (Section 7, item 10) |
| (Revision 5) Conditioning on ECG-route codes absorbs part of grounded effects (18–22% of I48's) | report effect magnitudes with and without route codes in the conditioning set (Section 7, item 10) |
| (Revision 5) What the no-route set A recovers may be label provenance only | provenance check (Phase 0.5b), whose rule sets the framing |
| (Revision 5) An unexplained residual remains after `z` (K59) | Γ̂ by rotation (Phase 0.5c); handled by the margin, not removed |
| (Revision 5) Only I48's effect vector clears zero in `h` | Part 2 injects I48's vector, scaled; the 3.6 rule decides whether real-code grounding can be shown |

## 6. Outcomes and next steps

| C1 | C2 | Next step |
|---|---|---|
| GO | GO | Develop the method (negative-control anchors; see `method_section.tex`, with the changes of Section 7). Evaluation: latent synthetic + MIMIC-IV-ECG (semi-synthetic and real codes). Optional second domain: MIMIC-CXR, which shares patients, codes and admissions, so the same linkage and injection protocol apply. Target: main track. |
| GO | NO-GO | The phenomenon exists but existing tools handle it. Options: a benchmark paper (latent generator + semi-synthetic protocol + baselines) for NeurIPS Datasets & Benchmarks or TMLR, or an atlas of which ICD codes are grounded in the ECG, built with the best baseline. |
| NO-GO | — | Drop the confounded axis. Options: an analysis paper ("diagnosis codes are not actions": variance decomposition, per-code grounding map, action ≈ zeroed-action ablation, risk-shortcut analysis), likely for ML4H, CHIL or TMLR. Much of its evidence already exists (1.1). Alternatively, re-scope to weak + ungrounded labels after a careful comparison with the sparsity literature. |

**Revision 5 notes (the table and its rules are unchanged).**
- **C1 is recorded as GO for observed confounding** (Phase 0 results, provisional).
- **The anchor premise of 2.3, "(2) vs (3)", is not met while `z` is observed.** By 2.3's own rule, the method then has nothing to work with in that regime. Its value lies where encounter metadata are missing, and Phase 0.5b qualifies what it recovers there.
- **How Phase 0.5 enters the path:**
  - If 0.5a finds no representation with three positive controls above zero, take the benchmark options of the GO / NO-GO row, whatever C2 says.
  - If 0.5b finds that A recovers provenance only, a GO / GO outcome keeps the method with the modest framing: anchors recover label provenance when metadata are missing.
  - If 0.5c finds that no positive control clears Γ̂, real-code grounding results are reported as inconclusive, and the paper rests on the benchmark.

## 7. Changes required in `method_section.tex`

These follow from the evidence above and should be made before any method work.

1. **Allow the recording route.** Let labels depend on the context, `a ~ P(· | e, u)`. Under proxy sufficiency, the grounding null still holds: an ungrounded code has no effect after adjustment. Effect magnitudes (`κ_c`) and the EM noise model, however, need the context. Restrict the non-differential noise assumption to effect sizes, and note that anchors then also proxy coding intensity.
2. **Drop the centering over label values, or center over admissible values only.**
   - The mechanism basis already makes a code channel zero when its label does not change.
   - Centering leaves the onset contrast unchanged.
   - What it does add is an offset on every stable pair, and it averages over a label value that is inadmissible given `y_t`, since only onset or resolution is possible.
3. **Separate selection from inference in the grounding test.** Compute `ψ²` and `S_c` from an unpenalized refit on a split not used to fit the gates. Otherwise shrinkage biases `ψ̂²` toward zero, the bootstrap is unreliable near zero, and the equivalence test can declare weak grounded codes ungrounded (the false ungrounding rate of 3.4).
4. **Fix the anchor set in advance.** Draw it from clinical knowledge and the full Ext-ICD labels, in the disjoint roles of 2.2. Check that anchors do not co-occur with grounded codes, as venous thrombosis does with pulmonary embolism; otherwise the context term absorbs grounded effects.
5. **Note that adjusting for all other codes (`a_{−c}`) can block mediated effects** through coding cascades, e.g. an MI leading to I25 and I50 being coded.
6. **Position the theory honestly.** Propositions 1–4 are standard: omitted-variable bias, backdoor adjustment with misclassification attenuation, Miao et al., and a margin argument. Present the contribution as three things:
   - the problem formalization for world models;
   - the anchor-calibrated grounding test with explicit non-identifiable and inconclusive classes;
   - the benchmark.

   Cite the negative-control literature (Lipsitch et al. 2010; Shi, Miao and Tchetgen Tchetgen 2020). Verify the 2026 citations (AD-JEPA, Lachapelle et al.) and the claims attributed to them.
7. **Minor fixes.**
   - Index `û_t` as the proxy of `u_{t+1}`.
   - State the propositions for the unpenalized objective.
   - Define how counterfactual rollouts are evaluated.
8. **Model the label source (revision 4).** In Ext-ICD the label of an ED-only stay is its ED discharge codes, and that of an admission its hospital discharge codes. Chronic codes are therefore omitted at ED-only stays almost always (2.4), and Ext-ICD converts ICD-9 codes, so the coding system changes too. Both are observed, so they belong in `z` and in the observation model of the weak-label section as covariates. The "not identifiable" class must cover codes whose onsets the later stay's setting determines, since codes recorded only during admissions have no onsets into ED-only stays.
9. **Qualify the anchor description (revision 4).** "Coded often and in every care setting" holds only for acute complaints. Chronic non-cardiac codes are recorded only during admissions, so they proxy coding intensity rather than the acute context. The approved anchor set contains both kinds (2.2), and the method section should say which role each kind plays.
10. **Phase 0 consolidation (revision 5; done 27 Oct in `method_section.tex`, which contains no MIMIC numbers).**
    - **Anchors.** The anchor set is the full set of no-route codes under a documented rule (an appendix table), not a curated list. Individual anchors are weak proxies; the context is recovered from many of them. This also implements item 9.
    - **Provenance.** The process that writes a label depends on the care setting, so provenance is a confounder that the label vector reveals. Pairs labelled from one document are excluded (implements item 8).
    - **Recording assumption.** Non-differential recording is assumed given the observed encounter covariates `z`, not unconditionally.
    - **Static covariates.** Age and sex enter the drift term and the context estimator.
    - **Over-adjustment.** Conditioning on ECG-route codes is discussed after proxy sufficiency. Magnitudes are reported with and without those codes.
    - **Proximal extension.** It is presented as optional, used only with validated outcome-side proxies.
    - **Residual bias.** Γ̂ is a rotation estimate over held-out no-route codes, always excluding the code under test. An unexplained residual is handled by the margin, not removed.
    - **Encoder.** The frozen encoder must not be trained with same-patient positives, with positive-control codes as the check.
11. **Revision 6 (3 Nov, in `method_section.tex`; no MIMIC numbers).**
    - **Γ̂.** The margin is a random-effects (DerSimonian–Laird) estimate over the withheld no-route codes, μ̂ + 1.645 τ̂, not an upper quantile of point estimates. prop:wm-sensitivity now states that a quantile-based margin protects codes whose residual lies within it, not the worst case.
    - **Over-adjustment.** The claim that conditioning on `a_{t,−c}` cannot make an ungrounded code appear grounded is removed. Tolerance is restricted to mediators. Colliders can create a spurious association. `Δ_c` is defined with the other events held fixed, which conditioning on `a_{t,−c}` approximates. *(Revision 7: the single-change estimate is no longer the collider check, because it conditions on the other codes at zero; see item 12.)*
12. **Revision 7 (3 Nov, in `method_section.tex`; no MIMIC numbers).**
    - **Collider check.** Each grounding decision is repeated without the other non-anchor labels in the conditioning set, keeping `z_t` and the anchor labels in both versions. A decision that holds only with them is flagged as possibly induced by a collider. The single-change estimate is kept only as a descriptive sensitivity analysis.
    - **Encoder candidates.** Only representations whose learned part satisfies the encoder requirement are candidates. The tie-break is concatenation with fixed features, then the learned representation alone, then fixed features. An encoder with same-patient positives is a reference only.
    - **Anchor history.** "Applied … before any result is examined" is replaced by "frozen before the pre-specified analyses" in the definition of `C_0`, the roles paragraph and Scope. A sentence adds that the form of the anchor set was informed by exploratory analyses of the same data. The generated appendix says the same.
    - **Typesetting.** The observation model of §wm-weak is a display (`eq:wm-obsmodel`), and the posterior is split over three lines.
13. **Revision 8 (9 Nov, in `method_section.tex` §wm-test and §wm-theory; no MIMIC numbers).**
    - Σ is estimated with Ledoit–Wolf shrinkage of the standardized stable-pair changes, per representation and on the patients of the statistic. The reason: a learned representation can vary along far fewer directions than it has, which makes the sample covariance ill-conditioned. The shrinkage has no tuning parameter.
    - The truncated pseudo-inverse and the full inverse are reported as sensitivity analyses, and decisions use the shrinkage estimate. This replaces revision 7's pending TODO.
    - prop:wm-sensitivity and the Γ̂ paragraph state that they use the same estimate.
    - `ledoit2004wellconditioned` is added to `method_refs.bib`, with a VERIFY comment on volume, issue and pages.
14. **Revision 9 (9–10 Nov, in `method_section.tex`; no MIMIC numbers).**
    - **Intra objective.** One sentence in §wm-setting, softened to "replaces the cross-visit target with a disjoint segment of the same recording". It was then folded into the narrowing below.
    - **Encoder requirement narrowed.** It excludes objectives that treat recordings of the same patient as positives for invariance (contrastive, such as CLOCS or PCLR). It does not exclude predictive cross-visit objectives, whose predictor can carry the change.
    - **Empirical sentence.** Objectives without cross-visit pairing, in our experiments, learned recording-specific detail rather than patient structure. This is stated as an observation, not a theorem, with the details in a new appendix (`method_appendix_encoder.tex`, `sec:wm-app-encoder`; prose, no numbers).
    - **z_t sentence.** Rewritten from the 0.5b outcome: the anchors are proxies of label provenance and coding intensity, and we do not rely on them as proxies of the acute context, which the data could not establish.
    - **0.5a.** The narrowing does not change the choice (0.5a results).
    - **Recording.** `z_t` is in `S_t`. The observation model of §wm-weak and ass:wm-recording both depend on the encounter only through `z_t` and the anchor labels.
    - **Encoder selection.** The encoder is chosen on one half of the patients and the positive controls are reported on the other. The tie-break order is concatenation, then learned, then fixed features, and the fixed features remain the validation readout.
    - **Housekeeping.** The li2026dueling TODO is removed, since its claims were verified. A TODO is added to rewrite "the context that matters … is largely unrecorded" after 0.5b. The anchor appendix (`sec:wm-app-anchors`, `method_appendix_anchors.tex`) is generated from the frozen `label_sets.json`.

## 8. Deliverables

- `part1/`: data assembly (`pairs.py`, `linkage.py`, `measurements.py`, `embed_h.py`), negative-control pool and roles (`controls.py`, `roles.py`, `control_roles.json`), estimator with CRT and permutation tests (`gain.py`), recording route (`recording_route.py`, `recording_route.json`), per-code results table, interpretation-matrix figure, positive-control effect vectors (`effects_h.npz`), driver ranking (`drivers.json`), noise floor. Outputs in `outputs/gonogo_part1/`.
  - Revision 5, Phase 0 done: `label_sets.py`, `ci.py` (paired bootstrap tables), `effects.py`, `noise_floor.py` and `k59_diag.py` are added. Results are in `task1_labelsets/`, `task2_metadata/`, `task3_diag/`, `task4_gbm/`, `task5_effects/` and `task5_perm/`, plus `effects_h.npz`, `noise_floor.json` and `drivers.json`. The per-code results table is the "Phase 0 results" section; the interpretation-matrix figure is not yet drawn.
- `phase05/` (revision 5): the encoder check (candidate encoders, ψ̂² in noise-floor units, short Task 1–2 reruns), the provenance check (AUROC, constant-setting recovery) and the Γ̂ rotation (margin, codes clearing it).
  - Revision 6. Code is in `gonogo/phase05/`: `freeze_label_sets.py`, `make_anchor_appendix.py` and `provenance_auroc.py`. The single-recording objective is `scripts/pretrain.py ++objective=single`. Outputs are in `outputs/phase05/`, one folder per run: `freeze/` (the frozen `label_sets.json` and its SHA-256), `3a_provenance_auroc/` and `3b_encoder_single/`. `RECOVERY.md` records progress and resume commands. The constant-setting statistic is now the absolute reduction D_A (0.5b), and the margin is the random-effects Γ̂ (0.5c). `gonogo/method_appendix_anchors.tex` is generated from the frozen file.
- `part2/latent/`: generator with config for γ, ε, `ε_u`, `p_delay`, effect size and patient dominance; power check
- `part2/semisynthetic/`: injection script with config for `γ_mix`, driver and effect vectors
- `part2/baselines/`: baseline implementations, sweep configs and results
- `decision.md`: one page with the C1 and C2 outcomes, the key figures, and the chosen path from Section 6
