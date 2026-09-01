# CPSC2018 Downstream Transfer — Reproduction Plan

## Context

`demo/csn_experiment_report.md` documents the first downstream-transfer pipeline built for this
project: finetuning the MIMIC-IV-ECG-pretrained encoder on CSN (Chapman-Shaoxing-Ningbo), plus a
zero-shot distribution-shift study against a MIMIC-trained classifier. `papel/experiments.tex`
carries a literal `%TODO: confirm CSN and list any further datasets to include here` at three call
sites — CPSC2018 is the next dataset needed to fill that gap in both the paper's classification
table (`tab:classification`: Dynamics / Inverse Dynamics / Supervised / SSL-baseline, each
finetuned on the dataset's own label space) and its distribution-shift table (`tab:distshift`:
same four MIMIC-trained models applied zero-shot via an ICD-10 crosswalk).

The goal here is to mirror the CSN pipeline file-for-file (download → pre-flight scan → Lance
build → dataset class → configs → finetune → evaluate → distribution-shift crosswalk + eval),
adapted for CPSC2018's specific structural differences from CSN. This is greenfield work — a repo
grep confirmed zero existing references to "cpsc" anywhere.

**Key difference from CSN that must not be glossed over:** CPSC2018 is *not* a standalone
PhysioNet project. It is the `training/cpsc_2018/` subfolder inside the much larger
`challenge-2020` project (v1.0.2, 7.5 GB total across 6 source databases: cpsc_2018,
cpsc_2018_extra, st_petersburg_incart, ptb, ptb-xl, georgia). Per user decision, download must
target only that subfolder (not the full bulk zip), must exclude `cpsc_2018_extra` (the "unused
data" subset, not part of the standard 6,877-record CPSC2018 benchmark), and the plan does include
the distribution-shift crosswalk step.

## Confirmed CPSC2018 facts (via PhysioNet content pages)

- Source: `training/cpsc_2018/` inside `physionet.org/content/challenge-2020/1.0.2/`
- 6,877 records, 12-lead, 500 Hz, WFDB format (`.hea` + `.mat` pairs, same as CSN)
- Duration **varies 6–60 s** (not a fixed 10 s like CSN/MIMIC) — this is the biggest structural
  difference from CSN and needs its own pre-flight histogram, not just a malformed-record count
- Subfolder layout: flat `g1`...`g7`, up to 1000 records each (simpler than CSN's two-level
  `<2-digit>/<3-digit>` nesting)
- Labels: `#Dx:` header line, SNOMED-CT codes, multi-label — same parsing logic as CSN
- Lead order: standard `I,II,III,aVR,aVL,aVF,V1-V6` (per PhysioNet's documented header example)
- Download: `wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/`
  or the S3 mirror (`s3://physionet-open/challenge-2020/1.0.2/training/cpsc_2018/`,
  `--no-sign-request`) — both scoped to the one subfolder, avoiding the other 5 databases
- SHA256SUMS.txt exists at the `challenge-2020` project root (covers all sources) for verification

## Pre-flight scan results (confirmed, `demo/cpsc_preflight_scan.py`)

- 6,877/6,877 headers parsed, 0 malformed
- Sampling frequency: 500 Hz uniform
- Sample length: min=3,000, max=72,000, mean=7,974, median=6,000 (target crop length 5,000).
  64.7% of records (4,451) exceed 5,000 samples; only 0.1% (10) are shorter. See the truncation
  policy decision in step 3 below.
- Lead order: 6,877/6,877 exact canonical `I,II,III,aVR,aVL,aVF,V1-V6` match — no reordering logic
  will actually trigger, same as CSN
- Distinct SNOMED Dx codes: **9** — matches CPSC2018's well-known official 9-class structure
  (top codes by frequency: 59118001 RBBB, 164889003 AF, 426783006 NORM, 429622005 STD, 270492004
  I-AVB, 164884008 STE, 284470004 PAC, 164909002 LBBB, 164931005 PVC — confirm exact
  code↔abbreviation mapping against `Dx_map.csv`/`dx_mapping_scored.csv` when building the ICD-10
  crosswalk in step 8). A much smaller, more tractable label space than CSN's 94 codes.
- **Gotcha found:** header Dx comment lines use `"# Dx:"` (space after `#`), not the `"#Dx:"`
  PhysioNet's documentation page illustrates — string matching must key off `"Dx:"` substring
  within a `#`-prefixed line, not a fixed `"#Dx:"` prefix. Cost the first scan run a false
  "0 distinct Dx codes" result before being caught and fixed.

## Files to create (mirroring the CSN set 1:1)

1. **`scripts/download_cpsc.py`** — adapt `scripts/download_csn.py`. Key differences:
   - No `zip`/`get-zip` mode targeting the whole dataset (that would pull 7.5 GB for 6 databases).
     Default mode instead recursively fetches only `training/cpsc_2018/` via `requests` (mirroring
     the existing `records`-mode listing/fetching logic in `download_csn.py`, since there's no
     single-subfolder zip endpoint), or shells out to scoped `wget -r -c -np`.
   - `list_records()` adapts to the flat `g1..g7` layout (no two-level nesting) — likely each `gN`
     folder has its own `RECORDS` index file to walk, same pattern as CSN's per-subfolder `RECORDS`.
   - `--verify` still spot-checks against `SHA256SUMS.txt`, but that file covers the whole
     `challenge-2020` project — filter to the subset of lines under `training/cpsc_2018/`.
   - Explicitly do **not** fetch `cpsc_2018_extra/`.

2. **Pre-flight header scan** (not committed, ad-hoc like CSN's) — before writing the converter:
   - Sample-length distribution (critical here — unlike CSN's near-uniform 5000, CPSC ranges up to
     30,000 samples; need to know what fraction of records get truncated by the first-10s-crop
     policy and whether that's an acceptable information loss, same policy MIMIC/CSN both use)
   - Lead order / naming check
   - Distinct SNOMED Dx code count
   - Malformed record count
   - Document results the same way CSN's report table does (this becomes part of a
     `demo/cpsc_experiment_report.md` written at the end, mirroring `csn_experiment_report.md`)

3. **`scripts/build_lance_cpsc.py`** — adapt `scripts/build_lance_csn.py` (WFDB → Lance, no HDF5
   hop). Reuse verbatim: `robust_z_score`, two-pass `scan_headers()`/`build_lance()` structure,
   `BATCH_SIZE=2000`, `_make_schema()` analog (swap `record_id`/`dx`/`dx_raw` field names, dynamic
   `n_classes`), `verify()` re-reading random rows from source WFDB. Import
   `to_fixed_list_float16`/`to_fixed_list_int8` from `src/dataset/lance_utils.py` unchanged.
   Vocabulary from `sorted(set of all observed Dx codes)`, written to `dx_vocabulary_cpsc.json`.

   **Gotcha carried over from the pre-flight scan:** this mirror's header comments use `"# Dx:"`
   (space after `#`), not the `"#Dx:"` PhysioNet's own docs page illustrates — `parse_dx()` must
   match on `"Dx:"` substring within a `#`-prefixed line, not a fixed `"#Dx:"` prefix.

   **Truncation policy — sliding-window segmentation (decided after the pre-flight scan):** CPSC
   records vary 3,000–72,000 samples (6–144s); a plain first-10s crop (CSN/MIMIC's policy) would
   discard a mean 37% (up to 93%) of signal on 64.7% of records. Instead, each record is split into
   `ceil(n_samples / 5000)` non-overlapping 5000-sample windows (zero-pad only the final partial
   window), every window becoming its own Lance row carrying the parent record's `dx`/`dx_raw`
   labels. Sample-length distribution: p50=6000 (2 windows), p90=14000 (3 windows), p99=28328 (6
   windows) — expect the ~6,877 records to expand to a noticeably larger row count.
   - **Fold assignment is per-record, not per-window** — compute the `fold<=17/==18/==19` split
     over the record IDs first, then propagate each record's fold to all of its windows. Getting
     this backwards (assigning folds per-window) would leak windows from the same recording across
     train/val/test.
   - Add a `window_idx` (and keep `record_id`) column so windows from one record are traceable and
     re-aggregatable later.
   - **Deferred to the evaluate_cpsc.py step:** whether to report window-level AUROC directly, or
     mean-pool a record's window predictions into one record-level score first (more comparable to
     CSN/MIMIC's one-score-per-patient convention). Revisit once the finetuning loop is in place.

4. **`src/dataset/cpsc_dataset.py`** — `CPSCLanceDataset`, copy `src/dataset/csn_dataset.py`
   verbatim except class name and any CSN-specific docstrings/comments. Same contract
   (`{"waveform": FloatTensor(12,5000), "label": FloatTensor(N,)}`), same lazy per-worker
   `_get_ds()` fork-safety pattern, same batched `__getitems__` path, same `fold<=17/==18/==19`
   split logic, same `train_frac`/`cache` constructor args.

5. **`configs/data_cpsc.yaml`** and **`configs/finetune_cpsc.yaml`** — copy CSN's configs,
   swap paths to a new `cpsc-monolith` data directory (mirroring `csn-monolith`'s layout), swap
   `wandb_project: lesaude-finetune-cpsc`, `ckpt_path: checkpoints/finetune_cpsc_supervised.ckpt`
   (or `_inverse`, one run per method as in CSN). `num_classes` stays absent from config — read at
   runtime from `len(json.load(open(cfg.vocabulary_path)))`. Keep `resume_ckpt`/
   `resume_weights_only` + stale-`wandb_resume.json`-cleanup pattern (copied from
   `scripts/pretrain.py:207-226` / `scripts/inverse_pretrain.py:237-256`, already ported once into
   `finetune_csn.py` — same port again here).

6. **`scripts/finetune_cpsc.py`** — copy `scripts/finetune_csn.py`, swap `CSNLanceDataset` →
   `CPSCLanceDataset`, config name to `finetune_cpsc`. Keep the checkpoint-loading block
   byte-for-byte (`strict=True` backbone-only load, discard old `projector.*`, fresh
   `nn.Linear(embedding_dim, num_classes)`, `freeze_encoder` toggle for linear-probe vs full
   finetune per the paper's Table 1 protocol).

7. **`scripts/evaluate_cpsc.py`** — copy `scripts/evaluate_csn.py`, swap dataset class and config
   name. Same bootstrap macro-AUROC (95% CI) machinery. Double-check `evaluate.ckpt_path` in
   `configs/finetune_cpsc.yaml` points at a checkpoint actually produced by *this* dataset's
   finetuning run (CSN's report flags exactly this mistake as an easy repeat: a `Linear` shape
   mismatch on `projector.*` almost always means a cross-dataset checkpoint mixup).

8. **`src/dataset/snomed_to_icd10_cpsc.csv`** — new hand-curated crosswalk, same structure as
   `snomed_to_icd10_csn.csv` (`snomed_code,name,mimic_icd10,confidence,notes`). Cross-reference
   every CPSC Dx code against the PhysioNet/CinC Challenge 2021 evaluation repo's
   `dx_mapping_scored.csv`/`dx_mapping_unscored.csv` (same authoritative source CSN used — that
   repo documents CPSC2018's codes too, since it's one of the 2021 challenge's constituent
   datasets) plus the official 2020-challenge `Dx_map.csv`. Since CPSC and CSN are both
   SNOMED-coded arrhythmia datasets, expect real overlap in codes — cross-check confidence tiers
   against the existing CSN crosswalk for consistency where codes coincide. No code changes needed
   to `src/dataset/icd10_crosswalk.py` (already dataset-agnostic: `crosswalk_path` is a parameter).

9. **`configs/distshift_cpsc.yaml`** — copy `configs/distshift.yaml`, rename keys to
   `cpsc_lance_path`/`cpsc_vocabulary_path` (keep `mimic_vocabulary_path` as-is) — deliberately
   standalone, not composed via Hydra `defaults`, for the same key-collision reason documented in
   the CSN version's comments.

10. **`scripts/evaluate_distshift_cpsc.py`** — copy `scripts/evaluate_distshift.py` verbatim,
    swap `CSNLanceDataset`→`CPSCLanceDataset`, `project_csn_labels`→`project_cpsc_labels` (or keep
    the crosswalk helper name if `icd10_crosswalk.py`'s function is already generic — verify at
    implementation time), config name to `distshift_cpsc`, docstring/print strings CSN→CPSC.

11. **`gorgonoid/finetune_cpsc.sh`** — copy `gorgonoid/finetune_csn.sh`, swap job name and the
    final `python scripts/finetune_cpsc.py` call. Same for an eval-side SLURM script if one exists
    for CSN evaluation (check `gorgonoid/` for an `evaluate_csn.sh` counterpart during
    implementation and mirror it if present).

12. **`demo/cpsc_experiment_report.md`** — write this last, mirroring
    `demo/csn_experiment_report.md`'s structure (dataset identification → download → pre-flight
    scan → Lance conversion → dataset class → configs → finetuning → evaluation → distshift
    crosswalk → distshift evaluation → environment gotchas), documenting actual run output
    (valid/skipped record counts, vocabulary size, split sizes, disk size) the same way CSN's
    report does, plus specifically calling out the sample-length-distribution finding from the
    pre-flight scan since that's CPSC's distinguishing risk area versus CSN.

## Non-goals / carried-over constraints

- Reuse `robust_z_score` and the crop/pad-to-5000 policy **exactly** as CSN copied them from
  `demo/preprocess_waveforms0.py` — this is the single most important correctness detail (matching
  the pretrained encoder's input distribution), not something to re-derive.
- `num_classes` must never be hardcoded anywhere — always read dynamically from the vocabulary
  JSON's length, since it's only known after `build_lance_cpsc.py` actually runs.
- All `import lance`-touching work (the build script, dataset class, any interactive testing) must
  run on a SLURM compute node (`gorgona*`), not the login node (`phocus4`) — AVX gotcha carries
  over unchanged from CSN.
- Data directory convention: mirror `csn-monolith`'s layout, e.g.
  `/snfs2/josefernandes/datasets/lesaude/cpsc-monolith/{raw,cpsc.lance,dx_vocabulary_cpsc.json}`.

## Progress log

- Step 1 (`scripts/download_cpsc.py`) done: 6,877/6,877 records downloaded and SHA-256 verified
  (200/200 sampled). Two gotchas found and fixed during the real run: (a) a single flaky connection
  timeout aborted the whole batch — added per-file retry with backoff; (b) each subfolder's RECORDS
  index over-claims its last entry (`g<N>/A<N*1000>` is actually stored under `g<N+1>/`, confirmed
  for all 6 boundaries) — added a same-name fallback to the next subfolder on 404.
- Step 2 (`demo/cpsc_preflight_scan.py`) done — see "Pre-flight scan results" above. Also fixed a
  `"#Dx:"` vs `"# Dx:"` matching bug in the ad-hoc scan script itself (not `build_lance_cpsc.py`,
  which uses `wfdb.rdheader()` and is unaffected — wfdb normalizes comment-prefix whitespace).
- Step 3 (`scripts/build_lance_cpsc.py`) run successfully on `gorgona10`: 6,877 valid records (0
  skipped) -> 13,563 window-rows, 9-class vocabulary, verify() passed, 1.63 GB on disk, splits
  train=12,221/val=671/test=671 window-rows. Confirmed the val/test window-count tie (671=671) is
  expected, not a bug: `6877 % 20 == 17`, so folds 18 and 19 both land in the "floor" group of the
  permutation split and structurally get the same record count (343 each).
- Step 4 (`src/dataset/cpsc_dataset.py`, `CPSCLanceDataset`) done and smoke-tested on a compute
  node — matches CSN's contract exactly (`{"waveform": (12,5000), "label": (9,)}`); the
  window-vs-record row semantics are invisible to this class. Confirmed `train=12221`, `val=671`,
  `test=671` rows and correct batch shapes for all three splits.
- Step 5 (`configs/data_cpsc.yaml`, `configs/finetune_cpsc.yaml`) done — verbatim structural copies
  of CSN's configs with paths/names swapped, YAML-parse validated.
- Step 6 (`scripts/finetune_cpsc.py`) done — verbatim port of `finetune_csn.py` swapping the
  dataset class and config name. Windows are treated as independent training examples (no
  aggregation needed here — that's only an evaluation-time concern).
- Step 7 (`scripts/evaluate_cpsc.py`) done, with the deferred aggregation decision resolved:
  record-level mean-pooling of window predictions before computing bootstrap macro-AUROC (per user
  decision), rather than scoring windows independently. `record_id` is fetched directly from the
  Lance dataset via `ds.rows` (relying on `shuffle=False` preserving row order) rather than adding
  it to `CPSCLanceDataset`'s output contract.
- Finetuning (step 6) completed independently: `archive/finetune_cpsc_supervised.ckpt` and
  `archive/finetune_cpsc_inverse.ckpt` both exist on disk. `configs/finetune_cpsc.yaml` was updated
  accordingly (`pretrained_ckpt`/`ckpt_path`/`evaluate.ckpt_path` now point at the inverse-dynamics
  variant) — taken as the current deliberate state, not reverted.
- Step 8 (`src/dataset/snomed_to_icd10_cpsc.csv`) done. All 8 of CPSC2018's SNOMED codes that have
  any MIMIC counterpart are literally identical code IDs to rows already in CSN's own crosswalk
  (both datasets share PhysioNet's SNOMED taxonomy), so those rows were reused verbatim for
  consistency; only `164884008` ("ventricular ectopics") is CPSC-specific, mapped the same way as
  CSN's sibling code `427172004` ("premature ventricular contractions") -> I49/high/I49.3. Verified
  via `build_intersection()` (bypassing `src/dataset/__init__.py`'s eager `lance` import with
  `importlib.util.spec_from_file_location`, same workaround as the CSN report's environment-gotcha
  section): matches 4 ICD-10 clusters (`I44, I45, I48, I49`) at all confidence tiers (the crosswalk
  only needed "high" rows — CPSC2018's curated 9-class label space had no ambiguous cases).
- Step 9 (`configs/distshift_cpsc.yaml`) done — standalone config with `cpsc_lance_path`/
  `cpsc_vocabulary_path` keys (not composed via Hydra defaults), mirroring CSN's version.
- Step 10 (`scripts/evaluate_distshift_cpsc.py`) done — mirrors `evaluate_distshift.py`, importing
  `icd10_crosswalk.py`'s `project_csn_labels` under a local alias (`project_labels`) rather than
  renaming the shared function (used as-is by CSN's script; body is dataset-agnostic despite the
  name). Applies the same record-level mean-pooling of window predictions as `evaluate_cpsc.py`
  before computing AUROC.
- **Not yet run**: `evaluate_distshift_cpsc.py` itself (needs a compute node — `lance` + a real
  MIMIC-trained checkpoint). Steps 11 (SLURM script) and 12 (final report) still pending.
- Noticed but out of scope for this plan: `plan_ptbxl.md` now exists in the repo root (created
  2026-09-01), suggesting a fourth dataset (PTB-XL, another `challenge-2020` source database) is
  being worked on in parallel — not part of this CPSC2018 plan.

## Verification

1. `download_cpsc.py --verify` confirms SHA-256 checksums against the filtered `SHA256SUMS.txt`
   subset for `training/cpsc_2018/`.
2. Pre-flight scan output reviewed manually before writing the converter (length histogram, lead
   order, code count, malformed count) — gate on this before proceeding, same as CSN's process.
3. `build_lance_cpsc.py`'s own `verify()` (5 random rows re-derived from source WFDB, exact
   equality against Lance) must pass; check printed valid/skipped counts and split sizes.
4. Smoke-test `CPSCLanceDataset` on a compute node: instantiate for each split, pull one batch via
   `DataLoader(..., multiprocessing_context="spawn")`, confirm shapes
   `(B, 12, 5000)`/`(B, n_classes)`.
5. Run `scripts/finetune_cpsc.py` for a few epochs (or `++max_epochs=1` smoke run) on a compute
   node via `gorgonoid/finetune_cpsc.sh`, confirm loss decreases and a checkpoint is written.
6. Run `scripts/evaluate_cpsc.py` against that checkpoint, confirm no `Linear` shape-mismatch
   errors and a sane macro-AUROC (95% CI) prints.
7. Run `scripts/evaluate_distshift_cpsc.py` against an existing MIMIC-trained checkpoint (e.g.
   `archive/supervised_0.ckpt`), confirm the crosswalk matches a non-empty set of ICD-10 clusters
   and per-code + macro AUROC print without error.
