# PTB-XL Downstream Transfer — Reproduction Plan

## Context

Two downstream-transfer pipelines already exist for the MIMIC-IV-ECG-pretrained encoder:
CSN (`demo/csn_experiment_report.md`, fully done including a distribution-shift crosswalk) and
CPSC2018 (`plan.md`, implementation files present but finetuning/evaluation not yet run). Both
fill the `%TODO: confirm CSN and list any further datasets to include here` gap in
`papel/experiments.tex`'s `tab:classification` (currently `Method | MIMIC | CSN`) and
`tab:distshift` tables. PTB-XL is the third dataset needed to extend this line of experiments.

Unlike CSN/CPSC, PTB-XL is not SNOMED-CT-coded and does not need CPSC's sliding-window
segmentation (its records are already a fixed 10s @ 500Hz, like CSN/MIMIC). It does have its own
official stratified 10-fold split, which should be used directly instead of reinventing a seeded
permutation. Per user decision, the label vocabulary will be built from **raw observed SCP-ECG
codes** (~71 classes), matching the CSN/CPSC convention of "vocabulary = everything actually
observed in the data," rather than the field-standard 5-class diagnostic-superclass reduction —
this keeps the pipeline mechanically consistent with the other two datasets even though it's a
larger/sparser label space than most published PTB-XL benchmarks use.

## Confirmed PTB-XL facts (via PhysioNet content page)

- Standalone PhysioNet project (**not** the older bundled copy inside `challenge-2020`, which
  lacks the same metadata/version fixes): slug `ptb-xl`, version **1.0.3**, CC BY 4.0 (open
  access, no credentialing), DOI `10.13026/kfzx-aw45`.
- 21,799 records, 18,869 patients, WFDB format (`.hea`/`.dat`, not `.mat`).
- Two sampling-rate copies of every record: `records500/` (500 Hz, matches this repo's 5000-sample
  convention — **use this**) and `records100/` (100 Hz, ignore).
- Directory layout: `records500/<thousands-bucket>/<ecg_id>_hr.{hea,dat}`, e.g.
  `records500/00000/00001_hr.dat`.
- 10 s/record, 12-lead, standard `I,II,III,aVR,aVL,aVF,V1-V6` order — same shape as CSN, no
  windowing needed (unlike CPSC).
- Labels live in `ptbxl_database.csv` (`ecg_id`, `filename_hr`, `scp_codes` — a dict-like
  `code:likelihood` string, `strat_fold` 1–10, plus demographics/signal-quality flags), joined
  with `scp_statements.csv` (code → diagnostic superclass/subclass hierarchy, AHA/DICOM
  cross-references). Not ICD-10/SNOMED-CT — a new crosswalk is needed for distshift.
- **Official recommended split** (PhysioNet page, matches Wagner et al. and the
  `ptbxl-benchmarking` convention): `strat_fold` 1–8 train, 9 val, 10 test. Use this directly.
- Bulk download: zip (1.7 GB) and S3 mirror (`s3://physionet-open/ptb-xl/1.0.3/`,
  `--no-sign-request`) both available; `SHA256SUMS.txt` provided for verification.
- Repo grep: only one incidental mention of `ptb-xl` in `scripts/download_cpsc.py`'s header
  comment (noting it's a sibling database inside `challenge-2020` that script deliberately
  ignores) — no existing PTB-XL code, and no mention in `papel/*.tex`.

## Design decisions carried over / diverging from CSN & CPSC

- **Reuse CSN's one-row-per-record shape**, not CPSC's windowing — PTB-XL records are uniformly
  10 s @ 500 Hz (5000 samples), so `record_id`/`window_idx` machinery and window→record
  mean-pooling at eval time are unnecessary.
- **Reuse the official `strat_fold` column as the fold field directly** (values 1–10) instead of a
  seeded 20-way permutation. `PTBXLLanceDataset`'s split logic becomes `fold <= 8` train,
  `fold == 9` val, `fold == 10` test — different thresholds from CSN/CPSC's `<=17/==18/==19`, so
  this one piece of `csn_dataset.py`/`cpsc_dataset.py` cannot be copied verbatim, everything else
  (fork-safety lazy `_get_ds()`, `spawn` context, batched `__getitems__`, `{"waveform","label"}`
  contract) can.
- **Label parsing differs structurally**: CSN/CPSC parse a `#Dx:`/`Dx:` comment line out of the
  WFDB header itself. PTB-XL's labels live in `ptbxl_database.csv`'s `scp_codes` column (a
  Python-dict-literal string, e.g. `"{'NORM': 100.0, 'SBRAD': 0.0}"`, parsed via
  `ast.literal_eval`), read once as a table, not per-header. Any code present in the dict counts
  as positive regardless of its likelihood value (0 conventionally means "asserted without a
  likelihood value" in PTB-XL's own documentation, not "absent") — mirrors CSN/CPSC's "any listed
  code = positive" treatment, no threshold filtering.
- **Same normalization** (`robust_z_score`, per-lead, NaN-safe, flat-lead guard) and **same
  physical-units WFDB read + lead reorder-by-name** as CSN/CPSC — this is the one thing that must
  not drift, it's what keeps the input distribution matched to the pretrained encoder.

## Files to create (mirroring the CSN set 1:1, minus windowing)

1. **`scripts/download_ptbxl.py`** — adapt `scripts/download_csn.py`'s `zip` mode (PTB-XL has a
   bulk zip like CSN, unlike CPSC which needed subfolder-scoped fetching), pointed at
   `physionet.org/content/ptb-xl/1.0.3/` or the S3 mirror. `--verify` checks `SHA256SUMS.txt`.
   Include a small `--limit`-style smoke-test mode before trusting the full pull, per the CSN
   report's standing advice not to assume a convenience downloader handles a new database
   correctly without checking live first.

2. **Pre-flight scan** (`demo/ptbxl_preflight_scan.py`, ad-hoc, not committed to the main
   pipeline) — before writing the converter:
   - Load `ptbxl_database.csv`/`scp_statements.csv`, confirm exact column names and `strat_fold`
     distribution (should be roughly even across 1–10).
   - Confirm sample length is uniformly 5000 (10s @ 500Hz) across all 21,799 `records500/` files
     — flag any exceptions.
   - Lead order/naming check (expect near-100% canonical match, as with CSN/CPSC).
   - Distinct raw SCP code count actually observed (expect ~71, cross-check against
     `scp_statements.csv`'s full code list — not every documented code need actually appear).
   - Malformed record count.
   - Document results in `demo/ptbxl_experiment_report.md` at the end, same table format CSN used.

3. **`scripts/build_lance_ptbxl.py`** — adapt `scripts/build_lance_csn.py` (WFDB → Lance directly,
   no HDF5 hop, no windowing). Key differences from CSN's converter:
   - Metadata-driven, not header-driven: read `ptbxl_database.csv` once, build a
     `{ecg_id: (filename_hr, scp_codes_dict, strat_fold)}` table; `scan_headers()`'s CSN role
     (build vocabulary from what's observed, without loading signals) becomes an `ast.literal_eval`
     pass over `scp_codes` strings.
   - `read_waveform`: `wfdb.rdrecord(records500/<bucket>/<ecg_id>_hr).p_signal` (physical units,
     mV), reorder to canonical lead order via `sig_name` (defensive, expected no-op), crop/pad to
     5000 (defensive, expected no-op since PTB-XL is already uniform), `robust_z_score`, cast
     `float16` — reuse verbatim from `build_lance_csn.py`.
   - Schema (swap CSN's `dx`/`dx_raw` names for `scp`/`scp_raw`, drop CPSC's `window_idx`):
     ```python
     pa.schema([
         pa.field("record_id", pa.string()),          # ecg_id
         pa.field("fold", pa.int8()),                  # strat_fold, 1-10, used directly
         pa.field("scp_raw", pa.list_(pa.string())),
         pa.field("scp", pa.list_(pa.int8(), n_classes)),  # n_classes = len(observed vocab), dynamic
         pa.field("waveform", pa.list_(pa.float16(), 5000 * 12)),
     ])
     ```
   - Vocabulary: `sorted({code for codes in scp_raw_list for code in codes})` over all observed
     codes, written to `dx_vocabulary_ptbxl.json` (keep the `dx_vocabulary_<name>.json` naming
     convention for config compatibility even though the field is called `scp` here).
   - Reuse `to_fixed_list_float16`/`to_fixed_list_int8` from `src/dataset/lance_utils.py` unchanged.
   - Two-pass batched structure (`BATCH_SIZE=2000`), `verify()` re-reading random rows against
     freshly-recomputed WFDB signals — same pattern as CSN/CPSC.

4. **`src/dataset/ptbxl_dataset.py`** — `PTBXLLanceDataset`, copy `src/dataset/csn_dataset.py`
   structurally, but split logic uses the official fold thresholds: `fold <= 8` train,
   `fold == 9` val, `fold == 10` test (not CSN's `<=17/==18/==19` — different scale since PTB-XL's
   `fold` column is the real `strat_fold`, 1–10, not a synthetic 20-way permutation). Same
   `{"waveform": FloatTensor(12,5000), "label": FloatTensor(N,)}` contract, same lazy
   `_get_ds()` fork-safety + `spawn` requirement, same batched `__getitems__`, same
   `train_frac`/`cache` constructor args.

5. **`configs/data_ptbxl.yaml`**, **`configs/finetune_ptbxl.yaml`** — copy CSN's configs, swap
   paths to a new `ptbxl-monolith` data directory (`/snfs2/josefernandes/datasets/lesaude/ptbxl-monolith/{raw,ptbxl.lance,dx_vocabulary_ptbxl.json}`),
   `wandb_project: lesaude-finetune-ptbxl`, `ckpt_path: checkpoints/finetune_ptbxl_{inverse,supervised}.ckpt`.
   `num_classes` stays absent, read at runtime from the vocabulary JSON length. Keep
   `resume_ckpt`/`resume_weights_only` + the stale-`wandb_resume.json` cleanup block (from
   `scripts/pretrain.py`, already ported twice into `finetune_csn.py`/`finetune_cpsc.py` — port
   again here).

6. **`scripts/finetune_ptbxl.py`** — copy `scripts/finetune_csn.py` (not `finetune_cpsc.py`, since
   there's no windowing to account for), swap `CSNLanceDataset` → `PTBXLLanceDataset`, config name
   to `finetune_ptbxl`. Keep the checkpoint-loading block byte-for-byte: `strict=True` backbone-only
   load (`k.removeprefix("backbone.")`, filtered to `backbone.*` keys), discard the checkpoint's
   old `projector.*`, fresh `nn.Linear(embedding_dim, num_classes)`, `freeze_encoder` toggle for
   linear-probe vs full finetune.

7. **`scripts/evaluate_ptbxl.py`** — copy `scripts/evaluate_csn.py` (no window-to-record
   aggregation needed, unlike `evaluate_cpsc.py`). Same bootstrap macro-AUROC (95% CI, 1000
   iterations) pattern as `evaluate.py`/`evaluate_csn.py`/`evaluate_cpsc.py`. Double-check
   `configs/finetune_ptbxl.yaml`'s `evaluate.ckpt_path` points at a checkpoint actually produced by
   *this* dataset's finetuning run — both prior reports flag a `Linear` shape-mismatch on
   `projector.*` as the easy repeat mistake (checkpoint from the wrong dataset's label space).

8. **`src/dataset/scp_to_icd10_ptbxl.csv`** — new hand-curated crosswalk, same structure as
   `snomed_to_icd10_csn.csv` (`scp_code,name,mimic_icd10,confidence,notes`). Source material:
   `scp_statements.csv`'s own diagnostic-class hierarchy and AHA/DICOM code cross-references, plus
   cross-checking against the existing `snomed_to_icd10_csn.csv` for consistency on overlapping
   conditions (AFib → I48, LBBB/RBBB → I44/I45, etc.). PTB-XL's strength is MI/ischemia coverage
   (its NORM/MI/STTC/CD/HYP superclass structure gives strong representation for `I21`-family
   codes), which is a real gap in CSN's crosswalk (CSN's report noted only 6-7 of MIMIC's 76
   ICD-10 clusters were reachable, with no ischemic-chronicity coverage) — PTB-XL's distshift table
   may fill in codes CSN's couldn't reach. No code changes needed to
   `src/dataset/icd10_crosswalk.py` — confirmed dataset-agnostic (`build_intersection`/
   `project_csn_labels` take `crosswalk_path`/vocab lists as parameters; the `csn_*` naming is
   historical, not enforced).

9. **`configs/distshift_ptbxl.yaml`** — copy `configs/distshift.yaml`'s structure, namespaced keys
   (`ptbxl_lance_path`, `ptbxl_vocabulary_path`, keep `mimic_vocabulary_path` as-is), deliberately
   standalone rather than Hydra-composed, for the same `vocabulary_path` key-collision reason
   documented in the CSN version.

10. **`scripts/evaluate_distshift_ptbxl.py`** — copy `scripts/evaluate_distshift.py`, swap
    `CSNLanceDataset` → `PTBXLLanceDataset`, crosswalk path, config name to `distshift_ptbxl`.

11. **`gorgonoid/finetune_ptbxl.sh`** — copy `gorgonoid/finetune_csn.sh` (the only existing
    per-dataset SLURM script; CPSC doesn't have one yet either), swap job name and the final
    `python scripts/finetune_ptbxl.py` call.

12. **`demo/ptbxl_experiment_report.md`** — written last, mirroring `csn_experiment_report.md`'s
    structure (dataset identification → download → pre-flight scan → Lance conversion → dataset
    class → configs → finetuning → evaluation → distshift crosswalk → distshift evaluation →
    environment gotchas), documenting actual run output (valid/skipped record counts, vocabulary
    size, split sizes, disk size).

## Non-goals / carried-over constraints

- No windowing/`window_idx` — PTB-XL doesn't need CPSC's variable-length handling.
- `num_classes` never hardcoded — always read from the vocabulary JSON's length after
  `build_lance_ptbxl.py` actually runs.
- All `import lance`-touching work (build script, dataset class, interactive testing) must run on
  a SLURM compute node (`gorgona*`), not the login node (`phocus4`) — same AVX gotcha as CSN/CPSC.
- `papel/experiments.tex`'s `tab:classification`/`tab:distshift` tables and the `%TODO` markers are
  **not** edited as part of this plan — that's a follow-up once real PTB-XL numbers exist, same as
  how CPSC's plan didn't touch the paper either.

## Verification

1. `download_ptbxl.py --verify` confirms SHA-256 checksums against `SHA256SUMS.txt`.
2. Pre-flight scan output reviewed manually (length uniformity, lead order, code count, malformed
   count, `strat_fold` distribution) before writing the converter — gate on this, same as CSN/CPSC.
3. `build_lance_ptbxl.py`'s own `verify()` (random rows re-derived from source WFDB, exact equality
   against Lance) must pass; check printed valid/skipped counts and per-fold split sizes.
4. Smoke-test `PTBXLLanceDataset` on a compute node: instantiate for each split, pull one batch via
   `DataLoader(..., multiprocessing_context="spawn")`, confirm shapes `(B,12,5000)`/`(B,n_classes)`.
5. Run `scripts/finetune_ptbxl.py` for a few epochs (or `++max_epochs=1` smoke run) on a compute
   node via `gorgonoid/finetune_ptbxl.sh`, confirm loss decreases and a checkpoint is written.
6. Run `scripts/evaluate_ptbxl.py` against that checkpoint, confirm no `Linear` shape-mismatch and
   a sane macro-AUROC (95% CI) prints.
7. Run `scripts/evaluate_distshift_ptbxl.py` against an existing MIMIC-trained checkpoint, confirm
   the crosswalk matches a non-empty set of ICD-10 clusters and per-code + macro AUROC print.
