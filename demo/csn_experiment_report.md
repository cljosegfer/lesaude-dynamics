# CSN Downstream Transfer — Report & Reproduction Guide

**Repo:** lesaude-dynamics · **Scope:** finetuning the MIMIC-IV-ECG-pretrained encoder on a smaller
external dataset (Chapman-Shaoxing-Ningbo / CSN), plus the distribution-shift transfer study of
[`papel/experiments.tex`](../papel/experiments.tex)
**New files:** [`scripts/download_csn.py`](../scripts/download_csn.py),
[`scripts/build_lance_csn.py`](../scripts/build_lance_csn.py),
[`src/dataset/csn_dataset.py`](../src/dataset/csn_dataset.py),
[`src/dataset/lance_utils.py`](../src/dataset/lance_utils.py),
[`configs/data_csn.yaml`](../configs/data_csn.yaml),
[`configs/finetune_csn.yaml`](../configs/finetune_csn.yaml),
[`scripts/finetune_csn.py`](../scripts/finetune_csn.py),
[`scripts/evaluate_csn.py`](../scripts/evaluate_csn.py),
[`src/dataset/snomed_to_icd10_csn.csv`](../src/dataset/snomed_to_icd10_csn.csv),
[`src/dataset/icd10_crosswalk.py`](../src/dataset/icd10_crosswalk.py),
[`configs/distshift.yaml`](../configs/distshift.yaml),
[`scripts/evaluate_distshift.py`](../scripts/evaluate_distshift.py)

## TL;DR

Took the MIMIC-IV-ECG pretrained encoder and built a complete second pipeline — download, Lance
conversion, dataset class, finetuning, evaluation, and a cross-label-space distribution-shift test —
for CSN (PhysioNet's `ecg-arrhythmia` database, 45,152 records). Every stage mirrors an existing
MIMIC-side pattern (`scripts/build_lance.py`, `src/dataset/dataset.py`, `scripts/finetune.py`,
`scripts/evaluate.py`) closely enough that adding a **third** dataset should mostly be find-and-replace
plus the dataset-specific judgment calls flagged below. The single most important correctness detail,
easy to miss by only reading `scripts/build_lance.py`: **MIMIC's Lance waveforms are already
per-lead z-score normalized** at a raw-WFDB-to-HDF5 stage that predates the Lance pipeline
([`demo/preprocess_waveforms0.py`](preprocess_waveforms0.py)) — `build_lance.py` itself just repackages
already-normalized HDF5 into Lance. Any new dataset has to replicate that normalization itself, or the
pretrained encoder sees an out-of-distribution input scale and transfer quietly degrades.

---

## 1. Dataset identification

CSN = **Chapman-Shaoxing-Ningbo**, distributed by PhysioNet as project slug `ecg-arrhythmia`,
version `1.0.0` (Zheng et al.). Confirmed via the PhysioNet content page before writing any code:

- 45,152 records, 12-lead, 500 Hz, WFDB format (`.hea` + `.mat` pairs)
- Directory layout: `WFDBRecords/<2-digit>/<3-digit>/<record>.{hea,mat}`, ~100 records per leaf folder
- Diagnoses: SNOMED-CT codes in the header's `#Dx:` comment line (multi-label, comma-separated)
- Total size: 5.1 GB uncompressed, 2.3 GB as a single zip
- A `ConditionNames_SNOMED-CT.csv` reference file ships with the raw data, giving human-readable
  names for a subset (63/94) of the codes actually present

**Lesson for a new dataset:** confirm the *exact* PhysioNet slug/version and file layout by fetching
the content page first (`https://physionet.org/content/<slug>/`), rather than guessing — the bulk zip
endpoint (`https://physionet.org/content/<slug>/get-zip/<version>/`) and the AWS S3 mirror
(`s3://physionet-open/<slug>/<version>/`, `--no-sign-request`) are usually both available for
open-access (non-credentialed) projects and are far faster than per-record downloads.

## 2. Download — [`scripts/download_csn.py`](../scripts/download_csn.py)

Two modes:

- **`zip`** (default): streams the bulk zip, extracts it. One HTTP request instead of ~90k.
  Resumable (`Range` header, mirrors `wget -c`).
- **`records --limit N`**: walks the two-level `RECORDS` index files directly via `requests`
  (`list_records()`) and fetches individual `.hea`/`.mat` files — useful for a quick smoke test
  before committing to the full pull.

**Gotcha found and worked around:** `wfdb.dl_database()` (the obvious first choice, since `wfdb` is
already a project dependency) mis-handles a versioned `db_dir` string — passing `"ecg-arrhythmia/1.0.0"`
makes it internally re-append the version, requesting `.../ecg-arrhythmia/1.0.0/1.0.0/` and 404ing.
Confirmed against `wfdb==4.3.1` by reading its source
(`wfdb.io.record.dl_database`). Also, the dataset's top-level `RECORDS` file only lists the ~452
*subfolders*, not the 45,152 leaf records — each subfolder has its own nested `RECORDS` file. Rather
than fight the library's version-handling bug, `download_csn.py` talks to PhysioNet directly via
`requests` for both listing and per-record fetching, and only uses `wfdb` itself later, for reading
signals (where it works fine).

`--verify` spot-checks downloaded files' SHA-256 against PhysioNet's `SHA256SUMS.txt`.

**Reproduction checklist for a new dataset:** prefer the bulk-zip approach if the dataset is under a
few GB and has one; don't assume a WFDB-family library's convenience downloader handles the target
database's directory structure/versioning correctly — verify with a small `--limit`-style smoke test
against the *live* server before trusting it at scale.

## 3. Full-corpus header scan (pre-flight, not a committed script)

Before writing the converter, every `.hea` file was scanned (first line + `#Dx:` line only — no
`.mat` signal loading, so this is fast) to check assumptions that `build_lance_csn.py` would otherwise
silently get wrong:

| Check | Result |
|---|---|
| Sample length | 45,151 / 45,152 records exactly 5000 samples (10 s @ 500 Hz) |
| Lead order | 45,150 / 45,152 in the canonical `I,II,III,aVR,aVL,aVF,V1-V6` order — matches MIMIC's convention exactly, no reordering needed in practice (still implemented defensively) |
| Distinct SNOMED Dx codes | 94 |
| Malformed records | 1 (`JS01052` — a genuinely corrupted header line in the source data, confirmed by inspecting raw bytes with `cat -A`) |

This scan caught the one malformed record *before* the full conversion run, and directly informed the
crop/pad and lead-reordering defensive logic in the build script (which, it turned out, is almost never
actually exercised for this dataset — but is cheap insurance and was already the pattern in
`demo/preprocess_waveforms0.py`).

**Reproduction checklist:** always do this pre-flight scan for a new dataset. It's minutes of work and
turns "the converter silently mis-handles some structural assumption" into "we know exactly which
`n` records the fallback path handles, and why."

## 4. Raw → Lance conversion — [`scripts/build_lance_csn.py`](../scripts/build_lance_csn.py)

Unlike MIMIC (raw WFDB → HDF5 → Lance, two separate historical stages), CSN goes **directly WFDB →
Lance** in one script — CSN is small enough (5.1 GB) that the HDF5 hop isn't needed; it was incidental
history for MIMIC, not a requirement of the Lance format.

Per-record pipeline (`read_waveform`):

1. `wfdb.rdrecord(...).p_signal` — physical units (mV), **not** raw ADC counts
2. Reorder channels to the canonical lead order (via `sig_name` lookup, not positional assumption)
3. Truncate to the **first** 5000 samples if longer, zero-pad the **end** if shorter (matches
   `demo/preprocess_waveforms0.py`'s policy exactly — it explicitly takes the first 10 s, not a center
   crop)
4. **`robust_z_score`**: per-lead z-score, `NaN`s zeroed first, flat leads (`std < 1e-6`) left at 0 —
   copied verbatim from the recovered MIMIC preprocessing script. This is the step that makes CSN's
   input distribution match what the pretrained encoder was actually trained on.
5. Cast to `float16`

Two-pass structure for memory efficiency: `scan_headers()` first does a cheap header-only pass (builds
the Dx vocabulary + skips malformed records without touching `.mat` files), then `build_lance()`
streams through in `BATCH_SIZE=2000` chunks, reading signals and writing to Lance batch-by-batch — never
holding all 45k waveforms in memory at once (unlike an earlier draft of this script, which did, and
was reworked specifically to avoid a ~5-6 GB peak-RAM two-pass-in-memory design).

**Schema** (`_make_schema`, mirrors `scripts/build_lance.py`'s `_make_schema` but for CSN's fields):

```python
pa.schema([
    pa.field("record_id", pa.string()),
    pa.field("fold", pa.int8()),
    pa.field("dx_raw", pa.list_(pa.string())),
    pa.field("dx", pa.list_(pa.int8(), n_classes)),          # n_classes = 94, not hardcoded
    pa.field("waveform", pa.list_(pa.float16(), 5000 * 12)),
])
```

**Label vocabulary:** the full raw SNOMED code vocabulary actually observed in the data (94 codes,
`sorted({c for codes in dx_raw_list for c in codes})`) — mirrors how MIMIC's 76-code ICD vocabulary was
built from `icd_raw`, rather than restricting to some curated subset up front. `ICDVocabulary`
(`src/dataset/icd.py`) is dataset-agnostic (just a JSON code list ↔ index map) and was reused as-is for
the pattern, though CSN's vocabulary file (`dx_vocabulary_csn.json`) is loaded directly by
`json.load()` in the finetune/evaluate scripts rather than through that class.

**Folds:** CSN is one ECG per patient (no longitudinal stays, unlike MIMIC's ICU visits), so a simple
seeded per-record permutation into 20 folds is sufficient — `fold <= 17` train / `== 18` val / `== 19`
test, the exact same convention `MIMICLanceDataset` uses, so `CSNLanceDataset`'s split logic could be
copied verbatim.

**Actual run output:**

```
Valid records: 45,150  (skipped 2)
  SKIP JS01052.hea: time data '/' does not match format '%d/%m/%Y'   (corrupted header line)
  SKIP JS23074.hea: missing leads: {'I'}                              (found only by wfdb's own header parsing)
Vocabulary: 94 SNOMED-CT Dx codes
Lance dataset size on disk: 5.42 GB
Splits: train=40,636  val=2,257  test=2,257
```

(Note: the pre-flight scan found only 1 malformed record; the real run found 2 — `JS23074`'s missing
lead wasn't visible from a first-line-only text scan, only from `wfdb`'s actual header parser. This is
normal and expected — the pre-flight scan is a cheap sanity check, not a substitute for the build
script's own defensive `try/except`-and-skip handling.)

**Verification:** `verify()` re-reads 5 random rows from the written Lance dataset and recomputes their
waveform from the source WFDB file, asserting exact equality — same pattern as `build_lance.py`'s
`verify()`.

**Reproduction checklist for a new dataset:**
- Read signals in *physical* units, matching whatever unit convention the pretrained pipeline uses (mV
  here).
- Apply the **same normalization** the pretrained encoder's training data received. Don't assume this
  from the current `MIMICLanceDataset`/`dataset.py` code alone — its z-score block is *commented out*
  precisely because normalization already happened upstream in a script (`demo/preprocess_waveforms0.py`)
  that isn't part of the main pipeline. If a similar "hidden" preprocessing step exists for whatever
  pretrained checkpoint you're transferring from, find it before assuming raw-signal input is correct.
- Build the label vocabulary from what's actually observed in the data, not an assumed/external list.
- Reuse the `fold <= 17 / == 18 / == 19` split convention if you want existing dataset-class logic to
  be a drop-in copy.
- Go directly to Lance (skip an HDF5 intermediate) unless the dataset is large enough that a separate
  raw-extraction stage genuinely earns its keep.

## 5. Dataset class — [`src/dataset/csn_dataset.py`](../src/dataset/csn_dataset.py)

`CSNLanceDataset` mirrors `MIMICLanceDataset` (`src/dataset/dataset.py`) but is simpler — CSN has no
`triage`/`monitoring`/`pair` distinction, just one row per patient. Same contract:
`{"waveform": FloatTensor(12, 5000), "label": FloatTensor(N,)}`. Same fork-safety pattern (`lance` is
not fork-safe — the dataset handle is opened lazily per DataLoader worker in `_get_ds()`, and callers
must use `DataLoader(..., multiprocessing_context="spawn")`). Same batched `__getitems__` zero-copy
decode path for I/O efficiency under PyTorch ≥ 2.0's batch-fetch protocol.

Kept in a **new file** rather than added as a fourth mode to `MIMICLanceDataset`, since that class is
already handling three modes and MIMIC-specific columns (`ecg_no_within_stay`, `subject_id`,
`pairs_path`) that don't apply here — a separate class per dataset keeps each one legible.

## 6. Configs — [`configs/data_csn.yaml`](../configs/data_csn.yaml), [`configs/finetune_csn.yaml`](../configs/finetune_csn.yaml)

Standard Hydra `defaults: [data_csn, _self_]` composition, same shape as `configs/data.yaml` +
`configs/finetune.yaml`. Two choices worth calling out:

- **`num_classes` is never hardcoded.** `finetune_csn.py`/`evaluate_csn.py` read
  `len(json.load(open(cfg.vocabulary_path)))` at runtime. CSN's vocabulary size (94) was only known
  *after* `build_lance_csn.py` ran — for a new dataset, don't assume you know the label count before
  the conversion step actually runs.
- **`pretrained_ckpt: archive/inverse.ckpt`** — the base encoder to finetune from. This has to be an
  encoder-only checkpoint whose `backbone.*` keys can be `strict=True`-loaded into a fresh `ResNet1d`;
  it does *not* need to have been trained on a compatible label space, since only the backbone weights
  transfer (the classification head is always freshly initialized to the new dataset's class count).

`resume_ckpt` / `resume_weights_only` (`configs/finetune_csn.yaml`) and the matching stale-sidecar
cleanup in `finetune_csn.py` were added by copying the pattern from `scripts/pretrain.py` /
`scripts/inverse_pretrain.py`: on a fresh (non-resume) run, delete any leftover `wandb_resume.json` in
the CWD so the run doesn't silently reattach to a previous, unrelated wandb run ID.

## 7. Finetuning — [`scripts/finetune_csn.py`](../scripts/finetune_csn.py)

Near-identical to `scripts/finetune.py`. Checkpoint loading is the one block worth being precise about,
since it's the actual transfer mechanism:

```python
state = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
backbone_state = {
    k.removeprefix("backbone."): v
    for k, v in state["state_dict"].items()
    if k.startswith("backbone.")
}
missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
```

`strict=True` is deliberate — it means any architecture mismatch (wrong `embedding_dim`, wrong
`ResNet1d` config) fails loudly at load time rather than silently loading a partial/wrong checkpoint.
Only `backbone.*` keys are pulled from the checkpoint; the checkpoint's own `projector.*` (its
old classification head, sized for whatever label space it was trained on) is discarded, and a fresh
`nn.Linear(embedding_dim, num_classes)` is created for the new dataset's class count.
`freeze_encoder` toggles between linear-probe (`requires_grad_(False)` on the whole backbone) and
full finetune — the same flag, same script, matching the paper's own Table 1 protocol.

## 8. Evaluation — [`scripts/evaluate_csn.py`](../scripts/evaluate_csn.py)

Same bootstrap macro-AUROC (95% CI) pattern as `scripts/evaluate.py`. One real bug hit and fixed during
this work, worth flagging because it's an easy mistake to repeat: `configs/finetune_csn.yaml`'s
`evaluate.ckpt_path` briefly pointed at `archive/finetune_inverse.ckpt` — the **original MIMIC**
finetune checkpoint (76-class ICD head) — instead of the actual CSN-finetuned checkpoint (94-class
SNOMED head). Loading it produced:

```
RuntimeError: Error(s) in loading state_dict for Linear:
    size mismatch for weight: copying a param with shape torch.Size([76, 256])
    from checkpoint, the shape in current model is torch.Size([94, 256]).
```

**Diagnosis pattern:** a `Linear` shape mismatch on `projector.weight`/`projector.bias` almost always
means the checkpoint's classification head was trained for a different label space than the one you're
currently evaluating against — check `evaluate.ckpt_path` (or whatever config value supplied it) points
at a checkpoint actually produced by *this* dataset's finetuning script, not a checkpoint from another
dataset's run that happens to share a similar name in `archive/`.

## 9. Distribution-shift crosswalk — [`src/dataset/snomed_to_icd10_csn.csv`](../src/dataset/snomed_to_icd10_csn.csv), [`src/dataset/icd10_crosswalk.py`](../src/dataset/icd10_crosswalk.py)

Separate from finetuning: `papel/experiments.tex`'s "Distribution Shift" study applies a
**MIMIC-trained** classifier (76-dim ICD-10 output), **with no further finetuning**, directly to CSN
data — testing whether action-conditioned pretraining transfers more robustly across acquisition sites
than a plain supervised classifier. This requires knowing which of the MIMIC classifier's 76 output
columns have any counterpart at all in CSN's label space, since CSN uses SNOMED-CT, not ICD-10.

**Method chosen: hand-curated crosswalk, not an automated license-gated one.** The official route (NLM's
SNOMED CT → ICD-10-CM map) requires a UMLS/UTS account this environment doesn't have. Since CSN's
vocabulary is small and bounded (94 codes), a hand-built table is tractable and — importantly — more
*auditable* than an automated many-to-one map would be here, because a large fraction of CSN's labels
are pure ECG-report descriptors (ST/T changes, axis deviation, voltage criteria, interval measurements)
with **no** ICD-10 Chapter IX disease code at all (they'd fall under R94.31 "abnormal ECG", a different
ICD-10 chapter) — an automated tool would need the same kind of judgment call to exclude these, so
hand-curation isn't giving up much rigor here.

**Accuracy check performed before trusting my own domain knowledge:** cross-referenced all 94 codes
against the PhysioNet Challenge 2021 evaluation repo's official label-name mappings
(`dx_mapping_scored.csv` / `dx_mapping_unscored.csv`, which document this exact dataset among others),
rather than relying purely on memory. Combined with the bundled `ConditionNames_SNOMED-CT.csv`
(63/94 codes), every one of the 94 codes got a confirmed clinical name before assigning an ICD-10
mapping.

**Table structure** (`snomed_to_icd10_csn.csv`, one row per SNOMED code — necessary since it's a
many-to-one relationship, e.g. both "atrial fibrillation" and "atrial flutter" map to `I48`):

```
snomed_code,name,mimic_icd10,confidence,notes
164889003,atrial fibrillation,I48,high,
270492004,1st degree AV block,I44,high,I44.0
164934002,T wave abnormal,,none,ECG descriptor
```

Confidence tiers: `high` (standard, unambiguous ICD-10-CM convention) / `medium` (plausible but
debatable, e.g. generic "myocardial infarction" → acute I21 vs. old I25, and MIMIC's 76-vocab only has
I21) / `low` (genuinely uncertain catch-all placements, e.g. rare ectopic-rhythm labels dumped into
`I49` "other arrhythmias") / `none` (no Chapter IX counterpart — excluded).

**Result:** only 6 of MIMIC's 76 ICD-10 clusters are reachable at `medium`+ confidence — `I21, I44, I45,
I47, I48, I49` (7 at `low`, adding `I51` for the hypertrophy/voltage-criteria mappings). This is expected,
not a bug: CSN is an arrhythmia/conduction-focused dataset and has no counterpart at all for MIMIC's
hypertension, ischemic-disease-chronicity, or heart-failure codes.

**`icd10_crosswalk.py`** is deliberately dataset-agnostic in its API (`crosswalk_path` is a parameter,
not hardcoded) — `build_intersection(mimic_vocab, csn_vocab, min_confidence, crosswalk_path)` returns
which MIMIC columns to keep (`mimic_indices`) and which source-dataset columns to OR-reduce for each
one's ground truth (`csn_groups`); `project_csn_labels()` does that reduction. For a new dataset with
its own label vocabulary, only a new crosswalk CSV is needed — the loader code is reusable as-is (the
parameter names say `csn`/`mimic` but nothing in the logic is CSN-specific beyond the file itself).

## 10. Distribution-shift evaluation — [`scripts/evaluate_distshift.py`](../scripts/evaluate_distshift.py), [`configs/distshift.yaml`](../configs/distshift.yaml)

Loads a MIMIC-trained/finetuned checkpoint (76-dim head) via the same `_load_model` pattern as
`evaluate.py`, runs it on CSN's test split, then:

```python
xwalk      = build_intersection(mimic_vocab, csn_vocab, min_confidence=cfg.min_confidence)
preds_k    = preds_76[:, xwalk["mimic_indices"]]                       # slice model output
targets_k  = project_csn_labels(csn_labels, xwalk["csn_groups"])       # project CSN ground truth
```

Reports both a per-ICD-10-code AUROC breakdown (useful since K is small, ~6-7 classes — a macro average
alone would hide a lot) and the bootstrap macro-AUROC with 95% CI, same statistical machinery as
`evaluate.py`/`evaluate_csn.py`. `evaluate.ckpt_path` is swapped to compare model types (Dynamics model
/ naive SSL baseline / supervised classifier — the paper's three-way comparison), all against the same
frozen CSN test set and the same matched label subset.

**Config gotcha:** `configs/distshift.yaml` is deliberately **not** composed from `configs/data.yaml` +
`configs/data_csn.yaml` via Hydra `defaults`, even though both are "the same" data — both define a
`vocabulary_path` key for their own (different) label space, and composing them would silently let one
overwrite the other. Solved by giving the standalone `distshift.yaml` explicitly distinct key names
(`mimic_vocabulary_path`, `csn_vocabulary_path`, `csn_lance_path`) instead of reusing `vocabulary_path`/
`lance_path`. **Watch for this whenever a script needs two datasets' configs at once.**

## 11. Environment gotcha: `lance` needs a real compute node

`import lance` crashes with `Illegal Instruction` on this project's interactive/login node (`phocus4`
— a 2010-era Xeon with no AVX support at all), because the shared environment's compiled `pylance`/
`numpy` wheels assume AVX. Confirmed via `faulthandler` (crash is inside `lance/blob.py`'s native
extension load) and CPU flags (`grep avx /proc/cpuinfo` → empty). This is a pre-existing
environment/hardware mismatch, not a code bug — training/data-build jobs must run on the SLURM compute
nodes (`gorgona*`), same as the rest of this repo's `gorgonoid/*.sh` scripts already assume. For
quick local testing of non-Lance logic (header parsing, normalization math, the ICD-10 crosswalk) on
the login node, either mock `sys.modules["lance"]` before importing anything that touches it, or load a
target module directly via `importlib.util.spec_from_file_location` to bypass
`src/dataset/__init__.py`'s eager `from dataset.dataset import MIMICLanceDataset` (which pulls in
`lance` as a side effect of importing *any* submodule of the `dataset` package).

---

## Reproduction checklist for a new downstream dataset

1. **Identify the source precisely**: PhysioNet slug + version (or equivalent), file format, sampling
   rate, duration, lead count/order, label coding system. Fetch the actual content page rather than
   assuming.
2. **Download**: prefer a bulk archive over per-record fetching if one exists and the dataset is a few
   GB or less. Don't trust a convenience downloader's handling of the specific database's directory
   structure without a small live smoke test first.
3. **Pre-flight scan** the raw headers/metadata only (no signal loading) across the *full* corpus:
   length distribution, lead order, label vocabulary size, malformed-record count. Cheap, and it tells
   you exactly what your converter's defensive logic needs to handle.
4. **Find the pretrained encoder's actual input normalization convention** before writing the
   converter — don't infer it purely from the current dataset-class code, since normalization done
   upstream (during whatever produced the pretrained checkpoint's training data) can be invisible in
   the code path you're looking at. Replicate it exactly, or transfer quality silently suffers.
5. **Write `build_lance_<name>.py`**: read signals in physical units, reorder to the canonical lead
   order, crop/pad to the expected length, apply the matched normalization, build the label vocabulary
   from what's actually observed in the data, assign folds (reuse the `<=17/==18/==19` convention for
   compatibility), write batched to Lance, verify against source.
6. **Write `<Name>LanceDataset`**: same `{"waveform", "label"}` contract, lazy per-worker Lance handle,
   `spawn` multiprocessing context, batched `__getitems__`.
7. **Write configs**: `num_classes` read dynamically from the vocabulary file, never hardcoded.
   Namespace config keys (`<name>_vocabulary_path`, not just `vocabulary_path`) if you'll ever need to
   compose this dataset's config alongside another dataset's in the same script.
8. **Write `finetune_<name>.py`**: copy the checkpoint-loading block verbatim (`strict=True`,
   `backbone.` prefix strip, discard the checkpoint's old head), `freeze_encoder` toggle, the
   `resume_ckpt`/stale-`wandb_resume.json` pattern from `pretrain.py`/`inverse_pretrain.py`.
9. **Write `evaluate_<name>.py`**: bootstrap macro-AUROC with 95% CI. If you hit a `Linear` shape
   mismatch on `projector.*`, it's almost always a checkpoint pointed at the wrong dataset's label
   space — check the config value, not the loading code.
10. **(Optional) Distribution-shift / transfer study**: build a label crosswalk from the new dataset's
    native label system to MIMIC's 76-code ICD-10 vocabulary. Use an official crosswalk if you have
    license access to one; otherwise hand-curate (tractable for datasets with a small, bounded label
    vocabulary) and tag each row with a confidence tier so downstream consumers can choose a strictness
    threshold. Verify code names against an authoritative source before trusting your own recall of
    what a given code means. `src/dataset/icd10_crosswalk.py`'s loader is already dataset-agnostic —
    only a new crosswalk CSV is needed.
11. **Remember the AVX gotcha**: anything importing `lance` (directly or via `src/dataset/__init__.py`)
    must run on a proper SLURM compute node, not the login node.
