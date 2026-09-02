#!/usr/bin/env python3
"""
Download the PTB-XL ECG database from PhysioNet.

PTB-XL (Wagner et al., v1.0.3): 21,799 12-lead ECG records (WFDB .hea/.dat
pairs) from 18,869 patients, each shipped at both 500 Hz (records500/) and
100 Hz (records100/). This pipeline only needs the 500 Hz version (10s @
500Hz = 5000 samples, matching this repo's pretraining convention), so
records100/ is skipped by default to save bandwidth/disk.

Source: https://physionet.org/content/ptb-xl/1.0.3/
Files:  https://physionet.org/files/ptb-xl/1.0.3/  (3.0 GB uncompressed, 1.7 GB zip)

Unlike CSN/CPSC, per-record file paths are not discovered by walking nested
RECORDS index files -- ptbxl_database.csv itself lists every record's exact
`filename_hr`/`filename_lr` relative path, so that CSV is the source of
truth for both the file list here and the per-record metadata (scp_codes,
strat_fold) the Lance conversion step needs anyway.

Two download modes:
  zip     (default) -- one 1.7 GB zip covering the whole project, extracted
                       locally. records100/ members are skipped during
                       extraction (not needed by this pipeline).
  records            -- fetches ptbxl_database.csv first, then walks its
                       filename_hr column to fetch individual records500/
                       .hea/.dat pairs directly. Supports --limit for a
                       quick smoke test; also a bandwidth-conscious full-
                       download option since it never touches records100/.

Usage:
  python scripts/download_ptbxl.py
  python scripts/download_ptbxl.py --out-dir /snfs2/josefernandes/datasets/lesaude/ptbxl-monolith/raw
  python scripts/download_ptbxl.py --mode records --limit 20   # quick smoke test
  python scripts/download_ptbxl.py --verify                    # SHA256 spot-check after download
"""

import argparse
import csv
import hashlib
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZIP_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
FILES_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/ptbxl-monolith/raw"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--mode", choices=["zip", "records"], default="zip")
    p.add_argument(
        "--limit", type=int, default=None,
        help="records mode only: download only the first N records.",
    )
    p.add_argument(
        "--zip-tmp", default=None,
        help="Where to stage the downloaded zip (default: <out-dir>/../ptb-xl.zip)",
    )
    p.add_argument(
        "--keep-records100", action="store_true",
        help="Also keep the 100Hz records100/ copy (not needed by this pipeline; "
             "default is to skip it to save disk/bandwidth).",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="After download, spot-check file hashes against SHA256SUMS.txt.",
    )
    p.add_argument("--verify-samples", type=int, default=200)
    return p.parse_args()


def _download_stream(url: str, dest: Path, desc: str):
    """Streaming download with resume support (Range header, like wget -c)."""
    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if resume_from and r.status_code == 416:
            print(f"{dest.name} already fully downloaded, skipping.")
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with open(dest, mode) as f, tqdm(
            total=total, initial=resume_from, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def _find_data_root(out_dir: Path) -> Path:
    """The zip's internal layout isn't guaranteed to put records500/ directly
    under out_dir -- find wherever it actually landed."""
    for p in out_dir.rglob("records500"):
        if p.is_dir():
            return p.parent
    return out_dir


def download_zip(out_dir: Path, zip_tmp: Path | None, skip_records100: bool = True) -> Path:
    zip_path = zip_tmp or out_dir.parent / "ptb-xl.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_URL} -> {zip_path}")
    _download_stream(ZIP_URL, zip_path, desc="ptb-xl.zip")

    print(f"Extracting to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if skip_records100:
            n_before = len(members)
            members = [m for m in members if "records100/" not in m.filename]
            print(f"Skipping records100/ ({n_before - len(members)} files) -- "
                  f"not needed (this pipeline uses the 500Hz/5000-sample copy).")
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, out_dir)
    print("Extraction complete.")
    return _find_data_root(out_dir)


def fetch_metadata(out_dir: Path) -> tuple[Path, Path]:
    """Download ptbxl_database.csv and scp_statements.csv to out_dir. Needed
    both to build the records500 file list here (records mode) and later by
    build_lance_ptbxl.py for scp_codes/strat_fold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_csv = out_dir / "ptbxl_database.csv"
    scp_csv = out_dir / "scp_statements.csv"
    if not db_csv.exists():
        _download_stream(FILES_BASE + "ptbxl_database.csv", db_csv, desc="ptbxl_database.csv")
    if not scp_csv.exists():
        _download_stream(FILES_BASE + "scp_statements.csv", scp_csv, desc="scp_statements.csv")
    return db_csv, scp_csv


def list_records500(db_csv: Path, limit: int | None = None) -> list[str]:
    """Return records500/<bucket>/<ecg_id>_hr relative paths (no extension),
    read directly from ptbxl_database.csv's filename_hr column -- PTB-XL's
    metadata CSV is the authoritative file list, no nested RECORDS index
    walking needed (unlike CSN/CPSC)."""
    records: list[str] = []
    with open(db_csv, newline="") as f:
        for row in csv.DictReader(f):
            path = row["filename_hr"].strip()
            if path:
                records.append(path)
            if limit and len(records) >= limit:
                break
    return records


def download_records(records: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="Downloading records"):
        for ext in (".hea", ".dat"):
            dest = out_dir / f"{rec}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            r = requests.get(f"{FILES_BASE}{rec}{ext}", timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
    return out_dir


def download_records_mode(out_dir: Path, limit: int | None) -> Path:
    db_csv, _ = fetch_metadata(out_dir)
    print("Listing records500/ files from ptbxl_database.csv ...")
    records = list_records500(db_csv, limit=limit)
    print(f"Found {len(records):,} records" + (f" (limited to {limit})" if limit else ""))
    download_records(records, out_dir)
    return out_dir


def verify(data_root: Path, n_samples: int, skip_records100: bool = True):
    sums_path = data_root / "SHA256SUMS.txt"
    if not sums_path.exists():
        print(f"Downloading {FILES_BASE}SHA256SUMS.txt for verification ...")
        _download_stream(FILES_BASE + "SHA256SUMS.txt", sums_path, desc="SHA256SUMS.txt")

    entries = []
    for line in sums_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected_hash, rel_path = parts[0], parts[1].strip()
            if skip_records100 and "records100/" in rel_path:
                continue
            entries.append((expected_hash, rel_path))

    sample = random.sample(entries, min(n_samples, len(entries)))
    n_ok = n_missing = n_bad = 0
    for expected_hash, rel_path in tqdm(sample, desc="Verifying"):
        f = data_root / rel_path
        if not f.exists():
            n_missing += 1
            continue
        actual_hash = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  MISMATCH: {rel_path}")
    print(f"Verified {len(sample)} sampled files: {n_ok} OK, {n_missing} missing, {n_bad} mismatched")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    skip_records100 = not args.keep_records100

    if args.mode == "zip":
        data_root = download_zip(
            out_dir, Path(args.zip_tmp) if args.zip_tmp else None, skip_records100=skip_records100
        )
    else:
        data_root = download_records_mode(out_dir, args.limit)

    n_headers = len(list(out_dir.rglob("*.hea")))
    n_dats = len(list(out_dir.rglob("*.dat")))
    print(f"Done. {n_headers:,} .hea files, {n_dats:,} .dat files under {out_dir}")
    print(f"Data root (contains records500/): {data_root}")

    if args.verify:
        verify(data_root, args.verify_samples, skip_records100=skip_records100)


if __name__ == "__main__":
    main()
