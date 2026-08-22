#!/usr/bin/env python3
"""
Download the Chapman-Shaoxing-Ningbo (CSN) ECG database from PhysioNet.

CSN = PhysioNet's "ecg-arrhythmia" database (Zheng et al., v1.0.0): 45,152
single 12-lead ECG records (WFDB .hea/.mat pairs, 500 Hz) from Chapman
University / Shaoxing People's Hospital and Ningbo First Hospital, with
SNOMED-CT diagnosis codes in the header comments.

Source: https://physionet.org/content/ecg-arrhythmia/1.0.0/
Files:  https://physionet.org/files/ecg-arrhythmia/1.0.0/  (5.1 GB uncompressed)

Records are organized as WFDBRecords/<2-digit>/<3-digit>/<record>.{hea,mat},
each subfolder holding its own RECORDS index (the top-level RECORDS file only
lists the ~452 subfolders, not the 45,152 leaf records).

Two download modes:
  zip     (default) — one 2.3 GB zip covering the whole database, extracted
                       locally. Far fewer HTTP round-trips than per-record
                       downloads; use this for the real pull.
  records            — fetches individual .hea/.mat files directly (walking
                       the nested RECORDS indexes). Slower (2 requests/record,
                       plus one per subfolder) but supports --limit, useful
                       for a quick smoke test before committing to the full
                       download.
  (Note: wfdb.dl_database() is not used here — its versioned-db_dir handling
  double-appends the version segment for this database, e.g. it requests
  .../ecg-arrhythmia/1.0.0/1.0.0/ and 404s. Confirmed against wfdb==4.3.1.)

Usage:
  python scripts/download_csn.py
  python scripts/download_csn.py --out-dir /snfs2/josefernandes/datasets/lesaude/csn-monolith/raw
  python scripts/download_csn.py --mode records --limit 20   # quick smoke test
  python scripts/download_csn.py --verify                    # SHA256 spot-check after download
"""

import argparse
import hashlib
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZIP_URL = "https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/"
FILES_BASE = "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/csn-monolith/raw"


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
        help="Where to stage the downloaded zip (default: <out-dir>/../ecg-arrhythmia.zip)",
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
    """The zip's internal layout isn't guaranteed to put WFDBRecords/ directly
    under out_dir — find wherever it actually landed."""
    for p in out_dir.rglob("WFDBRecords"):
        if p.is_dir():
            return p.parent
    return out_dir


def download_zip(out_dir: Path, zip_tmp: Path | None) -> Path:
    zip_path = zip_tmp or out_dir.parent / "ecg-arrhythmia.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_URL} -> {zip_path}")
    _download_stream(ZIP_URL, zip_path, desc="ecg-arrhythmia.zip")

    print(f"Extracting to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            zf.extract(member, out_dir)
    print("Extraction complete.")
    return _find_data_root(out_dir)


def list_records(limit: int | None = None) -> list[str]:
    """Walk the two-level RECORDS index and return record paths, e.g.
    'WFDBRecords/01/010/JS00001' (relative to FILES_BASE, no extension)."""
    top = requests.get(FILES_BASE + "RECORDS", timeout=30)
    top.raise_for_status()
    subfolders = [line.strip() for line in top.text.splitlines() if line.strip()]

    records: list[str] = []
    for sub in tqdm(subfolders, desc="Listing subfolders"):
        sub = sub.rstrip("/")
        r = requests.get(f"{FILES_BASE}{sub}/RECORDS", timeout=30)
        r.raise_for_status()
        for name in r.text.splitlines():
            name = name.strip()
            if name:
                records.append(f"{sub}/{name}")
        if limit and len(records) >= limit:
            return records[:limit]
    return records


def download_records(records: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="Downloading records"):
        for ext in (".hea", ".mat"):
            dest = out_dir / f"{rec}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            r = requests.get(f"{FILES_BASE}{rec}{ext}", timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
    return out_dir


def download_records_mode(out_dir: Path, limit: int | None) -> Path:
    print("Listing records (walking nested RECORDS indexes) ...")
    records = list_records(limit=limit)
    print(f"Found {len(records):,} records" + (f" (limited to {limit})" if limit else ""))
    return download_records(records, out_dir)


def verify(data_root: Path, n_samples: int):
    sums_path = data_root / "SHA256SUMS.txt"
    if not sums_path.exists():
        print(f"Downloading {FILES_BASE}SHA256SUMS.txt for verification ...")
        _download_stream(FILES_BASE + "SHA256SUMS.txt", sums_path, desc="SHA256SUMS.txt")

    entries = []
    for line in sums_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1].strip()))

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

    if args.mode == "zip":
        data_root = download_zip(out_dir, Path(args.zip_tmp) if args.zip_tmp else None)
    else:
        data_root = download_records_mode(out_dir, args.limit)

    n_headers = len(list(out_dir.rglob("*.hea")))
    n_mats = len(list(out_dir.rglob("*.mat")))
    print(f"Done. {n_headers:,} .hea files, {n_mats:,} .mat files under {out_dir}")
    print(f"Data root (contains WFDBRecords/): {data_root}")

    if args.verify:
        verify(data_root, args.verify_samples)


if __name__ == "__main__":
    main()
