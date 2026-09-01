#!/usr/bin/env python3
"""
Download the CPSC2018 ECG database from PhysioNet.

CPSC2018 = the "cpsc_2018" source database (China Physiological Signal
Challenge 2018, Nanjing): 6,877 single 12-lead ECG records (WFDB .hea/.mat
pairs, 500 Hz, 6-60 s duration) with SNOMED-CT diagnosis codes in the header
comments.

Unlike CSN, CPSC2018 is not a standalone PhysioNet project — it is the
training/cpsc_2018/ subfolder inside the much larger "challenge-2020" project
(v1.0.2, 7.5 GB uncompressed across 6 source databases: cpsc_2018,
cpsc_2018_extra, st_petersburg_incart, ptb, ptb-xl, georgia). There is no
bulk-zip endpoint scoped to a single subfolder, so this script always walks
the nested RECORDS indexes and fetches individual .hea/.mat files, filtered
to the cpsc_2018 prefix only (cpsc_2018_extra — the original challenge's
unused-data subset — is deliberately excluded, to match the standard
6,877-record CPSC2018 benchmark other papers report against).

Source: https://physionet.org/content/challenge-2020/1.0.2/
Files:  https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/

Records are organized as training/cpsc_2018/g<N>/<record>.{hea,mat}, N in
1..7, up to 1000 records per subfolder (flat, unlike CSN's two-level
nesting). The project root's RECORDS file lists every subfolder across all
6 source databases; each subfolder has its own RECORDS index listing its
bare record names (e.g. "A0001").

Downloads run concurrently (default 16 threads, --workers to adjust) since
there's no bulk-zip endpoint here — the full pull is ~13,754 individual
.hea/.mat requests.

Gotcha found and worked around: each subfolder's RECORDS index over-claims
its last entry — g<N>/RECORDS lists A<N*1000>, but that record is actually
stored under g<N+1>/, not g<N>/ (confirmed for all 6 boundaries, g1..g6).
_download_one() falls back to the next subfolder on a 404 before giving up.

Usage:
  python scripts/download_cpsc.py
  python scripts/download_cpsc.py --out-dir /snfs2/josefernandes/datasets/lesaude/cpsc-monolith/raw
  python scripts/download_cpsc.py --limit 20     # quick smoke test
  python scripts/download_cpsc.py --verify       # SHA256 spot-check after download
"""

import argparse
import hashlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_WORKERS = 16
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=DEFAULT_WORKERS * 2))

PROJECT_BASE = "https://physionet.org/files/challenge-2020/1.0.2/"
SUBSET_PREFIX = "training/cpsc_2018/"
FILES_BASE = PROJECT_BASE + SUBSET_PREFIX
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/cpsc-monolith/raw"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Download only the first N records (quick smoke test).",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Concurrent download threads.",
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


def list_subfolders() -> list[str]:
    """The project-root RECORDS file lists every subfolder across all 6
    source databases (e.g. 'training/cpsc_2018/g1/'); filter to cpsc_2018
    only. A strict '.../cpsc_2018/' prefix (trailing slash) is required so
    'training/cpsc_2018_extra/...' entries don't slip in."""
    r = requests.get(PROJECT_BASE + "RECORDS", timeout=30)
    r.raise_for_status()
    subfolders = []
    for line in r.text.splitlines():
        line = line.strip().rstrip("/")
        if line and (line + "/").startswith(SUBSET_PREFIX):
            subfolders.append(line[len(SUBSET_PREFIX):])  # e.g. "g1"
    return subfolders


def list_records(limit: int | None = None) -> list[str]:
    """Walk cpsc_2018's per-subfolder RECORDS indexes and return record
    names relative to FILES_BASE, e.g. 'g1/A0001' (no extension)."""
    subfolders = list_subfolders()
    records: list[str] = []
    for sub in tqdm(subfolders, desc="Listing subfolders"):
        r = requests.get(f"{FILES_BASE}{sub}/RECORDS", timeout=30)
        r.raise_for_status()
        for name in r.text.splitlines():
            name = name.strip()
            if name:
                records.append(f"{sub}/{name}")
        if limit and len(records) >= limit:
            return records[:limit]
    return records


MAX_RETRIES = 4
_SUBFOLDER_RE = re.compile(r"^g(\d+)/(.+)$")


def _fallback_url(rec: str) -> str | None:
    """Each subfolder's RECORDS index over-claims its last (1000th) entry:
    e.g. g1/RECORDS lists A1000, but A1000.hea/.mat is actually stored under
    g2/, not g1/ — confirmed for every g<N>/A<N*1000> boundary in this
    dataset (g1..g6). A 404 on the nominal path falls back to the next
    subfolder before giving up."""
    m = _SUBFOLDER_RE.match(rec)
    if not m:
        return None
    n, name = m.groups()
    return f"g{int(n) + 1}/{name}"


def _download_one(rec: str, ext: str, out_dir: Path):
    dest = out_dir / f"{rec}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _session.get(f"{FILES_BASE}{rec}{ext}", timeout=60)
            if r.status_code == 404:
                fallback = _fallback_url(rec)
                if fallback:
                    r = _session.get(f"{FILES_BASE}{fallback}{ext}", timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
    raise last_exc


def download_records(records: list[str], out_dir: Path, workers: int = DEFAULT_WORKERS) -> Path:
    """A single file's transient network failure (rare but expected at
    ~13.7k requests) shouldn't abort thousands of already-succeeded
    downloads, so failures are retried (see _download_one) and, if still
    failing, collected and reported rather than raised mid-batch. Rerunning
    the same command afterwards resumes cleanly (skip-if-exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(rec, ext) for rec in records for ext in (".hea", ".mat")]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, rec, ext, out_dir): (rec, ext) for rec, ext in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            rec, ext = futures[future]
            try:
                future.result()
            except requests.exceptions.RequestException as exc:
                failures.append((rec, ext, exc))

    if failures:
        print(f"\n{len(failures)} file(s) failed after {MAX_RETRIES} retries each:")
        for rec, ext, exc in failures:
            print(f"  {rec}{ext}: {exc}")
        print("Rerun the same command to retry only the missing files (already-downloaded ones are skipped).")
    return out_dir


def verify(out_dir: Path, n_samples: int):
    """SHA256SUMS.txt lives at the challenge-2020 project root and covers
    all 6 source databases; filter entries to the cpsc_2018 subset and
    re-root them relative to out_dir (which mirrors g<N>/<record>.{hea,mat},
    stripped of the training/cpsc_2018/ prefix)."""
    sums_path = out_dir / "SHA256SUMS.txt"
    if not sums_path.exists():
        print(f"Downloading {PROJECT_BASE}SHA256SUMS.txt for verification ...")
        _download_stream(PROJECT_BASE + "SHA256SUMS.txt", sums_path, desc="SHA256SUMS.txt")

    entries = []
    for line in sums_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, rel_path = parts[0], parts[1].strip()
        if not rel_path.startswith(SUBSET_PREFIX):
            continue
        rel_path = rel_path[len(SUBSET_PREFIX):]
        if rel_path.endswith("/RECORDS"):
            # Per-subfolder index files (g<N>/RECORDS) are only ever read
            # in-memory by list_records() to build the download list — the
            # script never writes them to out_dir, so they're not part of
            # what this check is verifying.
            continue
        entries.append((digest, rel_path))

    sample = random.sample(entries, min(n_samples, len(entries)))
    n_ok = n_missing = n_bad = 0
    for expected_hash, rel_path in tqdm(sample, desc="Verifying"):
        f = out_dir / rel_path
        if not f.exists():
            n_missing += 1
            print(f"  MISSING: {rel_path}")
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

    print("Listing cpsc_2018 records (walking nested RECORDS indexes) ...")
    records = list_records(limit=args.limit)
    print(f"Found {len(records):,} records" + (f" (limited to {args.limit})" if args.limit else ""))
    download_records(records, out_dir, workers=args.workers)

    n_headers = len(list(out_dir.rglob("*.hea")))
    n_mats = len(list(out_dir.rglob("*.mat")))
    print(f"Done. {n_headers:,} .hea files, {n_mats:,} .mat files under {out_dir}")

    if args.verify:
        verify(out_dir, args.verify_samples)


if __name__ == "__main__":
    main()
