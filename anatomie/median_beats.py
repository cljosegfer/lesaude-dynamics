"""
Median Beats
============
The representative beat and two rhythm features of every ECG in the Lance table: the targets that
anatomie/signal_decoder.py decodes from frozen embeddings.

h is a global average over time, so it does not know where the beats fall in the 10 s window: cropping the
same ECG at a different start moves h by 0.3-2% of the distance between two unrelated ECGs (forward
encoders; 8-24% for random init). A decoder of the raw 12 x 5000 strip would learn the average over beat
positions, a near-flat trace. What h can carry is the beat shape and the rhythm statistics:

  beat     (12, 500) per-lead median over the complete beats, 0.4 s before to 0.6 s after the QRS, in the
           input's per-lead z-units at 500 Hz
  rhythm   heart rate (bpm, from the median RR interval) and RR irregularity (normalized RMSSD: RMS of the
           successive RR differences over the mean RR; on 2,000 val ECGs median 0.02, 0.15 on I48 ECGs)
  n_beats  complete beats behind the median; 0 = no usable beats (< 3 QRS or rate outside 20-250 bpm)

QRS detection: 5-20 Hz band-pass of every lead, squared slope summed over the leads, 100 ms moving average,
peaks >= 250 ms apart above 30% of the envelope's 99th percentile. Each beat is then re-aligned to the
first-pass median by cross-correlation (+-20 ms) before the final median.

The cache holds one entry per Lance row and lives next to the Lance table (<lance dir>/median_beats, ~10 GB).
load_cache() builds it on first use, in one pass over the whole table (~2.6 ms of CPU per ECG, spread over the
DataLoader workers).

Example
-------
python anatomie/median_beats.py            # build (or validate) the cache for configs/data.yaml
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import lance
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import MIMICLanceDataset, data_paths  # noqa: E402

FS = 500
PRE, POST = 200, 300           # samples kept before / after the QRS fiducial (0.4 s / 0.6 s)
BEAT_LEN = PRE + POST
LAG = 10                       # re-alignment search, +-20 ms
MIN_BEATS = 3
VERSION = 1                    # bump when the extraction changes, so stale caches are rebuilt
_SOS = butter(2, (5, 20), btype="bandpass", fs=FS, output="sos")


def _windows(x, peaks, pre, post):
    """(K, leads, pre + post) segments of x (leads, T) around each peak."""
    return sliding_window_view(x, pre + post, axis=-1)[:, peaks - pre].transpose(1, 0, 2)


def median_beats(x: np.ndarray):
    """x (B, 12, T) z-scored waveforms -> beats (B, 12, BEAT_LEN) float32, rhythm (B, 2) float32 as
    [heart rate bpm, normalized RMSSD] (NaN where unusable), n_beats (B,) int16 (0 = unusable)."""
    B, T = len(x), x.shape[-1]
    env = np.square(np.diff(sosfiltfilt(_SOS, x, axis=-1), axis=-1)).sum(1)
    env = uniform_filter1d(env, size=FS // 10, axis=-1)                       # (B, T - 1)
    beats = np.zeros((B, x.shape[1], BEAT_LEN), np.float32)
    rhythm = np.full((B, 2), np.nan, np.float32)
    n_beats = np.zeros(B, np.int16)
    for b in range(B):
        peaks, _ = find_peaks(env[b], height=0.3 * np.percentile(env[b], 99), distance=FS // 4)
        if len(peaks) < MIN_BEATS:
            continue
        rr = np.diff(peaks) / FS
        hr = 60.0 / np.median(rr)
        if not 20 <= hr <= 250:
            continue
        peaks = peaks[(peaks >= PRE + LAG) & (peaks + POST + LAG <= T)]
        if len(peaks) < MIN_BEATS:
            continue
        template = np.median(_windows(x[b], peaks, PRE, POST), 0)             # (12, BEAT_LEN)
        wide = _windows(x[b], peaks, PRE + LAG, POST + LAG)                     # (K, 12, BEAT_LEN + 2 LAG)
        score = np.einsum("klsw,lw->ks", sliding_window_view(wide, BEAT_LEN, axis=-1), template)
        peaks = peaks + score.argmax(1) - LAG
        beats[b] = np.median(_windows(x[b], peaks, PRE, POST), 0)
        rhythm[b] = hr, np.sqrt(np.mean(np.diff(rr) ** 2)) / np.mean(rr)
        n_beats[b] = len(peaks)
    return beats, rhythm, n_beats


def _collate(items):
    """Runs in the DataLoader workers, so the extraction is parallel."""
    beats, rhythm, n_beats = median_beats(torch.stack([it["waveform"] for it in items]).numpy())
    return beats.astype(np.float16), rhythm, n_beats


def _meta(lance_path):
    return {"version": VERSION, "lance_version": lance.dataset(lance_path).version, "fs": FS, "pre": PRE,
            "post": POST}


def build_cache(lance_path, out_dir, batch_size=256, num_workers=16):
    n = lance.dataset(lance_path).count_rows()
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    ds = MIMICLanceDataset(lance_path, split="train", mode="monitoring")
    ds.rows = np.arange(n)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
                        multiprocessing_context="spawn")
    beats = np.lib.format.open_memmap(tmp / "beats.npy", mode="w+", dtype=np.float16, shape=(n, 12, BEAT_LEN))
    rhythm, n_beats = np.zeros((n, 2), np.float32), np.zeros(n, np.int16)
    s = 0
    for b, r, k in tqdm(loader, desc="Median beats"):
        beats[s:s + len(k)], rhythm[s:s + len(k)], n_beats[s:s + len(k)] = b, r, k
        s += len(k)
    beats.flush()
    np.save(tmp / "rhythm.npy", rhythm)
    np.save(tmp / "n_beats.npy", n_beats)
    (tmp / "meta.json").write_text(json.dumps(_meta(lance_path)))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    print(f"Saved {out_dir}: {(n_beats > 0).mean():.1%} of {n:,} ECGs have usable beats", flush=True)


def load_cache(lance_path, cache_dir=None, **build_kwargs):
    """{"beats": (N, 12, BEAT_LEN) float16 memmap, "rhythm": (N, 2), "n_beats": (N,)}, indexed by Lance row, from
    cache_dir/median_beats (default: the Lance table's directory).
    Call from under `if __name__ == "__main__":` — a (re)build uses spawned DataLoader workers."""
    out_dir = Path(cache_dir or Path(lance_path).parent) / "median_beats"
    meta = out_dir / "meta.json"
    if not meta.exists() or json.loads(meta.read_text()) != _meta(lance_path):
        print(f"Building the median-beat cache at {out_dir} (one pass over the data)", flush=True)
        build_cache(lance_path, out_dir, **build_kwargs)
    return {"beats": np.load(out_dir / "beats.npy", mmap_mode="r"),
            "rhythm": np.load(out_dir / "rhythm.npy"),
            "n_beats": np.load(out_dir / "n_beats.npy")}


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    cache = load_cache(data_paths(args.config)["lance_path"], num_workers=args.workers)
    print(f"{(cache['n_beats'] > 0).mean():.1%} of {len(cache['n_beats']):,} ECGs have usable beats")
