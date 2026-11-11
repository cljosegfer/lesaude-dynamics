"""
Part 1 Representation h
=======================
Frozen embeddings h of every ECG in the Part 1 pairs (gonogo/part1/pairs.py), from an encoder trained without labels
or actions (gonogo/go_no_go_plan.md, 2.1). Default: the no-action ViT of scripts/pretrain.py ++use_actions=false, run
runs/runs/20261019/182223/646e2795c834 (ViT 128 x 4, k=1 cross_stay pairs of folds <= 17, 405 epochs), checkpoint
pretrain_zeroactions.ckpt, selected on the label-free val/loss (epoch 339). Its sibling pretrain_zeroactions_auroc.ckpt
was selected on the online probe's AUROC, i.e. on fold-18 labels, so it is not the default.

That encoder was trained to predict h_{t+1} from h_t on the cross-stay pairs of folds <= 17, i.e. on the very pairs
Part 1 scores, without their labels or actions. Memorization check: the relative pair distance
||h_{t+1} - h_t||^2 / E||h_i - h_j||^2 (two random ECGs of these pairs), for the encoder's training folds (<= 17), its
val fold (18) and its test fold (19). If the training pairs are much closer, the encoder compressed exactly the
changes Part 1 measures there, and the gain must be compared between fold groups.

Resumable: embeddings are written per chunk of rows under outputs/gonogo_part1/cache/h_<tag>/ and assembled into
outputs/gonogo_part1/h_<tag>.npz (Lance rows, h float32) at the end, so a job cut by the allocation resumes where it
stopped.

Example
-------
python gonogo/part1/embed_h.py        # on a node whose /scratch holds the Lance table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "anatomie"))
from encoders import MIMICLanceDataset, data_paths, load_encoder  # noqa: E402

OUT = REPO / "outputs" / "gonogo_part1"
DEFAULT_CKPT = "runs/runs/20261019/182223/646e2795c834/checkpoints/pretrain_zeroactions.ckpt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--tag", default="zeroactions", help="names the cache: h_<tag>.npz")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def embed_chunks(enc, lance_path, rows, chunk_dir, args):
    """Writes chunk_<i>.npy (len(chunk), 256) float32 for every chunk of `rows` not yet on disk."""
    bounds = list(range(0, len(rows), args.chunk))
    todo = [i for i, s in enumerate(bounds) if not (chunk_dir / f"chunk_{i:04d}.npy").exists()]
    if not todo:
        return
    print(f"embedding {sum(len(rows[bounds[i]:bounds[i] + args.chunk]) for i in todo):,} ECGs "
          f"in {len(todo)} chunks", flush=True)
    ds = MIMICLanceDataset(lance_path, split="train", mode="monitoring")
    ds.rows = np.concatenate([rows[bounds[i]:bounds[i] + args.chunk] for i in todo])   # any rows, in this order
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        multiprocessing_context="spawn", pin_memory=True)
    pending, buf, n_buf = list(todo), [], 0
    for batch in loader:
        buf.append(enc(batch["waveform"].cuda(non_blocking=True))["h"].cpu().numpy())
        n_buf += len(buf[-1])
        while pending and n_buf >= len(rows[bounds[pending[0]]:bounds[pending[0]] + args.chunk]):
            i = pending.pop(0)
            n = len(rows[bounds[i]:bounds[i] + args.chunk])
            h = np.concatenate(buf)
            np.save(chunk_dir / f"chunk_{i:04d}.npy.tmp.npy", h[:n].astype(np.float32))
            (chunk_dir / f"chunk_{i:04d}.npy.tmp.npy").rename(chunk_dir / f"chunk_{i:04d}.npy")
            buf, n_buf = [h[n:]], len(h) - n
            print(f"  chunk {i + 1}/{len(bounds)} saved", flush=True)


def main():
    args = parse_args()
    paths = data_paths(args.config)
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    rows = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    out = OUT / f"h_{args.tag}.npz"
    if not out.exists():
        chunk_dir = OUT / "cache" / f"h_{args.tag}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        meta = {"ckpt": args.ckpt, "n_rows": int(len(rows)), "chunk": args.chunk,
                "rows_checksum": int(rows.sum() % (2 ** 61))}
        if (chunk_dir / "meta.json").exists():
            assert json.loads((chunk_dir / "meta.json").read_text()) == meta, f"stale chunks in {chunk_dir}"
        (chunk_dir / "meta.json").write_text(json.dumps(meta))
        embed_chunks(load_encoder(args.ckpt), paths["lance_path"], rows, chunk_dir, args)
        n_chunks = len(range(0, len(rows), args.chunk))
        h = np.concatenate([np.load(chunk_dir / f"chunk_{i:04d}.npy") for i in range(n_chunks)])
        assert len(h) == len(rows)
        np.savez(out, rows=rows, h=h, ckpt=args.ckpt)
        print(f"saved {out}", flush=True)
    z = np.load(out)
    h, lookup = z["h"], pd.Series(np.arange(len(z["rows"])), index=z["rows"])

    # Memorization check: relative pair distance by the encoder's own split of the folds
    ht, ht1 = h[lookup.loc[pairs["idx_t"].values].values], h[lookup.loc[pairs["idx_t1"].values].values]
    rel = ((ht1 - ht) ** 2).sum(1) / (2 * h.var(0).sum())
    group = np.select([pairs["fold"] <= 17, pairs["fold"] == 18], ["train folds <= 17", "val fold 18"], "test fold 19")
    report = {}
    for g in ("train folds <= 17", "val fold 18", "test fold 19"):
        m = group == g
        for kind, mk in (("stable", pairs["stable"].values), ("changed", ~pairs["stable"].values)):
            report[f"{g}, {kind}"] = {"n": int((m & mk).sum()), "rel_dist_mean": float(rel[m & mk].mean()),
                                      "rel_dist_median": float(np.median(rel[m & mk]))}
    print("relative pair distance ||h_t+1 - h_t||^2 / E||h_i - h_j||^2 (mean / median):", flush=True)
    for k, v in report.items():
        print(f"  {k:<28} n={v['n']:>7,}  {v['rel_dist_mean']:.3f} / {v['rel_dist_median']:.3f}", flush=True)
    (OUT / f"h_{args.tag}_memorization.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()
