"""
Per-Layer Readout Diagnostic
=============================
Where in a frozen encoder do (a) memorized training pairs and (b) linearly decodable diagnoses live?

For every readout layer of every encoder (anatomie/encoders.py), on train folds 0-17 and val fold 18:

  Pair distance = MSE(own pair) / MSE(same ECG vs a different patient's ECG); 0 = same point, 1 = unrelated.
    trained   first ECG of stay s -> first ECG of stay s+1, training patient (a pair the cross-visit losses see)
    unseen    second ECG of stay s -> first ECG of stay s+1, training patient (one recording no pair uses)
    new       first ECG of stay s -> first ECG of stay s+1, val patient
  Probe AUROC = linear probe trained on train triage ECGs, macro AUROC over the classes with both outcomes on
    all val triage ECGs (the macro of scripts/evaluate_inverse.py). Differences under ~0.01 are noise.

Interpretation
--------------
  trained << unseen ~ new  -> the layer memorizes training recordings (forward pretraining: layer4 and h only).
  trained ~ unseen ~ new   -> no memorization at that layer (every layer of the inverse encoder).
  Probe AUROC picks the readout layer; choose it on these val numbers, never on test.

Reference values (2026-10-05, default protocol; demo/plan_lenepa_wm.md has the full table):
  forward epoch 55: layer3 pair distance 0.29 / 0.30 / 0.27, probe 0.649;  layer4 0.017 / 0.12 / 0.24;  h probe 0.583
  random init: best probe 0.626 (layer3);  inverse_pretrain: 0.652 (layer3), h 0.644
  The probe depends on the protocol: with 20k / 10k samples and 1,000 steps, memorized h layers scored up to 0.05 higher.

Example
-------
python anatomie/layer_readout.py --ckpt random=random --ckpt inverse=archive/inverse.ckpt \\
    --ckpt fwd_ep55=runs/runs/20260930/062649/71f4c433519a/checkpoints/last.ckpt --out layer_readout.json
"""

import argparse
import json
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_auroc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import MIMICLanceDataset, data_paths, embed, load_encoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=SPEC",
                   help="encoder to evaluate (repeatable); SPEC is 'random' or a checkpoint path")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--n-pairs", type=int, default=3000)
    # 20k/10k samples moved layer rankings by up to 0.04 between draws; 100k train ECGs and the full val
    # triage set (14,456 ECGs) are the defaults so rankings are stable
    p.add_argument("--n-probe-train", type=int, default=100000)
    p.add_argument("--n-probe-val", type=int, default=10 ** 9, help="default: every val triage ECG")
    p.add_argument("--probe-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=None, help="optional JSON file for the results")
    return p.parse_args()


def stay_tables(lance_path):
    meta = lance.dataset(lance_path).to_table(
        columns=["subject_id", "ecg_time", "fold", "ecg_no_within_stay"]).to_pandas()
    meta["idx"] = meta.index
    meta = meta[meta["ecg_no_within_stay"] >= 0].sort_values(["subject_id", "ecg_time"])
    meta["stay"] = (meta["ecg_no_within_stay"] == 0).astype(int).groupby(meta["subject_id"]).cumsum()
    meta = meta[meta["stay"] >= 1]
    first = meta[meta["ecg_no_within_stay"] == 0].set_index(["subject_id", "stay"])["idx"]
    second = (meta[meta["ecg_no_within_stay"] == 1].drop_duplicates(["subject_id", "stay"])
              .set_index(["subject_id", "stay"])["idx"])
    fold = meta.groupby("subject_id")["fold"].first()
    return first, second, fold


def pair_sets(first, second, fold, n, rng):
    """{set: (ecg_a, ecg_b, subject, partner permutation)}; one permutation per set, shared by all encoders."""
    nxt = lambda k: pd.MultiIndex.from_arrays([k.get_level_values(0), k.get_level_values(1) + 1])
    sets = {}
    k = second.index[fold.loc[second.index.get_level_values(0)].values <= 17]
    k = k[nxt(k).isin(first.index)]
    k = k[rng.choice(len(k), size=min(n, len(k)), replace=False)]
    subj = k.get_level_values(0).values
    sets["trained"] = (first.loc[k].values, first.loc[nxt(k)].values, subj)
    sets["unseen"] = (second.loc[k].values, first.loc[nxt(k)].values, subj)
    k = first.index[fold.loc[first.index.get_level_values(0)].values == 18]
    k = k[nxt(k).isin(first.index)]
    k = k[rng.choice(len(k), size=min(n, len(k)), replace=False)]
    sets["new"] = (first.loc[k].values, first.loc[nxt(k)].values, k.get_level_values(0).values)
    return {name: (*v, rng.permutation(len(v[0]))) for name, v in sets.items()}


def sample_triage(lance_path, split, n, rng):
    rows = MIMICLanceDataset(lance_path, split=split, mode="triage").rows
    return np.sort(rng.choice(rows, size=min(n, len(rows)), replace=False))


def pair_distance(E, pos, a, b, subj, perm):
    A, B = E[[pos[i] for i in a]], E[[pos[i] for i in b]]
    other = subj[perm] != subj          # the partner must be a different patient
    return float(((A - B) ** 2).mean() / ((A[other] - B[perm][other]) ** 2).mean())


def probe_auroc(Xtr, Ytr, Xva, Yva, steps, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = ((Xtr - mu) / sd).cuda(), ((Xva - mu) / sd).cuda()
    Ytr, Yva = Ytr.cuda(), Yva.cuda()
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1]).cuda()
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (4096,), device="cuda")
        loss = F.binary_cross_entropy_with_logits(lin(Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        per = multilabel_auroc(torch.sigmoid(lin(Xva)), Yva.long(), num_labels=Yva.shape[1], average="none")
    n_pos = Yva.sum(0)
    return float(per[(n_pos > 0) & (n_pos < len(Yva))].mean())


def main():
    args = parse_args()
    lance_path = data_paths(args.config)["lance_path"]
    rng = np.random.default_rng(args.seed)

    first, second, fold = stay_tables(lance_path)
    sets = pair_sets(first, second, fold, args.n_pairs, rng)
    tr_rows = sample_triage(lance_path, "train", args.n_probe_train, rng)
    va_rows = sample_triage(lance_path, "val", args.n_probe_val, rng)
    rows = np.unique(np.concatenate([tr_rows, va_rows] + [np.concatenate(v[:2]) for v in sets.values()]))
    print({k: len(v[0]) for k, v in sets.items()},
          f"probe {len(tr_rows)} train / {len(va_rows)} val; {len(rows):,} ECGs to embed", flush=True)

    specs = dict(c.split("=", 1) for c in args.ckpt)
    encoders = {n: load_encoder(s) for n, s in specs.items()}
    feats, labels = embed(encoders, rows, lance_path, batch_size=args.batch_size, num_workers=args.workers)
    pos = {int(r): i for i, r in enumerate(rows)}
    tr, va = [pos[int(r)] for r in tr_rows], [pos[int(r)] for r in va_rows]

    results = {}
    print(f"\n{'encoder':<12}{'layer':<8}{'dim':>5}{'trained':>9}{'unseen':>8}{'new':>7}{'probe AUROC':>13}")
    for n in specs:
        for layer, E in feats[n].items():
            d = {k: pair_distance(E.numpy(), pos, *v) for k, v in sets.items()}
            auc = probe_auroc(E[tr], labels[tr], E[va], labels[va], args.probe_steps, args.seed)
            results.setdefault(n, {})[layer] = {**d, "probe_auroc": auc, "dim": E.shape[1]}
            print(f"{n:<12}{layer:<8}{E.shape[1]:>5}{d['trained']:>9.3f}{d['unseen']:>8.3f}{d['new']:>7.3f}"
                  f"{auc:>13.4f}", flush=True)
        print(flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"encoders": specs, "results": results}, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()
