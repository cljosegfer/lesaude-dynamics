"""
Frozen-Encoder Inverse Refit
=============================
How much transition information does a frozen encoder carry, independent of the head trained with it?

Embeds every ECG of the k=1 cross-stay pairs (pairs.lance; train folds 0-17, val 18, test 19) once with each
encoder (anatomie/encoders.py), then fits fresh inverse heads on d = h(t+1) - h(t) of the training pairs:
  MLP     ActionPredictor(D, 512, 76) — the head architecture of scripts/inverse_pretrain.py
  linear  one affine layer D -> 2 x 76 (onset and resolution logits)
Each head trains 30 epochs (AdamW 1e-3, weight decay 1e-2, batch 1024); the epoch with the best val
contextualized AUROC is kept and scored on train / val / test. Checkpoints that carry an inverse head
(inverse_pretrain.py, joint_pretrain.py) are also scored with that co-trained head (layer h only).

Contextualized AUROC is scripts/evaluate_inverse.py's readout, yt*(1-p_res) + (1-yt)*p_ons against y(t+1),
macro over the classes with both outcomes. Without --bootstrap it is a point estimate; --bootstrap 1000
adds the paper's 95% CI on test (same resampling function as evaluate_inverse.py).

Interpretation
--------------
  Fresh MLP test ctx is the encoder's transition information (2026-10-05 point estimates, k=1 test pairs):
  random init 0.811 (h) / 0.823 (layer3); inverse_pretrain encoder 0.832 (h) / 0.836 (layer3). The range is
  narrow next to the ±0.017 test CI: use it to check transition information survives, not to rank encoders.
  demo/plan_lenepa_wm.md has the full table.
  A fresh head that peaks at epoch 0, or a linear head with train ctx >> val ctx, means the encoder itself
  memorized training transitions (late forward / joint checkpoints).

Example
-------
python anatomie/refit_inverse_head.py --ckpt random=random --ckpt inverse=archive/inverse.ckpt \\
    --layers h layer3 --bootstrap 1000 --out refit.json
"""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import lance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_auroc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, ActionPredictor, data_paths, embed, load_encoder, load_inverse_head  # noqa: E402

SPLITS = {"train": "fold <= 17", "val": "fold = 18", "test": "fold = 19"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=SPEC",
                   help="encoder to evaluate (repeatable); SPEC is 'random' or a checkpoint path")
    p.add_argument("--layers", nargs="+", default=["h"], help="readout layers to refit on")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for the test CI (0 = off)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=None, help="optional JSON file for the results")
    return p.parse_args()


class LinearInv(nn.Module):
    def __init__(self, dim: int, action_dim: int = 76):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, d):
        out = self.fc(d)
        return out[:, :self.action_dim], out[:, self.action_dim:]


def macro_auroc(preds, target):
    per = multilabel_auroc(preds, target.long(), num_labels=target.shape[1], average="none")
    n_pos = target.sum(0)
    return float(per[(n_pos > 0) & (n_pos < len(target))].mean())


@torch.no_grad()
def logits(head, d):
    on, re = zip(*(head(chunk) for chunk in d.split(16384)))
    return torch.cat(on).float(), torch.cat(re).float()


def evaluate(head, d, yt, at):
    on, re = logits(head, d)
    loss = (F.binary_cross_entropy_with_logits(on, (at == 1).float())
            + F.binary_cross_entropy_with_logits(re, (at == -1).float()))
    ctx = yt * (1 - torch.sigmoid(re)) + (1 - yt) * torch.sigmoid(on)
    return {"ctx": macro_auroc(ctx, (yt + at).clamp(0, 1)), "loss": float(loss)}, ctx


def fit(make, data, epochs):
    """Fresh head on train d; returns (best-val epoch, {split: metrics} at that epoch, last-epoch metrics, head)."""
    torch.manual_seed(0)
    head = make().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    d, yt, at = data["train"]
    best = None
    for ep in range(epochs):
        head.train()
        for b in torch.randperm(len(d), device="cuda").split(1024):
            on, re = head(d[b])
            loss = (F.binary_cross_entropy_with_logits(on, (at[b] == 1).float())
                    + F.binary_cross_entropy_with_logits(re, (at[b] == -1).float()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        v = evaluate(head, *data["val"])[0]["ctx"]
        if best is None or v > best[0]:
            best = (v, copy.deepcopy(head.state_dict()), ep)
    last = {s: evaluate(head, *data[s])[0] for s in ("train", "val")}
    head.load_state_dict(best[1])
    return best[2], {s: evaluate(head, *data[s])[0] for s in ("train", "val", "test")}, last, head


def bootstrap_fn():
    spec = importlib.util.spec_from_file_location("evaluate_inverse", REPO / "scripts" / "evaluate_inverse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bootstrap_auroc


def main():
    args = parse_args()
    lance_path, pairs_path = (data_paths(args.config)[k] for k in ("lance_path", "pairs_path"))

    pairs = {}
    for split, cond in SPLITS.items():
        t = (lance.dataset(pairs_path).to_table(filter=f"({cond}) AND (pair_type = 'cross_stay')",
                                                columns=["idx_t", "idx_t1"]).to_pandas())
        pairs[split] = (t["idx_t"].values, t["idx_t1"].values)
    rows = np.unique(np.concatenate([np.concatenate(v) for v in pairs.values()]))
    print({k: len(v[0]) for k, v in pairs.items()}, f"{len(rows):,} unique ECGs", flush=True)

    specs = dict(c.split("=", 1) for c in args.ckpt)
    encoders = {n: load_encoder(s) for n, s in specs.items()}
    feats, labels = embed(encoders, rows, lance_path, layers=set(args.layers),
                          batch_size=args.batch_size, num_workers=args.workers)
    pos = {k: (np.searchsorted(rows, a), np.searchsorted(rows, b)) for k, (a, b) in pairs.items()}
    boot = bootstrap_fn() if args.bootstrap else None

    results = {}
    for n, spec in specs.items():
        for layer in args.layers:
            E = feats[n][layer]
            data = {}
            for split, (pt, pt1) in pos.items():
                yt, yt1 = labels[pt], labels[pt1]
                data[split] = ((E[pt1] - E[pt]).cuda(), yt.cuda(), (yt1 - yt).clamp(-1, 1).cuda())
            heads = {}
            co = load_inverse_head(spec) if layer == "h" else None
            if co is not None and co.input_proj.in_features == E.shape[1]:
                heads["co-trained"] = (None, {s: evaluate(co, *data[s])[0] for s in SPLITS}, None, co)
            heads["fresh MLP"] = fit(lambda: ActionPredictor(E.shape[1], 512, labels.shape[1]), data, args.epochs)
            heads["fresh linear"] = fit(lambda: LinearInv(E.shape[1], labels.shape[1]), data, args.epochs)

            print(f"\n== {n}  [{layer}, dim {E.shape[1]}]", flush=True)
            for h, (ep, m, last, head) in heads.items():
                r = {"best_epoch": ep, **{s: m[s] for s in SPLITS}, "last": last}
                ci = ""
                if boot is not None:
                    d, yt, at = data["test"]
                    ctx = evaluate(head, d, yt, at)[1].cpu().numpy()
                    target = (yt + at).clamp(0, 1).cpu().numpy()
                    r["test_ci"] = boot(ctx, target, args.bootstrap, 42, f"{n} {layer} {h}")
                    ci = f" [{r['test_ci'][1]:.4f}-{r['test_ci'][2]:.4f}]"
                extra = (f"  best-val ep {ep:>2}; after {args.epochs} ep: train {last['train']['ctx']:.4f} "
                         f"val {last['val']['ctx']:.4f}") if last else ""
                print(f"  {h:<13} test {m['test']['ctx']:.4f}{ci} | train {m['train']['ctx']:.4f} "
                      f"val {m['val']['ctx']:.4f} (loss train {m['train']['loss']:.4f} val {m['val']['loss']:.4f})"
                      f"{extra}", flush=True)
                results.setdefault(n, {}).setdefault(layer, {})[h] = r

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"encoders": specs, "results": results}, f, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()
