"""
Forward-Head Training Objectives on Frozen Embeddings
=======================================================
Stage 2 of demo/plan_lenepa_wm.md: which objective lets the forward energy E(a) = mean_D ||Dyn(h_t, a) - h_t+1||^2
rank actions? Heads (ActionProjector + DynamicsPredictor) train on frozen, standardized k=1 pair embeddings:

  mse        MSE of the true action only
  nce        + cross-entropy over candidates, logits -E/tau (tau learned). Candidates: the true action,
             16 same-count and 16 extra-flip decoys (_CEMProbe in scripts/joint_pretrain.py), the zero action
  nce-prior  + the same cross-entropy, with 32 decoys drawn from the base-rate prior given y_t (CEM's proposal)
  sigT       + SIGReg per pair over its candidates' predictions (T up to 34): decoy-SIGReg, per sample
  sigF       + SIGReg pooled over all candidates' predictions of the batch: decoy-SIGReg, pooled

Metrics on test, at the epoch with the best val "same":
  same / extra / zero / prior   within-pair AUROC of the true action vs same-count, extra-flip, zero-action
                                and prior-sampled decoys (0.5 = chance)
  (shuf)                        the same, scored against another pair's h_t+1: what survives is ranking by
                                plausibility given h_t, not evidence from the outcome
  (none)                        the same, scored against the pair's own h_t ("nothing changed"): keeps the patient,
                                removes only the change; a head that ranks as well here doesn't use the change
  cf_rel / true_rel             MSE of counterfactual (decoy) / true-action predictions to the pair's own h_t
                                over MSE to another pair's h_t (0 = stays with the patient, 1 = unrelated)
  CEM soft / hard               scripts/cem_planning.py's planner with plain E; contextualized macro AUROC

Example
-------
python anatomie/decoy_objectives.py      # resumable: heads already in results.json are skipped
"""

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "anatomie"))
from encoders import data_paths, embed, load_encoder  # noqa: E402
from models.dynamics import ActionProjector, DynamicsPredictor, EppsPulley  # noqa: E402

OUT = REPO / "outputs" / "decoy_objectives"
DEV, K, N_PRIOR, BS, EPOCHS, S = "cuda", 16, 32, 1024, 20, 128
SPLITS = {"train": "fold <= 17", "val": "fold = 18", "test": "fold = 19"}
ENCODERS = {"random/layer3": ("random", "layer3"), "inverse/h": ("archive/inverse.ckpt", "h")}
VARIANTS = [("mse", 0.0), ("nce", 1.0), ("nce-prior", 1.0), ("sigT", 0.1), ("sigT", 1.0), ("sigF", 0.1), ("sigF", 1.0)]
PRIOR = {}   # "ons", "res": per-class base rates on DEV, set in main()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- data
def build_data():
    import lance
    paths = data_paths()
    pairs = {}
    for split, cond in SPLITS.items():
        t = (lance.dataset(paths["pairs_path"]).to_table(filter=f"({cond}) AND (pair_type = 'cross_stay')",
                                                        columns=["idx_t", "idx_t1"]).to_pandas())
        pairs[split] = (t["idx_t"].values, t["idx_t1"].values)
    rows = np.unique(np.concatenate([np.concatenate(v) for v in pairs.values()]))
    cache = OUT / "feats.pt"
    if cache.exists():
        z = torch.load(cache, weights_only=False)   # our own cache; holds a numpy row index
        assert np.array_equal(z["rows"], rows), "cached features are for other pairs"
        feats, labels = z["feats"], z["labels"]
    else:
        t0 = time.time()
        encs = {n: load_encoder(spec) for n, (spec, _) in ENCODERS.items()}
        f, labels = embed(encs, rows, paths["lance_path"], layers={"layer3", "h"}, batch_size=256, num_workers=16)
        feats = {n: f[n][layer] for n, (_, layer) in ENCODERS.items()}
        torch.save({"rows": rows, "feats": feats, "labels": labels}, cache)
        print(f"embedded {len(rows):,} ECGs in {(time.time() - t0) / 60:.1f} min", flush=True)
    pos = {k: (np.searchsorted(rows, a), np.searchsorted(rows, b)) for k, (a, b) in pairs.items()}
    train_rows = np.unique(np.concatenate(pos["train"]))
    data = {}
    for n, E in feats.items():
        mu, sd = E[train_rows].mean(0), E[train_rows].std(0) + 1e-6
        E = ((E - mu) / sd).to(DEV)
        data[n] = {}
        for split, (pt, pt1) in pos.items():
            yt, yt1 = labels[pt].to(DEV), labels[pt1].to(DEV)
            data[n][split] = (E[pt], E[pt1], yt, (yt1 - yt).clamp(-1, 1))
    print({k: len(v[0]) for k, v in pairs.items()}, f"{len(rows):,} ECGs", flush=True)
    return data, paths


# ---------------------------------------------------------------- candidates and losses
def probe_candidates(probe, yt, at):
    """(B, T, C) actions [true, K same-count, K extra-flip, zero] and (B, T) validity (decoys equal to the truth,
    and same-count decoys of stable pairs, are invalid)."""
    cand = probe._candidates(yt, at)                                   # (B, 1+2K, C)
    cand = torch.cat([cand, torch.zeros_like(cand[:, :1])], 1)
    active = (at != 0).any(1)
    valid = (cand != at.unsqueeze(1)).any(-1)
    valid[:, 0] = True
    valid[:, 1:K + 1] &= active.unsqueeze(1)
    return cand, valid


def prior_decoys(yt, n, gen):
    """n admissible actions per pair from the base-rate prior given y_t: CEM's initial proposal."""
    B, C = yt.shape
    u_ons = torch.bernoulli(PRIOR["ons"].expand(B, n, C).contiguous(), generator=gen) * (1 - yt).unsqueeze(1)
    u_res = torch.bernoulli(PRIOR["res"].expand(B, n, C).contiguous(), generator=gen) * yt.unsqueeze(1)
    return u_ons - u_res


def prior_candidates(gen, yt, at):
    cand = torch.cat([at.unsqueeze(1), prior_decoys(yt, N_PRIOR, gen)], 1)
    valid = (cand != at.unsqueeze(1)).any(-1)
    valid[:, 0] = True
    return cand, valid


EP = EppsPulley().to(DEV)


def slices(D, step):
    g = torch.Generator(device=DEV).manual_seed(step)
    A = torch.randn(D, S, device=DEV, generator=g)
    return A / A.norm(dim=0)


def sig_per_sample(P, valid, A):
    """Temporal-style SIGReg: Epps-Pulley over each pair's valid predictions, scaled by their number."""
    x = (P @ A).unsqueeze(-1) * EP.t                                   # (B, T, S, nt)
    m = valid.float()[:, :, None, None]
    n = valid.sum(1).float()
    c = (x.cos() * m).sum(1) / n[:, None, None]
    s = (x.sin() * m).sum(1) / n[:, None, None]
    err = (c - EP.phi).square() + s.square()                           # (B, S, nt)
    return ((err @ EP.weights) * n[:, None]).mean()


def sig_pooled(X, A):
    x = (X @ A).unsqueeze(-1) * EP.t                                   # (N, S, nt)
    err = (x.cos().mean(0) - EP.phi).square() + x.sin().mean(0).square()
    return ((err @ EP.weights) * X.shape[0]).mean()


def predict(proj, pred, ht, cand):
    B, T, C = cand.shape
    return pred(ht.repeat_interleave(T, 0), proj(cand.reshape(B * T, C))).view(B, T, -1)


def win_rate(E_dec, E_true, valid):
    w = (E_dec > E_true).float() + 0.5 * (E_dec == E_true).float()
    return (w * valid).sum(1) / valid.sum(1).clamp(min=1), valid.any(1)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(proj, pred, split, probe, target="real"):
    ht, ht1, yt, at = split
    if target == "shuffled":    # another pair's outcome
        ht1 = ht1[torch.randperm(len(ht1), generator=torch.Generator().manual_seed(7)).to(DEV)]
    elif target == "nochange":  # the patient, without the change
        ht1 = ht
    probe._generator = torch.Generator(device=DEV).manual_seed(1234)
    pgen = torch.Generator(device=DEV).manual_seed(99)
    acc = {k: [] for k in ("same", "extra", "zero", "prior", "e_true", "nothing", "sigT", "cf", "tp")}
    for b in torch.arange(len(ht), device=DEV).split(BS):
        cand, valid = probe_candidates(probe, yt[b], at[b])
        P = predict(proj, pred, ht[b], cand)
        E = (P - ht1[b, None]).pow(2).mean(-1)
        active = (at[b] != 0).any(1)
        same, has = win_rate(E[:, 1:K + 1], E[:, :1], valid[:, 1:K + 1])
        acc["same"].append(same[active & has])
        acc["extra"].append(win_rate(E[:, K + 1:2 * K + 1], E[:, :1], valid[:, K + 1:2 * K + 1])[0])
        acc["zero"].append(win_rate(E[:, -1:], E[:, :1], valid[:, -1:])[0][active])
        dp = prior_decoys(yt[b], K, pgen)
        Ep = (predict(proj, pred, ht[b], dp) - ht1[b, None]).pow(2).mean(-1)
        pr, has = win_rate(Ep, E[:, :1], (dp != at[b].unsqueeze(1)).any(-1))
        acc["prior"].append(pr[has])
        acc["e_true"].append(E[:, 0])
        acc["nothing"].append((ht1[b] - ht[b]).pow(2).mean(-1))
        acc["sigT"].append(sig_per_sample(P, valid, slices(P.shape[-1], 0)).reshape(1))
        other = ht[b].roll(1, 0)
        dec = valid.clone()
        dec[:, 0] = False
        d_own, d_oth = (P - ht[b, None]).pow(2).mean(-1), (P - other[:, None]).pow(2).mean(-1)
        acc["cf"].append(torch.stack([d_own[dec].mean(), d_oth[dec].mean()]).reshape(1, 2))
        acc["tp"].append(torch.stack([d_own[:, 0].mean(), d_oth[:, 0].mean()]).reshape(1, 2))
    c = {k: torch.cat(v) for k, v in acc.items()}
    cf, tp = c["cf"].mean(0), c["tp"].mean(0)
    nothing = c["nothing"].mean()
    return {"same": c["same"].mean().item(), "extra": c["extra"].mean().item(), "zero": c["zero"].mean().item(),
            "prior": c["prior"].mean().item(),
            "mse_ratio": (c["e_true"].mean() / nothing).item() if nothing > 0 else float("nan"),
            "sigT": c["sigT"].mean().item(), "cf_rel": (cf[0] / cf[1]).item(), "true_rel": (tp[0] / tp[1]).item()}


def macro_auroc(preds, target):
    from torchmetrics.functional.classification import multilabel_auroc
    per = multilabel_auroc(preds, target.long(), num_labels=target.shape[1], average="none")
    n_pos = target.sum(0)
    return per[(n_pos > 0) & (n_pos < len(target))].mean().item()


@torch.no_grad()
def cem_eval(cem, proj, pred, split):
    ht, ht1, yt, at = split
    planner = cem.OnsetResolutionCEM(n_samples=512, n_elites=64, n_iters=16, smoothing=0.1, p_min=0.005,
                                     temperature=None)
    gen = torch.Generator(device=DEV).manual_seed(42)
    soft, hard, ons = [], [], []
    for b in torch.arange(len(ht), device=DEV).split(256):
        r = planner.plan(cem._make_energy_fn(proj, pred, ht[b], ht1[b], 65536), yt[b], PRIOR["ons"], PRIOR["res"], gen)
        p_ons, p_res = torch.nan_to_num(r["p_ons"]), torch.nan_to_num(r["p_res"])
        soft.append(torch.where(yt[b] == 1, 1 - p_res, p_ons))
        hard.append((yt[b] + r["a_star"]).clamp(0, 1))
        ons.append((r["a_star"] == 1).sum(1).float())
    yt1 = (yt + at).clamp(0, 1)
    return {"cem_soft": macro_auroc(torch.cat(soft), yt1), "cem_hard": macro_auroc(torch.cat(hard), yt1),
            "onsets": torch.cat(ons).mean().item()}


# ---------------------------------------------------------------- training
def train(data, variant, lam, jp, seed=0):
    torch.manual_seed(seed)
    D = data["train"][0].shape[1]
    proj, pred = ActionProjector(76, D).to(DEV), DynamicsPredictor(D, 512).to(DEV)
    log_tau = torch.nn.Parameter(torch.tensor(math.log(0.05), device=DEV))
    nce = variant.startswith("nce")
    opt = torch.optim.AdamW(list(proj.parameters()) + list(pred.parameters()) + ([log_tau] if nce else []),
                            lr=1e-3, weight_decay=1e-2)
    probe, eval_probe = jp._CEMProbe(n_decoys=K, seed=0), jp._CEMProbe(n_decoys=K, seed=0)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    ht, ht1, yt, at = data["train"]
    best, step = None, 0
    for ep in range(EPOCHS):
        t0 = time.time()
        proj.train(), pred.train()
        for b in torch.randperm(len(ht), device=DEV).split(BS):
            if variant == "nce-prior":
                cand, valid = prior_candidates(gen, yt[b], at[b])
            else:
                probe._generator = gen
                cand, valid = probe_candidates(probe, yt[b], at[b])
            P = predict(proj, pred, ht[b], cand)
            E = (P - ht1[b, None]).pow(2).mean(-1)
            loss = E[:, 0].mean()
            if nce:
                logits = (-E / log_tau.exp()).masked_fill(~valid, float("-inf"))
                loss = loss + lam * F.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long, device=DEV))
            elif variant == "sigT":
                loss = loss + lam * sig_per_sample(P, valid, slices(P.shape[-1], step))
            elif variant == "sigF":
                loss = loss + lam * sig_pooled(P[valid], slices(P.shape[-1], step))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        proj.eval(), pred.eval()
        m = evaluate(proj, pred, data["val"], eval_probe)
        if best is None or m["same"] > best[0]:
            best = (m["same"], ep, copy.deepcopy(proj.state_dict()), copy.deepcopy(pred.state_dict()))
        if ep in (0, 4, 9, EPOCHS - 1):
            print(f"    ep {ep:>2} ({time.time() - t0:4.1f}s) val same {m['same']:.3f} extra {m['extra']:.3f} "
                  f"zero {m['zero']:.3f} prior {m['prior']:.3f} mse/nothing {m['mse_ratio']:.3f} "
                  f"sigT {m['sigT']:7.2f} cf_rel {m['cf_rel']:.3f}"
                  + (f" tau {log_tau.exp().item():.4f}" if nce else ""), flush=True)
    proj.load_state_dict(best[2])
    pred.load_state_dict(best[3])
    proj.eval(), pred.eval()
    return proj, pred, best[1], eval_probe


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jp, cem = load_script("joint_pretrain"), load_script("cem_planning")
    data, paths = build_data()
    counts = cem._transition_counts(paths["lance_path"], paths["pairs_path"], ("cross_stay",), 76)
    p_ons, p_res = cem._priors(counts, "base_rate")
    PRIOR["ons"], PRIOR["res"] = torch.from_numpy(p_ons).to(DEV), torch.from_numpy(p_res).to(DEV)
    res_path = OUT / "results.json"
    results = json.load(open(res_path)) if res_path.exists() else {}
    for enc, d in data.items():
        ht, ht1, yt, at = d["test"]
        results.setdefault(enc, {})["carry_forward"] = macro_auroc(yt, (yt + at).clamp(0, 1))
        print(f"\n===== {enc}: test carry-forward {results[enc]['carry_forward']:.4f}", flush=True)
        for variant, lam in VARIANTS:
            name = variant + (f" λ{lam:g}" if variant.startswith("sig") else "")
            if name in results[enc]:
                print(f"  -- {name}: done, skipped", flush=True)
                continue
            print(f"  -- {name}", flush=True)
            proj, pred, ep, probe = train(d, variant, lam, jp)
            (OUT / "heads").mkdir(exist_ok=True)
            torch.save({"proj": proj.state_dict(), "pred": pred.state_dict(), "best_epoch": ep},
                       OUT / "heads" / f"{enc.replace('/', '_')}__{name.replace(' ', '_')}.pt")
            r = {"best_epoch": ep, "test": evaluate(proj, pred, d["test"], probe),
                 "test_shuffled": evaluate(proj, pred, d["test"], probe, target="shuffled"),
                 "test_nochange": evaluate(proj, pred, d["test"], probe, target="nochange"),
                 **cem_eval(cem, proj, pred, d["test"])}
            results[enc][name] = r
            t, s, z = r["test"], r["test_shuffled"], r["test_nochange"]
            fmt = lambda k: f"{k} {t[k]:.3f} ({s[k]:.3f}/{z[k]:.3f})"
            print(f"  => {name:<10} ep {ep:>2} | {fmt('same')} {fmt('extra')} {fmt('zero')} {fmt('prior')} | "
                  f"mse/nothing {t['mse_ratio']:.3f} cf_rel {t['cf_rel']:.3f} | CEM soft {r['cem_soft']:.4f} "
                  f"hard {r['cem_hard']:.4f} onsets {r['onsets']:.2f}", flush=True)
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()
