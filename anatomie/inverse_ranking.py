"""
Inverse-Head Reference for anatomie/decoy_objectives.py
=========================================================
How much evidence about the action does the outcome carry? A fresh inverse head (ActionPredictor on d = h_t+1 - h_t,
BCE onset + resolution) on the same frozen, standardized features scores candidate actions by their log-likelihood
under its Bernoulli outputs, E(a) = -log p(a | d), on the same decoys and targets as the forward heads:
real h_t+1, another pair's h_t+1 (shuf) and the pair's own h_t (none, d = 0). Also reports the paper's
contextualized AUROC and CEM planning with this energy.

Example
-------
python anatomie/inverse_ranking.py     # needs outputs/decoy_objectives/feats.pt from decoy_objectives.py
"""

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_objectives as do  # noqa: E402
from models.dynamics import ActionPredictor  # noqa: E402

DEV, K, BS = do.DEV, do.K, do.BS


def loglik(lo, lr, cand, yt):
    """log p(a | d) of candidate actions (B, T, C) under onset/resolution logits (B, C) given y_t (B, C)."""
    on, rs = (cand == 1).float(), (cand == -1).float()
    st, y = 1 - on - rs, yt[:, None]
    ll = (on * F.logsigmoid(lo)[:, None] + rs * F.logsigmoid(lr)[:, None]
          + st * (1 - y) * F.logsigmoid(-lo)[:, None] + st * y * F.logsigmoid(-lr)[:, None])
    return ll.sum(-1)


@torch.no_grad()
def evaluate(head, split, probe, target="real"):
    ht, ht1, yt, at = split
    if target == "shuffled":
        ht1 = ht1[torch.randperm(len(ht1), generator=torch.Generator().manual_seed(7)).to(DEV)]
    elif target == "nochange":
        ht1 = ht
    probe._generator = torch.Generator(device=DEV).manual_seed(1234)   # the decoys of decoy_objectives.evaluate
    pgen = torch.Generator(device=DEV).manual_seed(99)
    acc = {k: [] for k in ("same", "extra", "zero", "prior", "ctx")}
    for b in torch.arange(len(ht), device=DEV).split(BS):
        cand, valid = do.probe_candidates(probe, yt[b], at[b])
        lo, lr = head(ht1[b] - ht[b])
        E = -loglik(lo, lr, cand, yt[b])
        active = (at[b] != 0).any(1)
        same, has = do.win_rate(E[:, 1:K + 1], E[:, :1], valid[:, 1:K + 1])
        acc["same"].append(same[active & has])
        acc["extra"].append(do.win_rate(E[:, K + 1:2 * K + 1], E[:, :1], valid[:, K + 1:2 * K + 1])[0])
        acc["zero"].append(do.win_rate(E[:, -1:], E[:, :1], valid[:, -1:])[0][active])
        dp = do.prior_decoys(yt[b], K, pgen)
        pr, has = do.win_rate(-loglik(lo, lr, dp, yt[b]), E[:, :1], (dp != at[b].unsqueeze(1)).any(-1))
        acc["prior"].append(pr[has])
        acc["ctx"].append(yt[b] * (1 - torch.sigmoid(lr)) + (1 - yt[b]) * torch.sigmoid(lo))
    c = {k: torch.cat(v) for k, v in acc.items()}
    out = {k: c[k].mean().item() for k in ("same", "extra", "zero", "prior")}
    out["ctx"] = do.macro_auroc(c["ctx"], (yt + at).clamp(0, 1))
    return out


@torch.no_grad()
def cem_eval(cem, head, split):
    ht, ht1, yt, at = split
    planner = cem.OnsetResolutionCEM(n_samples=512, n_elites=64, n_iters=16, smoothing=0.1, p_min=0.005,
                                     temperature=None)
    gen = torch.Generator(device=DEV).manual_seed(42)
    soft, hard, ons = [], [], []
    for b in torch.arange(len(ht), device=DEV).split(256):
        lo, lr = head(ht1[b] - ht[b])
        y = yt[b]
        r = planner.plan(lambda a: -loglik(lo, lr, a, y), y, do.PRIOR["ons"], do.PRIOR["res"], gen)
        p_ons, p_res = torch.nan_to_num(r["p_ons"]), torch.nan_to_num(r["p_res"])
        soft.append(torch.where(y == 1, 1 - p_res, p_ons))
        hard.append((y + r["a_star"]).clamp(0, 1))
        ons.append((r["a_star"] == 1).sum(1).float())
    yt1 = (yt + at).clamp(0, 1)
    return {"cem_soft": do.macro_auroc(torch.cat(soft), yt1), "cem_hard": do.macro_auroc(torch.cat(hard), yt1),
            "onsets": torch.cat(ons).mean().item()}


def train(data, jp, seed=0):
    torch.manual_seed(seed)
    ht, ht1, yt, at = data["train"]
    head = ActionPredictor(ht.shape[1], 512, 76).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    probe = jp._CEMProbe(n_decoys=K, seed=0)
    best = None
    for ep in range(do.EPOCHS):
        t0 = time.time()
        head.train()
        for b in torch.randperm(len(ht), device=DEV).split(BS):
            lo, lr = head(ht1[b] - ht[b])
            loss = (F.binary_cross_entropy_with_logits(lo, (at[b] == 1).float())
                    + F.binary_cross_entropy_with_logits(lr, (at[b] == -1).float()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        m = evaluate(head, data["val"], probe)
        if best is None or m["same"] > best[0]:
            best = (m["same"], ep, copy.deepcopy(head.state_dict()))
        if ep in (0, 4, 9, do.EPOCHS - 1):
            print(f"    ep {ep:>2} ({time.time() - t0:4.1f}s) val same {m['same']:.3f} extra {m['extra']:.3f} "
                  f"zero {m['zero']:.3f} prior {m['prior']:.3f} ctx {m['ctx']:.4f}", flush=True)
    head.load_state_dict(best[2])
    head.eval()
    return head, best[1], probe


def main():
    jp, cem = do.load_script("joint_pretrain"), do.load_script("cem_planning")
    data, paths = do.build_data()
    counts = cem._transition_counts(paths["lance_path"], paths["pairs_path"], ("cross_stay",), 76)
    p_ons, p_res = cem._priors(counts, "base_rate")
    do.PRIOR["ons"], do.PRIOR["res"] = torch.from_numpy(p_ons).to(DEV), torch.from_numpy(p_res).to(DEV)
    results = {}
    for enc, d in data.items():
        print(f"\n===== {enc}", flush=True)
        head, ep, probe = train(d, jp)
        r = {"best_epoch": ep, **{t: evaluate(head, d["test"], probe, t) for t in ("real", "shuffled", "nochange")},
             **cem_eval(cem, head, d["test"])}
        results[enc] = r
        t, s, z = r["real"], r["shuffled"], r["nochange"]
        fmt = lambda k: f"{k} {t[k]:.3f} ({s[k]:.3f}/{z[k]:.3f})"
        print(f"  => inverse    ep {ep:>2} | {fmt('same')} {fmt('extra')} {fmt('zero')} {fmt('prior')} | ctx "
              f"{t['ctx']:.4f} ({s['ctx']:.4f}/{z['ctx']:.4f}) | CEM soft {r['cem_soft']:.4f} hard "
              f"{r['cem_hard']:.4f} onsets {r['onsets']:.2f}", flush=True)
        with open(do.OUT / "inverse_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
