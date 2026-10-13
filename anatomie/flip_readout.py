"""
Per-Label Flip Readout for the Forward Heads of anatomie/decoy_objectives.py
==============================================================================
CEM returns one action per pair, so with weak per-label evidence it falls back to the prior mode (with the inverse
energy it returns the zero action and scores exactly carry-forward). This readout instead asks each label separately:
  delta_i = E(0) - E(e_i),    e_i = the admissible single flip of label i (onset if y_i = 0, resolution if y_i = 1)
  p_ons_i = sigmoid(logit(pi_ons_i) + delta_i / tau),  p_res_i likewise with pi_res_i   (pi: train base rates)
  ctx     = y * (1 - p_res) + (1 - y) * p_ons            (the paper's contextualized next-state score)
tau is chosen on val; tau = inf is the prior alone (carry-forward ordering). Scored against the real h_t+1 and
against the pair's own h_t ("nothing changed"), so the gain over that control is the evidence the energy carries.

Example
-------
python anatomie/flip_readout.py     # needs outputs/decoy_objectives/{feats.pt, heads/}
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_objectives as do  # noqa: E402
from models.dynamics import ActionProjector, DynamicsPredictor  # noqa: E402

DEV = do.DEV
TAUS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


@torch.no_grad()
def deltas(proj, pred, split, target="real"):
    ht, ht1, yt, at = split
    if target == "nochange":
        ht1 = ht
    out = []
    eye = torch.eye(76, device=DEV)
    for b in torch.arange(len(ht), device=DEV).split(256):
        flips = eye.unsqueeze(0) * (1 - 2 * yt[b]).unsqueeze(1)                 # (B, C, C): row i flips label i
        cand = torch.cat([torch.zeros_like(flips[:, :1]), flips], 1)            # (B, 1 + C, C)
        E = (do.predict(proj, pred, ht[b], cand) - ht1[b, None]).pow(2).mean(-1)
        out.append(E[:, :1] - E[:, 1:])
    return torch.cat(out)


def ctx_auroc(delta, split, tau):
    ht, ht1, yt, at = split
    logit = lambda p: torch.log(p) - torch.log1p(-p)
    shift = 0 if tau is None else delta / tau
    p_ons = torch.sigmoid(logit(do.PRIOR["ons"]) + shift)
    p_res = torch.sigmoid(logit(do.PRIOR["res"]) + shift)
    return do.macro_auroc(yt * (1 - p_res) + (1 - yt) * p_ons, (yt + at).clamp(0, 1))


def main():
    cem = do.load_script("cem_planning")
    data, paths = do.build_data()
    counts = cem._transition_counts(paths["lance_path"], paths["pairs_path"], ("cross_stay",), 76)
    p_ons, p_res = cem._priors(counts, "base_rate")
    do.PRIOR["ons"], do.PRIOR["res"] = torch.from_numpy(p_ons).to(DEV), torch.from_numpy(p_res).to(DEV)
    results = {}
    for path in sorted((do.OUT / "heads").glob("*.pt")):
        enc_key, name = path.stem.split("__")
        enc = enc_key.replace("_", "/", 1)
        d = data[enc]
        D = d["train"][0].shape[1]
        sd = torch.load(path, weights_only=False)
        proj, pred = ActionProjector(76, D).to(DEV), DynamicsPredictor(D, 512).to(DEV)
        proj.load_state_dict(sd["proj"])
        pred.load_state_dict(sd["pred"])
        proj.eval(), pred.eval()
        val = {tau: ctx_auroc(deltas(proj, pred, d["val"]), d["val"], tau) for tau in TAUS}
        tau = max(val, key=val.get)
        test_d, none_d = deltas(proj, pred, d["test"]), deltas(proj, pred, d["test"], "nochange")
        r = {"tau": tau, "val": val[tau], "test": ctx_auroc(test_d, d["test"], tau),
             "test_nochange": ctx_auroc(none_d, d["test"], tau), "test_prior_only": ctx_auroc(test_d, d["test"], None)}
        results.setdefault(enc, {})[name.replace("_", " ")] = r
        print(f"{enc:<14} {name.replace('_', ' '):<11} tau {tau:<7g} val {r['val']:.4f} | test {r['test']:.4f} "
              f"(nothing changed {r['test_nochange']:.4f}, prior only {r['test_prior_only']:.4f})", flush=True)
    with open(do.OUT / "flip_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
