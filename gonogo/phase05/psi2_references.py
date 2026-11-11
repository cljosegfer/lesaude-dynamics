"""
Phase 0.5a References: Debiased Effect Norms in Noise-Floor Units (folds 0-9 only)
=================================================================================
psi^2 of the five positive controls for the representations h, b and [h; b] (go_no_go_plan.md, 0.5a, revision 7:
h and [h; b] are references only, b is eligible), on folds 0-9 only, the selection half. Nothing on folds 10-19 is
read, and the choosing rule is not applied (it needs the encoder g).

Input: fresh gonogo/part1/gain.py runs with --save-residuals, Z = z, state s_t = [b, h, m] (the current h), targets
dh and db, for the clean subset (primary) and the adjusted subset (power check):
  outputs/phase05/4b_references/gain_{clean,adjusted}/<subset>_bhm_{h,b}_z_<code>.resid.npz
Per code and target, the R-learner effect theta = sum (a - pi) r / sum (a - pi)^2 (in z-units of the analysis set,
times the saved sd for raw units), from folds 0-4 (theta_A) and from folds 5-9 (theta_B), which share no patients.

Statistic: psi^2 = theta_A' Sigma^-1 theta_B in raw units, which equals the same product in any z-units, so it does
not depend on how each run standardised its target. Sigma is the noise floor: the covariance of the raw change
(target(t+1) - target(t)) over the stable usable between-episode pairs of all folds (no cardiac code change; b and m
usable; not same-episode), with a ridge of 1e-6 x mean eigenvalue. For [h; b] the effects and Sigma are concatenated,
cross-covariances included. One unit is an effect as large as the stable-pair change along its own direction.
Uncertainty: patient-level bootstrap, 1,000 multinomial resamples (gonogo/part1/effects.py), percentile 95% CI.

Output: outputs/phase05/4b_references/{results.json, table.md}

Example
-------
python -I gonogo/phase05/psi2_references.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from linkage import load_z  # noqa: E402
from pairs import OUT as P1, POSITIVE, load as load_pairs  # noqa: E402

D = ROOT / "outputs" / "phase05" / "4b_references"
N_BOOT, SEED, H_TAG = 1000, 0, "zeroactions"
QA, QB = range(0, 5), range(5, 10)
REPS = {"h": ["h"], "b": ["b"], "[h; b]": ["h", "b"]}


def noise_floor():
    """Raw stable-pair change of each target, (n_stable, dims), on usable between-episode pairs of all folds."""
    pairs, _, _, _, b_lookup, b, b_names = load_pairs()
    z = load_z()
    mz = np.load(P1 / "m.npz")
    m_lookup = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][m_lookup.loc[pairs["idx_t"].values].values] & mz["has_m"][m_lookup.loc[pairs["idx_t1"].values].values]
    keep = pairs["has_b"].values & has_m & ~z["same_episode"].values & pairs["stable"].values
    P = pairs[keep]
    hz = np.load(P1 / f"h_{H_TAG}.npz")
    h_lookup = pd.Series(np.arange(len(hz["rows"])), index=hz["rows"])
    dh = hz["h"][h_lookup.loc[P["idx_t1"].values].values] - hz["h"][h_lookup.loc[P["idx_t"].values].values]
    db = b[b_lookup.loc[P["idx_t1"].values].values] - b[b_lookup.loc[P["idx_t"].values].values]
    return {"h": dh.astype(np.float64), "b": db.astype(np.float64)}, int(keep.sum())


def precision(deltas, keep=0.999):
    """Two metrics. 'full': Sigma^-1 with a 1e-6 ridge, as the plan writes it. 'pc': the pseudo-inverse of Sigma on its
    principal subspace holding `keep` of the stable-pair variance. The ViT's change lives in about 42 of 256 directions
    (condition number about 6e7), so the full inverse multiplies estimation noise in near-flat directions by up to 1e7;
    the pc metric measures effects only where the noise floor is not degenerate. Also returns the subspace size."""
    S = np.cov(np.concatenate(deltas, 1), rowvar=False)
    full = np.linalg.inv(S + np.eye(len(S)) * 1e-6 * np.trace(S) / len(S))
    ev, V = np.linalg.eigh(S)
    ev, V = ev[::-1], V[:, ::-1]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), keep) + 1)
    pc = (V[:, :k] / ev[:k]) @ V[:, :k].T
    return {"full": full, "pc": pc}, k, float(ev[0] / max(ev[-1], 1e-30))


def per_patient(f):
    """Patient sums of (a - pi) r (P, K) and (a - pi)^2 (P,), patient folds, onsets per fold, raw sd."""
    zz = np.load(f)
    subj, folds = zz["subj"], zz["folds"].astype(int)
    u, inv = np.unique(subj, return_inverse=True)
    ra, ry = zz["ra"].astype(np.float64), zz["ry"].astype(np.float64)
    num = np.zeros((len(u), ry.shape[1]))
    np.add.at(num, inv, ra[:, None] * ry)
    den = np.bincount(inv, weights=ra ** 2, minlength=len(u))
    pfold = np.zeros(len(u), int)
    pfold[inv] = folds
    return u, num, den, pfold, zz["sd"].astype(np.float64)


def main():
    deltas, n_stable = noise_floor()
    print(f"noise floor: {n_stable:,} stable usable between-episode pairs; dims h {deltas['h'].shape[1]}, "
          f"b {deltas['b'].shape[1]}", flush=True)
    prec, spaces = {}, {}
    for rep, ts in REPS.items():
        mats, k, cond = precision([deltas[t] for t in ts])
        prec[rep], spaces[rep] = mats, {"dims": int(sum(deltas[t].shape[1] for t in ts)), "pc_dims_99.9": k,
                                        "condition_number": float(f"{cond:.3g}")}
        print(f"{rep}: dims {spaces[rep]['dims']}, principal subspace (99.9% of stable-pair variance) {k}, "
              f"condition number {cond:.3g}", flush=True)
    results, lines = {}, []
    for subset in ("clean", "adjusted"):
        for code in POSITIVE:
            files = {t: D / f"gain_{subset}" / f"{subset}_bhm_{t}_z_{code}.resid.npz" for t in ("h", "b")}
            if not all(f.exists() for f in files.values()):
                print(f"{subset} {code}: residual files missing, skipped", flush=True)
                continue
            parts = {t: per_patient(f) for t, f in files.items()}
            rows = {t: np.load(D / f"gain_{subset}" / f"{subset}_bhm_{t}_z_{code}.pairs.npz")["rows"] for t in parts}
            assert (rows["h"] == rows["b"]).all(), "h and b runs on different pairs"
            u, _, _, pfold, _ = parts["h"]
            assert (parts["b"][0] == u).all()
            inA, inB = np.isin(pfold, list(QA)), np.isin(pfold, list(QB))
            pp = np.load(D / f"gain_{subset}" / f"{subset}_bhm_h_z_{code}.pairs.npz")
            fold_of_pair = np.load(files["h"])["folds"].astype(int)
            a = pp["a"].astype(bool)
            n_on = {"q0-4": int((a & np.isin(fold_of_pair, list(QA))).sum()),
                    "q5-9": int((a & np.isin(fold_of_pair, list(QB))).sum())}
            Pn = len(u)
            W = np.random.default_rng(SEED).multinomial(Pn, np.full(Pn, 1 / Pn), size=N_BOOT).astype(np.float64)
            for rep, ts in REPS.items():
                tA, tB, bA, bB = [], [], [], []
                for t in ts:
                    _, num, den, _, sd = parts[t]
                    tA.append(num[inA].sum(0) / den[inA].sum() * sd)
                    tB.append(num[inB].sum(0) / den[inB].sum() * sd)
                    bA.append((W[:, inA] @ num[inA]) / (W[:, inA] @ den[inA])[:, None] * sd)
                    bB.append((W[:, inB] @ num[inB]) / (W[:, inB] @ den[inB])[:, None] * sd)
                tA, tB = np.concatenate(tA), np.concatenate(tB)
                bA, bB = np.concatenate(bA, 1), np.concatenate(bB, 1)
                for metric, M in prec[rep].items():
                    point = float(tA @ M @ tB)
                    boot = np.einsum("bi,ij,bj->b", bA, M, bB)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    results.setdefault(metric, {}).setdefault(subset, {}).setdefault(rep, {})[code] = {
                        "psi2": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                        "lower_above_0": bool(lo > 0), "onsets_folds_0_4": n_on["q0-4"],
                        "onsets_folds_5_9": n_on["q5-9"]}
            print(f"{subset} {code}: onsets {n_on}  " + "  ".join(
                f"{rep} {m} {results[m][subset][rep][code]['psi2']:.3f} [{results[m][subset][rep][code]['ci95'][0]:.3f}, "
                f"{results[m][subset][rep][code]['ci95'][1]:.3f}]" for m in ("full", "pc") for rep in REPS), flush=True)
    for metric in results:
        for subset, reps in results[metric].items():
            for rep, codes in reps.items():
                vals = [v["psi2"] for v in codes.values()]
                reps[rep]["_summary"] = {"n_lower_above_0": int(sum(v["lower_above_0"] for v in codes.values())),
                                         "median_psi2": round(float(np.median(vals)), 4), "n_codes": len(vals)}
    out = {"folds": "0-9 only (quarters 0-4 vs 5-9); folds 10-19 not read",
           "units": "noise floor: 'full' = Sigma^-1 (as written; ill-conditioned for h), 'pc' = pseudo-inverse on the "
                    "principal subspace holding 99.9% of the stable-pair variance (proposed, revision 7)",
           "noise_floor_pairs": n_stable, "spaces": spaces,
           "state": "s_t = [b, h, m], h = no-action ViT (zeroactions)", "Z": "z", "rule_applied": False,
           "results": results}
    (D / "results.json").write_text(json.dumps(out, indent=1))
    names = {"pc": "proposed metric: principal subspace, 99.9% of stable-pair variance",
             "full": "as written: full Σ⁻¹ (ill-conditioned for h; not interpretable for h and [h; b])"}
    for metric in ("pc", "full"):
        for subset in results[metric]:
            r = results[metric][subset]
            lines.append(f"\n**{subset} subset, {names[metric]}** (onsets folds 0-4 / 5-9; ψ̂² [95% CI])\n")
            lines.append("| Code | Onsets | h (reference) | [h; b] (reference) | b (eligible) |\n|---|---|---|---|---|")
            for code in POSITIVE:
                if code not in r["h"]:
                    continue
                e = r["h"][code]
                cells = [f"{r[x][code]['psi2']:.3f} [{r[x][code]['ci95'][0]:.3f}, {r[x][code]['ci95'][1]:.3f}]"
                         + ("*" if r[x][code]["lower_above_0"] else "") for x in ("h", "[h; b]", "b")]
                lines.append(f"| {code} | {e['onsets_folds_0_4']} / {e['onsets_folds_5_9']} | " + " | ".join(cells) + " |")
            s = {x: r[x]["_summary"] for x in ("h", "[h; b]", "b")}
            lines.append("| lower bound > 0 (descriptive) | | " + " | ".join(
                f"{s[x]['n_lower_above_0']} of {s[x]['n_codes']}" for x in ("h", "[h; b]", "b")) + " |")
            lines.append("| median ψ̂² | | " + " | ".join(f"{s[x]['median_psi2']:.3f}" for x in ("h", "[h; b]", "b")) + " |")
    head = "Subspace sizes: " + "; ".join(f"{rep} {v['dims']} dims, {v['pc_dims_99.9']} in the 99.9% subspace, "
                                          f"condition number {v['condition_number']:.3g}" for rep, v in spaces.items())
    (D / "table.md").write_text(head + "\n" + "\n".join(lines) + "\n\n`*` lower bound above 0. The 0.5a choosing rule "
                                "is not applied until g exists.\n")
    print(head + "\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
