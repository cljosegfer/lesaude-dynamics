"""
Phase 0.5: Debiased Effect Norms psi^2 Under the Revision-8 Metric
==================================================================
psi^2 of the five positive controls for candidate representations (go_no_go_plan.md, 0.5a and 0.5c, revision 8),
from gonogo/part1/gain.py residual runs (--save-residuals, Z = z). A candidate is a list of parts, each a target of
gain.py from one residual folder and state; the parts of a concatenation must come from runs on the same analysis set.

Estimate. Per part, the R-learner effect theta = sum (a - pi) r / sum (a - pi)^2 on the pairs of fold set A and of
fold set B (disjoint patients), times the saved sd for raw units; psi^2 = theta_A' Sigma^-1 theta_B over the
concatenated parts. Patient-level bootstrap (1,000 multinomial resamples, Sigma held fixed), percentile 95% CI.

Sigma, the noise floor: the covariance of the raw change target(t+1) - target(t) over the stable usable between-episode
pairs (no cardiac code change; b and m usable; not same-episode) of the folds the statistic uses (A and B), per
representation, cross-covariances included. Three metrics, always reported together:
  lw    primary (revision 8): Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf, analytical intensity) of the
        per-dimension standardised changes, i.e. toward a scaled identity in standardised coordinates, which is the
        diagonal of the stable-pair variances in raw units. Standardising first makes the metric independent of each
        feature's units (b mixes bpm, ms and mV); it has no tuning parameter. The intensity is recorded.
  pc    sensitivity: the pseudo-inverse of the raw sample covariance on its principal subspace holding 99.9% of the
        stable-pair variance (revision 7's metric).
  full  sensitivity: the raw sample covariance's inverse with a 1e-6 ridge (the plan's original wording).
Decisions use lw only.

Per-feature readout (the validation readout on db; for I48's P absence on dm, the only representation that has it):
the standardised effect on the code's signature feature, theta_raw / SD of that feature's stable-pair change on the
same folds, on the pairs of A and B pooled, with a patient-bootstrap CI.

Output: <out>/results.json, <out>/table.md

Examples
--------
python -I gonogo/phase05/psi2.py --preset references-rev8 --out outputs/phase05/4b_rev8      # done 9 Nov
python -I gonogo/phase05/psi2.py --preset g-selection --out outputs/phase05/3c_selection     # 3c, once g exists
python -I gonogo/phase05/psi2.py --preset g-selection --fold-a 10-14 --fold-b 15-19 --out outputs/phase05/3c_report
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from linkage import load_z  # noqa: E402
from pairs import OUT as P1, POSITIVE, load as load_pairs  # noqa: E402

N_BOOT, SEED, KEEP = 1000, 0, 0.999
P05 = ROOT / "outputs" / "phase05"
SIGNATURE = {"I44": [("b", "qrs_width")], "I45": [("b", "qrs_width")], "I47": [("b", "hr")],
             "I48": [("m", "p_missing"), ("b", "hr")], "I49": [("b", "rr_irr")]}
# name -> {candidate: [(target, state, folder template, embedding tag of target h)]}; {subset} is filled in.
# A part with target "h" reads its residuals from a gain.py run with --h-tag <tag>; its noise floor uses h_<tag>.npz.
REF_H = ("h", "bhm", "4b_references/gain_{subset}", "zeroactions")
REF_HB = [REF_H, ("b", "bhm", "4b_references/gain_{subset}", None)]
B_ONLY = [("b", "bm", "4b_rev8/gain_bm_{subset}", None)]
PRESETS = {
    "references-rev8": {"h": [REF_H], "[h; b]": REF_HB, "b": B_ONLY},
    # After g exists (3c; do not run before): gain.py --h-tag single --state bhm --targets h,b --conds z
    # --save-residuals --codes positive --subset {clean,adjusted} --out-dir outputs/phase05/3c_g/gain_{subset}
    "g-selection": {"g": [("h", "bhm", "3c_g/gain_{subset}", "single")],
                    "[g; b]": [("h", "bhm", "3c_g/gain_{subset}", "single"), ("b", "bhm", "3c_g/gain_{subset}", None)],
                    "b": B_ONLY, "h": [REF_H], "[h; b]": REF_HB},
}
INTRA = ("h", "bhm", "7_intra_gain/gain_{subset}", "intra_last")
PRESETS["intra-exploratory"] = {      # plan revision 9: exploratory only, for the encoder ablation; b is chosen
    "[intra; b]": [INTRA, ("b", "bhm", "7_intra_gain/gain_{subset}", None)], "intra": [INTRA], "b": B_ONLY}
ROLES = {"g": "eligible", "[g; b]": "eligible", "b": "eligible", "h": "reference", "[h; b]": "reference",
         "intra": "exploratory", "[intra; b]": "exploratory"}
READOUT_FROM = {"b": ("bm", "4b_rev8/gain_bm_{subset}"), "m": ("bm", "4b_rev8/gain_bm_{subset}")}


def part_key(part):
    target, _, _, tag = part
    return f"h:{tag}" if target == "h" else target


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="references-rev8", choices=list(PRESETS))
    p.add_argument("--fold-a", default="0-4")
    p.add_argument("--fold-b", default="5-9")
    p.add_argument("--out", default=str(P05 / "4b_rev8"))
    return p.parse_args()


def folds_of(spec):
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi + 1))


def stable_changes(fold_set, h_tags):
    """Raw stable-pair change of b, m and of the embedding h_<tag> for each tag ("h:<tag>"), on the usable
    between-episode pairs of `fold_set`."""
    pairs, _, _, _, b_lookup, b, b_names = load_pairs()
    z = load_z()
    mz = np.load(P1 / "m.npz")
    m_lookup = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][m_lookup.loc[pairs["idx_t"].values].values] & mz["has_m"][m_lookup.loc[pairs["idx_t1"].values].values]
    keep = (pairs["has_b"].values & has_m & ~z["same_episode"].values & pairs["stable"].values
            & np.isin(pairs["fold"].values, fold_set))
    P = pairs[keep]
    take = lambda arr, lk: arr[lk.loc[P["idx_t1"].values].values].astype(np.float64) - arr[lk.loc[P["idx_t"].values].values]
    out = {"b": take(b, b_lookup), "m": take(mz["m"], m_lookup)}
    names = {"b": list(b_names), "m": [str(n) for n in mz["names"]]}
    for tag in sorted(set(h_tags)):
        hz = np.load(P1 / f"h_{tag}.npz")
        h_lookup = pd.Series(np.arange(len(hz["rows"])), index=hz["rows"])
        out[f"h:{tag}"] = take(hz["h"], h_lookup)
        names[f"h:{tag}"] = [f"h{i}" for i in range(hz["h"].shape[1])]
    return out, names, int(keep.sum())


def metrics(X):
    """Precision matrices of the three metrics for the raw changes X (n, p), plus diagnostics."""
    assert np.isfinite(X).all(), "non-finite stable-pair changes"
    sd = X.std(0)
    ok = sd > 0
    S = np.cov(X, rowvar=False)
    full = np.linalg.inv(S + np.eye(len(S)) * 1e-6 * np.trace(S) / len(S))
    ev, V = np.linalg.eigh(S)
    ev, V = ev[::-1], V[:, ::-1]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), KEEP) + 1)
    pc = (V[:, :k] / ev[:k]) @ V[:, :k].T
    lw = LedoitWolf().fit(X[:, ok] / sd[ok])
    lw_prec = np.zeros_like(S)
    Dinv = 1 / sd[ok]
    lw_prec[np.ix_(ok, ok)] = Dinv[:, None] * np.linalg.inv(lw.covariance_) * Dinv[None]
    diag = {"dims": int(X.shape[1]), "dims_constant": int((~ok).sum()), "pc_dims_99.9": k,
            "condition_number_sample": float(f"{ev[0] / max(ev[-1], 1e-300):.3g}"),
            "lw_shrinkage": round(float(lw.shrinkage_), 4),
            "condition_number_lw_standardised": float(f"{np.linalg.cond(lw.covariance_):.3g}")}
    return {"lw": lw_prec, "pc": pc, "full": full}, diag


def patient_sums(path, fold_sets):
    """Per patient: sum (a - pi) r (P, K), sum (a - pi)^2 (P,), fold-set membership; also sd and pair arrays."""
    zz = np.load(path)
    u, inv = np.unique(zz["subj"], return_inverse=True)
    ra, ry = zz["ra"].astype(np.float64), zz["ry"].astype(np.float64)
    num = np.zeros((len(u), ry.shape[1]))
    np.add.at(num, inv, ra[:, None] * ry)
    den = np.bincount(inv, weights=ra ** 2, minlength=len(u))
    pfold = np.zeros(len(u), int)
    pfold[inv] = zz["folds"].astype(int)
    member = {name: np.isin(pfold, fs) for name, fs in fold_sets.items()}
    return {"u": u, "num": num, "den": den, "member": member, "sd": zz["sd"].astype(np.float64),
            "folds": zz["folds"].astype(int)}


def theta(part, mask, W=None):
    if W is None:
        return part["num"][mask].sum(0) / part["den"][mask].sum() * part["sd"]
    return (W[:, mask] @ part["num"][mask]) / (W[:, mask] @ part["den"][mask])[:, None] * part["sd"]


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fA, fB = folds_of(args.fold_a), folds_of(args.fold_b)
    cands = PRESETS[args.preset]
    tags = [p[3] for parts in cands.values() for p in parts if p[0] == "h"]
    deltas, names, n_stable = stable_changes(fA + fB, tags)
    print(f"noise floor: {n_stable:,} stable usable between-episode pairs in folds {args.fold_a}, {args.fold_b}", flush=True)
    mets, diags = {}, {}
    for cand, parts in cands.items():
        X = np.concatenate([deltas[part_key(p)] for p in parts], 1)
        mets[cand], diags[cand] = metrics(X)
        print(f"{cand}: {diags[cand]}", flush=True)
    res = {m: {} for m in ("lw", "pc", "full")}
    readout, draws, fingerprints = {}, {}, {}
    for subset in ("clean", "adjusted"):
        for code in POSITIVE:
            for cand, parts in cands.items():
                files = [P05 / tmpl.format(subset=subset) / f"{subset}_{state}_{t}_z_{code}.resid.npz"
                         for t, state, tmpl, _ in parts]
                if not all(f.exists() for f in files):
                    print(f"{subset} {code} {cand}: residual files missing, skipped", flush=True)
                    continue
                ps = [patient_sums(f, {"A": fA, "B": fB}) for f in files]
                rows = [np.load(str(f).replace(".resid.npz", ".pairs.npz"))["rows"] for f in files]
                assert all((r == rows[0]).all() for r in rows) and all((p["u"] == ps[0]["u"]).all() for p in ps)
                mA, mB = ps[0]["member"]["A"], ps[0]["member"]["B"]
                pp = np.load(str(files[0]).replace(".resid.npz", ".pairs.npz"))
                a = pp["a"].astype(bool)
                n_on = [int((a & np.isin(ps[0]["folds"], fA)).sum()), int((a & np.isin(ps[0]["folds"], fB)).sum())]
                W = np.random.default_rng(SEED).multinomial(len(ps[0]["u"]), np.full(len(ps[0]["u"]), 1 / len(ps[0]["u"])),
                                                            size=N_BOOT).astype(np.float64)
                tA = np.concatenate([theta(p, mA) for p in ps])
                tB = np.concatenate([theta(p, mB) for p in ps])
                bA = np.concatenate([theta(p, mA, W) for p in ps], 1)
                bB = np.concatenate([theta(p, mB, W) for p in ps], 1)
                # Same seed and same patients give the same resamples across candidates of a code: paired draws
                fingerprints.setdefault(subset, {}).setdefault(code, {})[cand] = int(ps[0]["u"].sum() % (2 ** 61))
                for m, M in mets[cand].items():
                    point = float(tA @ M @ tB)
                    boot = np.einsum("bi,ij,bj->b", bA, M, bB)
                    draws[f"{m}|{subset}|{cand}|{code}"] = boot.astype(np.float32)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    res[m].setdefault(subset, {}).setdefault(cand, {})[code] = {
                        "psi2": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                        "lower_above_0": bool(lo > 0), "onsets": n_on}
            # Per-feature readout on the pooled folds A and B
            for target, feat in SIGNATURE[code]:
                state, tmpl = READOUT_FROM[target]
                f = P05 / tmpl.format(subset=subset) / f"{subset}_{state}_{target}_z_{code}.resid.npz"
                if not f.exists():
                    continue
                p = patient_sums(f, {"AB": fA + fB})
                j = names[target].index(feat)
                sd_stable = deltas[target][:, j].std()
                W = np.random.default_rng(SEED).multinomial(len(p["u"]), np.full(len(p["u"]), 1 / len(p["u"])),
                                                            size=N_BOOT).astype(np.float64)
                mk = p["member"]["AB"]
                t_raw = theta(p, mk)[j]
                b_raw = theta(p, mk, W)[:, j]
                lo, hi = np.percentile(b_raw / sd_stable, [2.5, 97.5])
                readout.setdefault(subset, {}).setdefault(code, {})[f"d{target} {feat}"] = ({
                    "target": f"d{target}", "feature": feat, "state": state, "effect_sd": round(float(t_raw / sd_stable), 3),
                    "ci95_sd": [round(float(lo), 3), round(float(hi), 3)], "effect_raw": round(float(t_raw), 4),
                    "sd_stable_change": round(float(sd_stable), 4)})
    summary = {m: {s: {c: {"n_lower_above_0": int(sum(v["lower_above_0"] for v in codes.values())),
                           "n_codes": len(codes), "median_psi2": round(float(np.median([v["psi2"] for v in codes.values()])), 4),
                           "qualifies_3_of_5": int(sum(v["lower_above_0"] for v in codes.values())) >= 3}
                       for c, codes in by_c.items()} for s, by_c in r.items()} for m, r in res.items()}
    disagree = {s: {c: {"lw": summary["lw"][s][c]["qualifies_3_of_5"], "pc": summary["pc"][s][c]["qualifies_3_of_5"]}
                    for c in summary["lw"][s] if summary["lw"][s][c]["qualifies_3_of_5"] != summary["pc"][s][c]["qualifies_3_of_5"]}
                for s in summary["lw"]}
    out = {"preset": args.preset, "folds": {"A": args.fold_a, "B": args.fold_b, "others": "not read"},
           "noise_floor_pairs": n_stable, "metrics": {"lw": "primary", "pc": "sensitivity", "full": "sensitivity"},
           "roles": ROLES, "diagnostics": diags, "results": res, "summary": summary,
           "qualification_disagreement_lw_vs_pc": disagree, "readout": readout, "choosing_rule_applied": False}
    paired = {s: {code: len(set(fp.values())) == 1 for code, fp in by_code.items()} for s, by_code in fingerprints.items()}
    out["bootstrap_draws"] = {"file": "boot_draws.npz", "keys": "metric|subset|candidate|code",
                              "paired_across_candidates": paired,
                              "use": "0.5a tie test: CI of the difference in median psi^2 between candidates, per draw"}
    np.savez_compressed(out_dir / "boot_draws.npz", **draws)
    (out_dir / "results.json").write_text(json.dumps(out, indent=1))
    L = [f"Folds {args.fold_a} vs {args.fold_b}; noise floor {n_stable:,} stable pairs of those folds. "
         "ψ̂² [95% CI]; `*` lower bound above 0. Decisions use lw.\n",
         "| Representation | Role | Dims | lw shrinkage | Dims in 99.9% subspace | Condition number (sample) |",
         "|---|---|---|---|---|---|"]
    L += [f"| {c} | {ROLES.get(c, '')} | {d['dims']} | {d['lw_shrinkage']:.3f} | {d['pc_dims_99.9']} | {d['condition_number_sample']:.3g} |"
          for c, d in diags.items()]
    label = {"lw": "Ledoit-Wolf (primary)", "pc": "99.9% principal-subspace pseudo-inverse (sensitivity)",
             "full": "full inverse (sensitivity)"}
    for subset in ("clean", "adjusted"):
        for m in ("lw", "pc", "full"):
            r = res[m].get(subset, {})
            if not r:
                continue
            L.append(f"\n**{subset} subset, {label[m]}**\n")
            L.append("| Code | Onsets A / B | " + " | ".join(f"{c} ({ROLES.get(c, '')})" for c in r) + " |")
            L.append("|---|---|" + "---|" * len(r))
            for code in POSITIVE:
                if not all(code in r[c] for c in r):
                    continue
                on = next(iter(r.values()))[code]["onsets"]
                cells = [f"{r[c][code]['psi2']:.3f} [{r[c][code]['ci95'][0]:.3f}, {r[c][code]['ci95'][1]:.3f}]"
                         + ("*" if r[c][code]["lower_above_0"] else "") for c in r]
                L.append(f"| {code} | {on[0]} / {on[1]} | " + " | ".join(cells) + " |")
            s = summary[m][subset]
            L.append("| Lower bound > 0 | | " + " | ".join(f"{s[c]['n_lower_above_0']} of {s[c]['n_codes']}" for c in r) + " |")
            L.append("| Median ψ̂² | | " + " | ".join(f"{s[c]['median_psi2']:.3f}" for c in r) + " |")
    L.append("\n**Per-feature readout** (standardised effect, SDs of the stable-pair change of the feature, folds "
             f"{args.fold_a} and {args.fold_b} pooled, state [b, m])\n")
    L.append("| Code | Feature | Clean: effect [95% CI] | Adjusted: effect [95% CI] |\n|---|---|---|---|")
    for code in POSITIVE:
        for target, feat in SIGNATURE[code]:
            key = f"d{target} {feat}"

            def cell(s):
                e = readout.get(s, {}).get(code, {}).get(key)
                return f"{e['effect_sd']:+.3f} [{e['ci95_sd'][0]:+.3f}, {e['ci95_sd'][1]:+.3f}]" if e else "n/a"
            L.append(f"| {code} | Δ{target} {feat} | {cell('clean')} | {cell('adjusted')} |")
    L.append(f"\nQualification (≥ 3 of 5 lower bounds > 0) disagreement between lw and pc: {disagree}. "
             "The choosing rule is not applied.")
    (out_dir / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
