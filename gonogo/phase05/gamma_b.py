"""
Phase 0.5c (Step 3e) on b: Γ̂ by Rotation
=========================================
The random-effects margin of go_no_go_plan.md 0.5c (revision 6, metric of revision 8) on the chosen representation b,
from gonogo/phase05/run_b.py's results (clean subset, state [b, m], target Δb).

Withheld codes  every rotation code n of group g, run under z + rot<g> (deconfounded with z and the other four groups;
                outputs/phase05/9_3e_b/gain). Statistic: debiased ψ̂²_n, θ from folds 0-9 against folds 10-19 (an
                all-fold statistic), Σ = Ledoit-Wolf on the standardised stable-pair changes of b over all folds (the
                primary metric); v_n = the variance of its 1,000 patient-bootstrap draws. The residual excess gain
                (R-learner E after the CRT null mean) is also reported.
Γ̂               DerSimonian-Laird: w_n = 1/v_n, μ̂_FE, Q, τ̂² = max(0, (Q - (N-1)) / (Σw - Σw²/Σw)); μ̂ with weights
                1/(v_n + τ̂²); Γ̂² = μ̂ + 1.645 τ̂, or μ̂ + 1.645 SE(μ̂) when τ̂ = 0. Codes whose ψ̂² or v_n is not finite (no
                onset in a half) are dropped and counted. Secondary margins within the "both" and "admission" profiles.
Tested codes    under z + A (A = C0 without the validation subset, the code under test removed; 8_3d_b/gain):
                positive controls on folds 10-19 (10-14 vs 15-19, Σ from folds 10-19), ambiguous codes and C1 controls
                on all folds (0-9 vs 10-19). A code clears the margin when its ψ̂² lower bound exceeds Γ̂². Sensitivity
                metrics (pseudo-inverse, full inverse) are reported for the tested codes.
Rule (0.5c)     usable: >= 2 positive controls clear and no C1 control clears; too little power: no positive control
                clears; margin too small: a C1 control clears.

Output: outputs/phase05/9_3e_b/{gamma.json, tables.md}

Example
-------
python -I gonogo/phase05/gamma_b.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "gonogo" / "phase05"))
from pairs import POSITIVE, load as load_pairs  # noqa: E402
from psi2 import metrics, patient_sums, stable_changes, theta  # noqa: E402

P05 = ROOT / "outputs" / "phase05"
FROZEN = P05 / "freeze" / "label_sets.json"
OUTD = P05 / "9_3e_b"
N_BOOT = 1000
AMBIGUOUS = ["I10", "I12", "I25", "I27", "I50", "I69", "I73", "I85", "I95"]


def psi2_code(path, fold_a, fold_b, M):
    """ψ̂², its bootstrap draws (for every metric in M) and onsets per half, from a gain.py residual file."""
    p = patient_sums(path, {"A": fold_a, "B": fold_b})
    mA, mB = p["member"]["A"], p["member"]["B"]
    if p["den"][mA].sum() == 0 or p["den"][mB].sum() == 0:
        return None
    W = np.random.default_rng(0).multinomial(len(p["u"]), np.full(len(p["u"]), 1 / len(p["u"])),
                                             size=N_BOOT).astype(np.float64)
    tA, tB = theta(p, mA), theta(p, mB)
    with np.errstate(invalid="ignore", divide="ignore"):
        bA, bB = theta(p, mA, W), theta(p, mB, W)
    pp = np.load(str(path).replace(".resid.npz", ".pairs.npz"))
    a = pp["a"].astype(bool)
    out = {"onsets": [int((a & np.isin(p["folds"], fold_a)).sum()), int((a & np.isin(p["folds"], fold_b)).sum())]}
    for m, P in M.items():
        boot = np.einsum("bi,ij,bj->b", bA, P, bB)
        boot = boot[np.isfinite(boot)]
        out[m] = {"psi2": float(tA @ P @ tB), "ci95": np.percentile(boot, [2.5, 97.5]).tolist() if len(boot) else [np.nan] * 2,
                  "var": float(boot.var()) if len(boot) > 1 else np.nan, "n_boot_finite": int(len(boot))}
    return out


def dersimonian_laird(y, v):
    w = 1 / v
    mu_fe = (w * y).sum() / w.sum()
    Q = (w * (y - mu_fe) ** 2).sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    se = ws.sum() ** -0.5
    tau = tau2 ** 0.5
    g2 = mu + 1.645 * tau if tau > 0 else mu + 1.645 * se
    return {"N": int(len(y)), "mu": float(mu), "se_mu": float(se), "tau2": float(tau2), "tau": float(tau),
            "Q": float(Q), "mu_fixed_effect": float(mu_fe), "Gamma2": float(g2),
            "Gamma2_rule": "mu + 1.645 tau" if tau > 0 else "mu + 1.645 SE(mu) (tau = 0)"}


def cardiac_profile(codes):
    """'both' if >= 25 clean onsets go into ED-only stays (roles.py's rule), else 'admission'."""
    import pandas as pd
    pairs, yt, yt1, vocab, _, _, _ = load_pairs()
    z = pd.read_parquet(ROOT / "outputs/gonogo_part1/z.parquet", columns=["setting_t1", "same_episode"])
    mz = np.load(ROOT / "outputs/gonogo_part1/m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    usable = pairs["has_b"].values & has_m & ~z["same_episode"].values
    out = {}
    for c in codes:
        j = vocab.index(c)
        onset = (yt[:, j] == 0) & (yt1[:, j] == 1)
        clean = pairs["stable"].values | ((pairs["single_code"].values == j) & (pairs["single_dir"].values == 1))
        n = int((usable & clean & onset & (z["setting_t1"].values == "ed_only")).sum())
        out[c] = {"onsets_into_ed_only": n, "profile": "both" if n >= 25 else "admission"}
    return out


def main():
    fz = json.loads(FROZEN.read_text())
    all_folds, f_lo, f_hi = list(range(20)), list(range(10)), list(range(10, 20))
    d_all, _, n_all = stable_changes(all_folds, [])
    d_hi, _, n_hi = stable_changes(f_hi, [])
    M_all, diag_all = metrics(d_all["b"])
    M_hi, diag_hi = metrics(d_hi["b"])
    # Withheld codes
    withheld, dropped = {}, []
    for g, codes in fz["rotation_groups"].items():
        for c in codes:
            f = OUTD / "gain" / f"clean_bm_b_z+rot{g}_{c}.resid.npz"
            if not f.exists():
                dropped.append((c, "no result"))
                continue
            r = psi2_code(f, f_lo, f_hi, {"lw": M_all["lw"]})
            if r is None or not np.isfinite(r["lw"]["psi2"]) or not np.isfinite(r["lw"]["var"]) or r["lw"]["var"] <= 0:
                dropped.append((c, "no onset in a half or degenerate bootstrap"))
                continue
            meta = json.loads(f.with_name(f.name.replace(".resid.npz", ".json")).read_text())
            withheld[c] = {"group": int(g), "profile": fz["C0"][c]["profile"], "onsets": r["onsets"],
                           "psi2": r["lw"]["psi2"], "ci95": r["lw"]["ci95"], "var": r["lw"]["var"],
                           "E_x1e5": (meta["rlearner"]["G"] - meta["rlearner"]["crt"]["mean"]) * 1e5,
                           "p_crt": meta["rlearner"]["crt"]["p"]}
    y = np.array([w["psi2"] for w in withheld.values()])
    v = np.array([w["var"] for w in withheld.values()])
    pooled = dersimonian_laird(y, v)
    by_profile = {}
    for prof in ("both", "admission"):
        k = np.array([w["profile"] == prof for w in withheld.values()])
        by_profile[prof] = dersimonian_laird(y[k], v[k]) if k.sum() >= 3 else {"N": int(k.sum()), "note": "too few"}
    by_group = {g: dersimonian_laird(y[[w["group"] == int(g) for w in withheld.values()]],
                                     v[[w["group"] == int(g) for w in withheld.values()]])
                for g in fz["rotation_groups"]}
    # Tested codes
    tested_codes = POSITIVE + AMBIGUOUS + list(fz["validation_c1"])
    prof = cardiac_profile(POSITIVE + AMBIGUOUS)
    prof.update({c: {"profile": fz["C0"][c]["profile"]} for c in fz["validation_c1"]})
    tested, missing = {}, []
    for c in tested_codes:
        f = P05 / "8_3d_b" / "gain" / f"clean_bm_b_z+A_{c}.resid.npz"
        if not f.exists():
            missing.append(c)
            continue
        pos = c in POSITIVE
        r = psi2_code(f, list(range(10, 15)) if pos else f_lo, list(range(15, 20)) if pos else f_hi,
                      M_hi if pos else M_all)
        if r is None:
            missing.append(c)
            continue
        margin_prof = by_profile.get(prof[c]["profile"], {}).get("Gamma2", np.nan)
        tested[c] = {"role": "positive" if pos else ("C1 control" if c in fz["validation_c1"] else "ambiguous"),
                     "folds": "10-14 vs 15-19" if pos else "0-9 vs 10-19", "profile": prof[c]["profile"],
                     "onsets": r["onsets"], **{m: r[m] for m in ("lw", "pc", "full")},
                     "clears": bool(r["lw"]["ci95"][0] > pooled["Gamma2"]),
                     "clears_profile_margin": bool(r["lw"]["ci95"][0] > margin_prof)}
    n_pos = sum(t["clears"] for t in tested.values() if t["role"] == "positive")
    n_c1 = sum(t["clears"] for t in tested.values() if t["role"] == "C1 control")
    if n_c1 > 0:
        decision = "margin too small (a C1 control clears): revisit the anchor rule before Part 2"
    elif n_pos >= 2:
        decision = "usable (>= 2 positive controls clear, no C1 control clears)"
    elif n_pos == 0:
        decision = "too little power (no positive control clears): real codes inconclusive, the paper rests on the benchmark"
    else:   # reading fixed by the user on 10 Nov, after the 90-code preview and before the final result (plan 0.5c)
        decision = ("too little power (exactly one positive control clears and no C1 control does: read as too "
                    "little power, since it fails the usable threshold of 2 without meeting margin too small)")
    out = {"metric": "Ledoit-Wolf (primary)", "noise_floor": {"all folds": [n_all, diag_all], "folds 10-19": [n_hi, diag_hi]},
           "withheld": withheld, "dropped": dropped, "pooled": pooled, "by_profile": by_profile, "by_group": by_group,
           "tested": tested, "tested_missing": missing, "n_positive_clear": n_pos, "n_c1_clear": n_c1,
           "decision": decision}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "gamma.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"## Γ̂ by rotation on b (Ledoit-Wolf; {pooled['N']} withheld codes, {len(dropped)} dropped)", "",
         "| Margin | N | μ̂ [SE] | τ̂ | Γ̂² | Rule |", "|---|---|---|---|---|---|"]
    for name, r in [("pooled", pooled)] + [(f"profile {k}", v_) for k, v_ in by_profile.items()] + \
                   [(f"group {k} (check)", v_) for k, v_ in by_group.items()]:
        if "mu" in r:
            L.append(f"| {name} | {r['N']} | {r['mu']:.3f} [{r['se_mu']:.3f}] | {r['tau']:.3f} | {r['Gamma2']:.3f} | {r['Gamma2_rule']} |")
    L += ["", "## Tested codes against the pooled margin (ψ̂², Ledoit-Wolf; 95% CI)", "",
          "| Code | Role | Folds | Profile | Onsets A / B | ψ̂² [95% CI] | Clears Γ̂² | Clears profile margin | ψ̂² pseudo-inverse | ψ̂² full inverse |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, t in tested.items():
        L.append(f"| {c} | {t['role']} | {t['folds']} | {t['profile']} | {t['onsets'][0]} / {t['onsets'][1]} | "
                 f"{t['lw']['psi2']:.3f} [{t['lw']['ci95'][0]:.3f}, {t['lw']['ci95'][1]:.3f}] | "
                 f"{'yes' if t['clears'] else 'no'} | {'yes' if t['clears_profile_margin'] else 'no'} | "
                 f"{t['pc']['psi2']:.3f} | {t['full']['psi2']:.3f} |")
    L += ["", f"Positive controls clearing: {n_pos}; C1 controls clearing: {n_c1}. **0.5c decision: {decision}.**",
          f"Dropped withheld codes: {len(dropped)}; tested codes missing: {missing}."]
    (OUTD / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
