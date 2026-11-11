"""
Phase 0.5a: Diagnosis of the Single-Recording Encoder g
=======================================================
Sequential checks of plan revision 9 (go_no_go_plan.md, 0.5a sanity rule). Nothing here computes psi^2.

probe     Offline frozen linear probe, identical for every encoder: embeddings of the Part 1 ECGs exported by
          gonogo/part1/embed_h.py (outputs/gonogo_part1/h_<tag>.npz), standardised with train-fold statistics; labels
          are each ECG's 76 cardiac codes (its stay's labels, as in the pair index); a linear layer trained with
          BCE (AdamW, lr 1e-3, wd 1e-4, batch 4096, seed 0) on patient folds 0-17, early-stopped on the macro AUROC of
          fold 18 (patience 5, at most 100 epochs), scored on fold 19 as macro AUROC over codes with both outcomes
          (the online probe's metric), with a patient-level bootstrap CI (200 resamples).
config    Export check (Step 1c): hparams of g's checkpoints against h's, backbone keys loaded with strict=False and
          the missing / unexpected keys listed, then strict=True, and how far the weights moved from initialisation.
aug       Augmentation check (Step 1d): one batch of training pairs, the sampled parameters of both views of
          scripts/pretrain.py's _single_views (shift, amplitude factor, noise level, mask start), verified by
          reproducing both views exactly from the logged parameters, and how much the two views differ.
random    Writes a randomly initialised checkpoint of g's exact architecture (build_backbone on g's hparams, seed 0),
          for the "learned nothing" floor through the same export path.

Outputs: outputs/phase05/5_diag_g/<command>.json

Examples
--------
python -I gonogo/phase05/diag_g.py probe --tags zeroactions single_last random_init
python -I gonogo/phase05/diag_g.py config
python -I gonogo/phase05/diag_g.py aug
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "anatomie"))
OUT = ROOT / "outputs" / "phase05" / "5_diag_g"
P1 = ROOT / "outputs" / "gonogo_part1"
G_RUN = ROOT / "runs/runs/20261103/165753/118441e564e0"
H_RUN = ROOT / "runs/runs/20261019/182223/646e2795c834"
G_CKPTS = {"last": G_RUN / "checkpoints/last.ckpt", "val_loss": G_RUN / "checkpoints/pretrain_single.ckpt",
           "online_probe_best": G_RUN / "checkpoints/pretrain_single_auroc.ckpt"}
H_CKPT = H_RUN / "checkpoints/pretrain_zeroactions.ckpt"


def ecg_table():
    """Per ECG of the pair index: labels (76 codes), patient fold and subject."""
    from pairs import load as load_pairs
    pairs, yt, yt1, _, _, _, _ = load_pairs()
    rows = np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values])
    lab = np.concatenate([yt, yt1]).astype(np.float32)
    fold = np.concatenate([pairs["fold"].values] * 2)
    subj = np.concatenate([pairs["subject_id"].values] * 2)
    u, first = np.unique(rows, return_index=True)
    return u, lab[first], fold[first], subj[first]


def macro_auroc(y, p, w=None):
    """Macro AUROC over codes with both outcomes (with weights: both outcomes carrying weight)."""
    keep = [k for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    vals = []
    for k in keep:
        if w is not None and not (0 < (w * y[:, k]).sum() < w.sum()):
            continue
        vals.append(roc_auc_score(y[:, k], p[:, k], sample_weight=w))
    return float(np.mean(vals)), len(keep)


def probe(tags, path=None):
    rows, lab, fold, subj = ecg_table()
    res = {}
    for tag in tags:
        z = np.load(P1 / f"h_{tag}.npz")
        assert np.array_equal(z["rows"], rows), f"{tag}: rows differ from the pair index"
        X = z["h"].astype(np.float32)
        tr, va, te = fold <= 17, fold == 18, fold == 19
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        dev = "cuda"
        torch.manual_seed(0)
        lin = torch.nn.Linear(X.shape[1], lab.shape[1]).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt, Yt = torch.tensor(Xs[tr], device=dev), torch.tensor(lab[tr], device=dev)
        Xv, Xe = torch.tensor(Xs[va], device=dev), torch.tensor(Xs[te], device=dev)
        best, best_state, since, ep_best = -1, None, 0, 0
        gen = torch.Generator(device=dev).manual_seed(0)
        for ep in range(100):
            perm = torch.randperm(len(Xt), device=dev, generator=gen)
            for i in range(0, len(perm), 4096):
                b = perm[i:i + 4096]
                loss = F.binary_cross_entropy_with_logits(lin(Xt[b]), Yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pv = torch.sigmoid(lin(Xv)).cpu().numpy()
            a, _ = macro_auroc(lab[va], pv)
            if a > best + 1e-4:
                best, ep_best, since = a, ep, 0
                best_state = {k: v.clone() for k, v in lin.state_dict().items()}
            else:
                since += 1
                if since >= 5:
                    break
        lin.load_state_dict(best_state)
        with torch.no_grad():
            pe = torch.sigmoid(lin(Xe)).cpu().numpy()
        auc, n_codes = macro_auroc(lab[te], pe)
        u, inv = np.unique(subj[te], return_inverse=True)
        rng = np.random.default_rng(0)
        boot = [macro_auroc(lab[te], pe, np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv])[0]
                for _ in range(200)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[tag] = {"test_macro_auroc": round(auc, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "val_macro_auroc": round(best, 4), "best_epoch": ep_best, "codes_scored": n_codes,
                    "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                    "embedding_std_mean": round(float(X.std(0).mean()), 5)}
        print(f"{tag}: test macro AUROC {auc:.4f} [{lo:.4f}, {hi:.4f}]  val {best:.4f} (epoch {ep_best})", flush=True)
    path = path or OUT / "probe.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))
    return res


def config():
    from models.backbones import build_backbone
    out = {}
    hh = torch.load(H_CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]
    torch.manual_seed(0)
    for name, path in G_CKPTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hp = ck.get("hyper_parameters") or {}
        sd = {k.removeprefix("backbone."): v for k, v in ck["state_dict"].items() if k.startswith("backbone.")}
        net = build_backbone(hp)
        init = {k: v.clone() for k, v in net.state_dict().items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        net.load_state_dict(sd, strict=True)
        moved = float(np.mean([float((sd[k].float() - init[k].float()).norm() / (init[k].float().norm() + 1e-12))
                               for k in init if init[k].dtype.is_floating_point]))
        keys = sorted(set(hp) | set(hh))
        diff = {k: [hh.get(k), hp.get(k)] for k in keys if hh.get(k) != hp.get(k)
                and not k.startswith(("slurm", "system"))}
        out[name] = {"path": str(path.relative_to(ROOT)), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
                     "missing_keys": list(missing), "unexpected_keys": list(unexpected), "strict_load": "ok",
                     "n_backbone_tensors": len(sd), "mean_relative_change_from_a_fresh_init": round(moved, 3),
                     "hparams_differing_from_h [h, g]": {k: [str(v[0]), str(v[1])] for k, v in diff.items()}}
        print(name, json.dumps(out[name], indent=1), flush=True)
    out["export_path"] = {
        "script": "gonogo/part1/embed_h.py -> anatomie/encoders.py load_encoder (build_backbone on the checkpoint's "
                  "hparams, load_state_dict strict) -> ViT1dLayers 'h' (the fc output)",
        "data": "the same Lance table (configs/data.yaml lance_path) and the same ECG rows (pair index)",
        "sampling_and_crop": "5000 samples (10 s at 500 Hz), no crop, as in training",
        "lead_order": "as stored: reshape (5000, 12) then transpose, the same in the pair (training) and single "
                      "(export) paths of src/dataset/dataset.py",
        "normalisation": "none in either path (the per-channel z-score in _getitems_single is commented out); raw "
                         "float16 -> float32, the same for g and h"}
    (OUT / "config.json").write_text(json.dumps(out, indent=1))


def aug():
    import pretrain as p
    from dataset.dataset import MIMICLanceDataset
    from torch.utils.data import DataLoader
    import yaml
    data = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    ds = MIMICLanceDataset(data["lance_path"], split="train", mode="pair", pairs_path=data["pairs_path"],
                           pair_types=("cross_stay",))
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0), num_workers=0)
    x = next(iter(loader))["xt"].cuda()
    aug = {**p.SINGLE_AUG, "scale": 0.05}
    B, L, T = x.shape

    def params_and_view(seed):
        torch.manual_seed(seed)
        dev = x.device
        shift = ((torch.rand(B, device=dev) * 2 - 1) * aug["shift"]).round().long()
        scale = 1 + (torch.rand(B, 1, 1, device=dev) * 2 - 1) * aug["scale"]
        noise = torch.randn(x.shape, device=dev) * aug["noise"] * x.std(dim=2, keepdim=True)
        start = (torch.rand(B, device=dev) * (T - aug["mask"])).long()
        torch.manual_seed(seed)
        view = p._single_views(x, aug)
        src = torch.arange(T, device=dev).unsqueeze(0) - shift.unsqueeze(1)
        valid = ((src >= 0) & (src < T)).unsqueeze(1)
        v = torch.gather(x, 2, src.clamp(0, T - 1).unsqueeze(1).expand(B, L, T)) * valid * scale + noise
        t = torch.arange(T, device=dev).unsqueeze(0)
        v = v * ~((t >= start.unsqueeze(1)) & (t < start.unsqueeze(1) + aug["mask"])).unsqueeze(1)
        assert torch.allclose(v, view, atol=1e-5), "logged parameters do not reproduce _single_views"
        return {"shift_samples": shift.cpu().numpy(), "amplitude_factor": scale.flatten().cpu().numpy(),
                "noise_sd_over_lead_sd": aug["noise"], "mask_start_samples": start.cpu().numpy()}, view

    p1, v1 = params_and_view(1)
    p2, v2 = params_and_view(2)
    rel = ((v1 - v2).norm(dim=(1, 2)) / v1.norm(dim=(1, 2))).cpu().numpy()
    out = {"batch": B, "aug": aug,
           "view1_first8": {k: (v[:8].tolist() if isinstance(v, np.ndarray) else v) for k, v in p1.items()},
           "view2_first8": {k: (v[:8].tolist() if isinstance(v, np.ndarray) else v) for k, v in p2.items()},
           "shift_range_view1": [int(p1["shift_samples"].min()), int(p1["shift_samples"].max())],
           "amplitude_range_view1": [float(p1["amplitude_factor"].min()), float(p1["amplitude_factor"].max())],
           "views_differ_all": bool((rel > 0).all()),
           "relative_difference_v1_v2": {"min": float(rel.min()), "median": float(np.median(rel)), "max": float(rel.max())},
           "same_shift_both_views": int((p1["shift_samples"] == p2["shift_samples"]).sum()),
           "verified_by_exact_reproduction": True}
    print(json.dumps(out, indent=1))
    (OUT / "aug.json").write_text(json.dumps(out, indent=1))


def random_ckpt():
    from models.backbones import build_backbone
    ck = torch.load(G_CKPTS["last"], map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    torch.manual_seed(0)
    net = build_backbone(hp)
    path = OUT / "random_init.ckpt"
    torch.save({"state_dict": {f"backbone.{k}": v for k, v in net.state_dict().items()}, "hyper_parameters": hp}, path)
    print(f"saved {path}")


@torch.no_grad()
def shortcut(ckpts, n=1000, path=None):
    """Step 3 (b)-(d) on held-out fold 19: recording-ID retrieval (two augmented views of each of n recordings),
    same-patient retrieval (two recordings of different stays of each of n patients, clean), effective rank and
    per-dimension std of the clean embeddings."""
    import pretrain as p
    from encoders import load_encoder
    from pairs import load as load_pairs
    import lance
    import yaml
    pairs, _, _, _, _, _, _ = load_pairs()
    te = pairs[pairs["fold"] == 19]
    rng = np.random.default_rng(0)
    one = te.groupby("subject_id").head(1)                                  # one cross-stay pair per patient
    pick = one.iloc[rng.permutation(len(one))[:n]]
    rows_a, rows_b = pick["idx_t"].values, pick["idx_t1"].values
    ds = lance.dataset(yaml.safe_load(open(ROOT / "configs/data.yaml"))["lance_path"])

    def wave(rows):
        t = ds.take([int(r) for r in rows], columns=["waveform"])
        w = t.column("waveform").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(rows), 5000, 12)
        return torch.from_numpy(w.astype(np.float32)).permute(0, 2, 1).contiguous()

    xa, xb = wave(rows_a), wave(rows_b)
    out = {"n": int(len(rows_a)), "fold": 19, "aug": p.SINGLE_AUG,
           "items": {"rows_a": [int(r) for r in rows_a], "rows_b": [int(r) for r in rows_b]}}

    def emb(enc, x):
        return torch.cat([enc(x[i:i + 250].cuda())["h"].float() for i in range(0, len(x), 250)])

    def top1(q, k):
        q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
        return float(((q @ k.T).argmax(1) == torch.arange(len(q), device=q.device)).float().mean())

    def eff_rank(e):
        s = torch.linalg.svdvals(e - e.mean(0))
        pr = s / s.sum()
        return float(torch.exp(-(pr * torch.log(pr + 1e-12)).sum()))

    for name, ck in ckpts.items():
        enc = load_encoder(ck)
        torch.manual_seed(1)
        v1 = torch.cat([p._single_views(xa[i:i + 250].cuda(), p.SINGLE_AUG) for i in range(0, len(xa), 250)])
        torch.manual_seed(2)
        v2 = torch.cat([p._single_views(xa[i:i + 250].cuda(), p.SINGLE_AUG) for i in range(0, len(xa), 250)])
        e1, e2 = emb(enc, v1.cpu()), emb(enc, v2.cpu())
        ea, eb = emb(enc, xa), emb(enc, xb)
        std = ea.std(0)
        out[name] = {"recording_id_top1": round(top1(e1, e2), 4), "same_patient_top1": round(top1(ea, eb), 4),
                     "chance_top1": round(1 / len(xa), 4), "effective_rank": round(eff_rank(ea), 2),
                     "dims": int(ea.shape[1]), "per_dim_std": {"min": float(std.min()), "median": float(std.median()),
                                                               "max": float(std.max())},
                     "dims_with_std_below_1e-3_of_max": int((std < 1e-3 * std.max()).sum())}
        print(name, json.dumps(out[name]), flush=True)
    (path or OUT / "shortcut.json").write_text(json.dumps(out, indent=1))
    return out


def losses():
    """Step 3 (a): prediction (invariance) and SIGReg loss trajectories of g and h, per epoch."""
    import pandas as pd
    out = {}
    for name, run in (("g", G_RUN), ("h", H_RUN)):
        d = pd.read_csv(run / "metrics.csv")
        cols = [c for c in ("train/pred_loss_epoch", "val/pred_loss", "train/reg_loss_epoch", "val/reg_loss",
                            "train/loss_epoch", "val/loss") if c in d]
        e = d.groupby("epoch")[cols].last()
        tr = e["val/pred_loss"].dropna()
        final = tr.iloc[-20:].mean()
        reach = int(tr.index[(tr <= final * 1.10).values.argmax()])
        out[name] = {"epochs": int(e.index.max()) + 1, "val_pred_loss": {str(int(k)): round(float(v), 5) for k, v in
                                                                          tr.iloc[[0, 1, 2, 5, 10, 25, 50, 100, 200, 300, -1]].items()},
                     "val_pred_loss_final_mean_last20": round(float(final), 5),
                     "first_epoch_within_10pct_of_final": reach,
                     "val_reg_loss_last": round(float(e["val/reg_loss"].dropna().iloc[-1]), 3)}
        print(name, json.dumps(out[name]), flush=True)
    (OUT / "losses.json").write_text(json.dumps(out, indent=1))


BAR = 0.529            # frozen sanity bar (plan revision 9): h's offline probe 0.579 minus 0.05, point estimate
H_PROBE = 0.579        # h's frozen offline probe (5_diag_g/probe.json, epoch-339 export); h is not re-exported


def report(run_dir, tag, out_dir):
    """Post-training report on a candidate encoder's LAST checkpoint (plan revision 9; not checkpoint selection):
    export through embed_h.py if needed, the offline clean probe against the frozen bar, recording-ID and same-patient
    retrieval and effective rank on the same 1,000 fold-19 items as the diagnosis of g (Step 3; h is recomputed on
    them as an identity check against 5_diag_g/shortcut.json), and the online clean-probe trajectory."""
    import subprocess
    import pandas as pd
    run_dir, out_dir = ROOT / run_dir, ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    assert ckpt.exists(), f"no last checkpoint at {ckpt}"
    if not (P1 / f"h_{tag}.npz").exists():
        subprocess.run([sys.executable, "gonogo/part1/embed_h.py", "--ckpt", str(ckpt.relative_to(ROOT)), "--tag", tag],
                       cwd=ROOT, check=True)
    pr = probe([tag], out_dir / "probe.json")[tag]
    sc = shortcut({tag: str(ckpt.relative_to(ROOT)), "h": str(H_CKPT.relative_to(ROOT))}, path=out_dir / "shortcut.json")
    ref = json.loads((OUT / "shortcut.json").read_text())["h"]
    same_items = all(abs(sc["h"][k] - ref[k]) < 1e-9 for k in ("recording_id_top1", "same_patient_top1", "effective_rank"))
    d = pd.read_csv(run_dir / "metrics.csv")
    col = "eval/auroc__MultilabelAUROC_epoch"
    traj = d.dropna(subset=[col]).groupby("epoch")[col].last()
    href = pd.read_csv(H_RUN / "metrics.csv").dropna(subset=[col]).groupby("epoch")[col].last()
    pick = [e for e in (0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400) if e in traj.index] + [traj.index.max()]
    online = {str(int(e)): round(float(traj.loc[e]), 4) for e in dict.fromkeys(pick)}
    passes = pr["test_macro_auroc"] >= BAR
    rep = {"run_dir": str(run_dir.relative_to(ROOT)), "checkpoint": "last.ckpt", "epoch_last": int(traj.index.max()),
           "offline_probe": pr, "h_offline_probe_frozen": H_PROBE, "bar": BAR,
           "sanity": "passes: enters the 0.5a comparison" if passes else
                     "training failed: b is the chosen representation, the encoder ablation is a limitation",
           "retrieval_and_rank": {tag: sc[tag], "h": sc["h"]}, "same_items_as_step3": same_items,
           "online_clean_probe": {"trajectory": online, "max": round(float(traj.max()), 4),
                                  "argmax_epoch": int(traj.idxmax()), "last": round(float(traj.iloc[-1]), 4),
                                  "h_max": round(float(href.max()), 4), "h_last": round(float(href.iloc[-1]), 4)}}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    g_ref = json.loads((OUT / "shortcut.json").read_text())["g_last"]
    md = [f"# Post-training report: `{tag}` (last checkpoint, epoch {rep['epoch_last']})", "",
          f"Run `{rep['run_dir']}`. Report on the last checkpoint, not checkpoint selection (plan 0.5a, revision 9).", "",
          "| Check | Candidate | h (frozen reference) | g (diagnosis) |", "|---|---|---|---|",
          f"| Offline clean probe, test fold 19 | {pr['test_macro_auroc']:.3f} [{pr['ci95'][0]:.3f}, {pr['ci95'][1]:.3f}] "
          f"| {H_PROBE:.3f} | 0.516 |",
          f"| Recording-ID retrieval top-1 | {sc[tag]['recording_id_top1']:.3f} | {sc['h']['recording_id_top1']:.3f} "
          f"| {g_ref['recording_id_top1']:.3f} |",
          f"| Same-patient retrieval top-1 | {sc[tag]['same_patient_top1']:.3f} | {sc['h']['same_patient_top1']:.3f} "
          f"| {g_ref['same_patient_top1']:.3f} |",
          f"| Effective rank (of {sc[tag]['dims']}) | {sc[tag]['effective_rank']:.1f} | {sc['h']['effective_rank']:.1f} "
          f"| {g_ref['effective_rank']:.1f} |",
          f"| Online clean probe, max (epoch) / last | {rep['online_clean_probe']['max']:.3f} "
          f"({rep['online_clean_probe']['argmax_epoch']}) / {rep['online_clean_probe']['last']:.3f} "
          f"| {rep['online_clean_probe']['h_max']:.3f} / {rep['online_clean_probe']['h_last']:.3f} | (augmented inputs) |", "",
          f"Online clean-probe trajectory (epoch: AUROC): {online}", "",
          f"Same 1,000 fold-19 items as the diagnosis of g: {same_items} (h reproduces Step 3 exactly).", "",
          f"**Sanity rule** (bar {BAR}, point estimate): {rep['sanity']}."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "config", "aug", "random", "shortcut", "losses", "report"])
    ap.add_argument("--tags", nargs="+", default=["zeroactions"])
    ap.add_argument("--run", help="report: the candidate's run dir (relative to the repository)")
    ap.add_argument("--tag", help="report: the export tag of its last checkpoint, e.g. intra_last")
    ap.add_argument("--out", default="outputs/phase05/6_intra_report", help="report: output folder")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cks = {"g_last": str(G_CKPTS["last"].relative_to(ROOT)), "h": str(H_CKPT.relative_to(ROOT))}
    {"probe": lambda: probe(a.tags), "config": config, "aug": aug, "random": random_ckpt,
     "shortcut": lambda: shortcut(cks), "losses": losses,
     "report": lambda: report(a.run, a.tag, a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
