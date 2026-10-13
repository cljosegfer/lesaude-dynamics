"""
Signal-Space Decoder for the Forward Dynamics Model
====================================================
Does Dyn move an embedding the way the transition moves the ECG? The latent diagnostics (papel/appendix.tex,
Embedding-Space Diagnostic; scripts/cem_planning.py) only compare predicted states with other embeddings. This
script trains a decoder from each frozen encoder's h back to the ECG and reads the forward model through it.

Targets (anatomie/median_beats.py): the 12-lead median beat, 0.4 s before to 0.6 s after the QRS in the input's
z-units, plus heart rate and RR irregularity (normalized RMSSD). h is a global time average and does not encode
where the beats fall in the window, so the 10 s strip itself is not decodable.

1. Decoder. Per encoder (anatomie/encoders.py), an MLP from h (standardized with fit-set statistics) to beat
   and log rhythm, trained with MSE on every usable ECG of folds 0-17; the epoch with the lowest fold-18 beat
   MSE is kept and scored on every split. Beat R^2 is against the fit-set mean beat.

2. Forward test, for checkpoints with a forward model (load_dynamics), on the k=1 cross-stay test pairs of
   scripts/evaluate_inverse.py whose two ECGs have usable beats. Each condition's estimate of h(t+1) is decoded
   and scored against the real median beat and rhythm of X(t+1):
     h_t      do-nothing: the current state
     true     Dyn(h_t, a_t)
     zero     Dyn(h_t, 0)
     decoy    Dyn(h_t, ã), ã admissible given y_t with a_t's onset and resolution counts on random labels (the
              decoys of scripts/joint_pretrain.py's _CEMProbe); ã = 0 on stable pairs
     oracle   h_{t+1}: the decoder's ceiling
   plus B_t, the real beat of X(t) (no decoder). Groups all / active (a_t != 0) / stable, as tab:embedding-mse;
   on active pairs, true is compared with h_t, zero and decoy pair by pair (bootstrap CI of the mean gap).

3. Counterfactual effects, for every label and direction that flips in >= --min-events test pairs:
     real     mean change of the decoded beat D(h_{t+1}) - D(h_t) over those pairs, minus the same mean over the
              pairs where the label stays as it was (raw: the same with the real beats)
     model    mean over the same pairs of D(Dyn(h_t, +-e_i)) - D(Dyn(h_t, 0)), a single-label intervention
   compared by cosine over the 12 x 500 beat and by the heart-rate change. "other" averages each model effect's
   cosine with the real effects of the other labels flipping in the same direction.

Interpretation
--------------
  Decoder R^2 is how much beat shape h carries; random init is the floor (--ckpt random=random).
  On active pairs a useful forward model has true below h_t, zero and decoy (P(true better) > 0.5, CI of the gap
  above 0). true ~ zero ~ decoy is the signal-space version of the latent finding that Dyn barely uses a_t.
  cos(model, real) well above "other", with a matching heart-rate change, means Dyn simulates that label's
  transition rather than a generic "sicker ECG" direction. cos(real, raw) says whether the decoder can express
  the label's change at all; where it is low, the model-vs-real cosine for that label means nothing.

Outputs (--out-dir): signal_decoder.json; per encoder decoder_<name>.pt (weights and standardization); per
forward model effects_<name>.npz (effect beats), pairs_<name>.png (example active test pairs, the same pairs for
every encoder) and effects_<name>.png (the effects with the most events).

Example
-------
python anatomie/signal_decoder.py --ckpt random=random --ckpt dyn=archive/pretrain_val.ckpt \\
    --ckpt multistep=archive/pretrain_multistep0.ckpt --out-dir outputs/signal_decoder
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import lance
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, data_paths, embed, load_dynamics, load_encoder  # noqa: E402
from median_beats import BEAT_LEN, FS, POST, PRE, load_cache  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RHYTHM_OFFSET = (0.0, 0.01)    # decoded as log(heart rate), log(nRMSSD + 0.01): nRMSSD can be 0
CONDITIONS = ("h_t", "true", "zero", "decoy", "oracle")

# Figures: text and grid tokens and categorical slots 1-3 of the dataviz reference palette (light mode)
INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3de", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIG_LEADS = (1, 6, 10)         # II, V1, V5
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.titlecolor": INK, "axes.edgecolor": GRID,
         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
         "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
         "legend.frameon": False}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=SPEC",
                   help="encoder to evaluate (repeatable); SPEC is 'random' or a checkpoint path")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--cache-dir", default=None,
                   help="where the median beats and embeddings are cached (default: the Lance table's directory)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=1024, help="decoder training batch")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-events", type=int, default=20, help="test pairs a label must flip in to get an effect")
    p.add_argument("--n-examples", type=int, default=4, help="rows per figure")
    p.add_argument("--bootstrap", type=int, default=1000, help="resamples for the paired gaps' 95%% CI")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/signal_decoder")
    return p.parse_args()


class BeatDecoder(nn.Module):
    """Standardized h -> median beat (12, BEAT_LEN) and standardized log rhythm (2,)."""

    def __init__(self, dim: int, hidden: int, leads: int = 12):
        super().__init__()
        self.leads = leads
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.beat = nn.Linear(hidden, leads * BEAT_LEN)
        self.rhythm = nn.Linear(hidden, len(RHYTHM_OFFSET))

    def forward(self, h):
        z = self.trunk(h)
        return self.beat(z).view(-1, self.leads, BEAT_LEN), self.rhythm(z)


class Readout:
    """A decoder with its standardization: raw h -> (beat, [heart rate bpm, nRMSSD])."""

    def __init__(self, dec, h_mu, h_sd, r_mu, r_sd):
        self.dec, self.h_mu, self.h_sd, self.r_mu, self.r_sd = dec, h_mu, h_sd, r_mu, r_sd
        self.offset = torch.tensor(RHYTHM_OFFSET, device=h_mu.device)

    def natural(self, rz):
        return torch.exp(rz * self.r_sd + self.r_mu) - self.offset

    @torch.no_grad()
    def __call__(self, h):
        beat, rz = zip(*(self.dec((x - self.h_mu) / self.h_sd) for x in h.split(8192)))
        return torch.cat(beat), self.natural(torch.cat(rz))


def lance_table(lance_path):
    t = lance.dataset(lance_path).to_table(columns=["fold", "icd"])
    fold = t.column("fold").to_numpy()
    y = t.column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(fold), -1)
    return fold, y


def test_pairs(pairs_path, usable):
    t = lance.dataset(pairs_path).to_table(filter="(fold = 19) AND (pair_type = 'cross_stay')",
                                           columns=["idx_t", "idx_t1"]).to_pandas()
    pt, pt1 = t["idx_t"].values, t["idx_t1"].values
    keep = usable[pt] & usable[pt1]
    return pt[keep], pt1[keep], len(t)


def _cache_key(spec, lance_path):
    """Pins the checkpoint file and the Lance table version, as scripts/cem_planning.py's embedding cache does."""
    version = lance.dataset(lance_path).version
    if spec == "random":
        return f"random_ecg_v{version}"
    stat = (REPO / spec).stat()
    return f"{Path(spec).stem}_{stat.st_size}_{int(stat.st_mtime)}_ecg_v{version}"


def embeddings(specs, rows, lance_path, cache_dir, batch_size, workers):
    """{name: (N, D) float16 h of every Lance row, NaN outside `rows`}, cached per checkpoint in cache_dir/embeddings
    and computed for the missing encoders in one pass over the data."""
    n = lance.dataset(lance_path).count_rows()
    out, todo = {}, {}
    for name, spec in specs.items():
        path = Path(cache_dir) / "embeddings" / f"h_{_cache_key(spec, lance_path)}.npy"
        H = np.load(path) if path.exists() else None
        if H is not None and not np.isnan(H[rows, 0]).any():
            out[name] = H
        else:
            todo[name] = (spec, path, H)
    if todo:
        print(f"Embedding {len(rows):,} ECGs with {', '.join(todo)}", flush=True)
        feats, _ = embed({name: load_encoder(spec) for name, (spec, _, _) in todo.items()}, rows, lance_path,
                         layers={"h"}, batch_size=batch_size, num_workers=workers)
        for name, (_, path, H) in todo.items():
            if H is None:
                H = np.full((n, feats[name]["h"].shape[1]), np.nan, np.float16)
            H[rows] = feats[name]["h"].numpy()
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, H)
            out[name] = H
    return out


def train_decoder(H, data, args):
    """Fresh decoder on the fit split; returns the Readout at the epoch with the lowest val beat MSE, that epoch,
    and the val beat MSE of every epoch."""
    h_mu, h_sd = H["fit"].mean(0), H["fit"].std(0) + 1e-6
    torch.manual_seed(args.seed)
    dec = BeatDecoder(H["fit"].shape[1], args.hidden).cuda()
    with torch.no_grad():
        dec.beat.bias.copy_(data["mean_beat"].flatten())   # start at the mean beat (R^2 ~ 0), not at a flat line
    read = Readout(dec, h_mu, h_sd, data["r_mu"], data["r_sd"])
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = -(-len(H["fit"]) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    fit = data["fit"]
    best, curve = None, []
    for ep in range(args.epochs):
        dec.train()
        for b in torch.randperm(len(H["fit"]), device="cuda").split(args.batch_size):
            beat, rz = dec((H["fit"][b] - h_mu) / h_sd)
            loss = F.mse_loss(beat, fit["beat"][b].float()) + F.mse_loss(rz, fit["rz"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        dec.eval()
        curve.append(reconstruction(read, H["val"], data["val"], data["mean_beat"])["beat_mse"])
        print(f"  epoch {ep:>2}  train loss {loss.item():.4f}  val beat MSE {curve[-1]:.4f}", flush=True)
        if best is None or curve[-1] < best[0]:
            best = (curve[-1], copy.deepcopy(dec.state_dict()), ep)
    dec.load_state_dict(best[1])
    return read, best[2], curve


def _pearson(a, b):
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@torch.no_grad()
def reconstruction(read, H, d, mean_beat):
    """Beat MSE and R^2 (overall and per lead) against the real beats, and the decoded rhythm's errors."""
    se = torch.zeros(len(LEADS), device="cuda")
    se_mean = torch.zeros(len(LEADS), device="cuda")
    rhythm = []
    for h, beat in zip(H.split(8192), d["beat"].split(8192)):
        pred, r = read(h)
        beat = beat.float()
        se += (pred - beat).pow(2).sum((0, 2))
        se_mean += (mean_beat - beat).pow(2).sum((0, 2))
        rhythm.append(r)
    rhythm, real = torch.cat(rhythm), d["rhythm"]
    log_pred, log_real = torch.log(rhythm + read.offset), torch.log(real + read.offset)
    return {
        "beat_mse": float(se.sum() / (len(H) * len(LEADS) * BEAT_LEN)),
        "beat_r2": float(1 - se.sum() / se_mean.sum()),
        "lead_r2": dict(zip(LEADS, (1 - se / se_mean).tolist())),
        "hr_mae": float((rhythm[:, 0] - real[:, 0]).abs().mean()),
        "hr_r": _pearson(log_pred[:, 0], log_real[:, 0]),
        "irr_mae": float((rhythm[:, 1] - real[:, 1]).abs().mean()),
        "irr_r": _pearson(log_pred[:, 1], log_real[:, 1]),
    }


def decoys(yt, at, generator):
    """One admissible decoy per pair with a_t's onset and resolution counts on random labels (onsets only where
    y_t = 0, resolutions only where y_t = 1), drawn as scripts/joint_pretrain.py's _CEMProbe draws them."""
    def subset(allowed, counts):
        scores = torch.rand(allowed.shape, generator=generator, device=allowed.device).masked_fill(~allowed, -1.0)
        return scores.argsort(-1, descending=True).argsort(-1) < counts.unsqueeze(1)
    return subset(yt == 0, (at == 1).sum(1)).float() - subset(yt == 1, (at == -1).sum(1)).float()


def _gap(e_other, e_true, n_boot, rng):
    """Paired comparison of per-pair errors: P(true better) and the mean gap other - true with a bootstrap CI."""
    d = (e_other - e_true).double().cpu().numpy()
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"p_true_better": float((d > 0).mean()), "gap": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


@torch.no_grad()
def forward_test(read, dyn, P, n_boot, seed):
    """Decodes every condition's estimate of h(t+1) on the test pairs P and scores it against X(t+1); returns
    (metrics, {condition: (decoded beat, decoded rhythm)})."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ht, at = P["ht"], P["at"]
    states = {"h_t": ht, "true": dyn(ht, at), "zero": dyn(ht, torch.zeros_like(at)),
              "decoy": dyn(ht, decoys(P["yt"], at, g)), "oracle": P["ht1"]}
    decoded = {c: read(s) for c, s in states.items()}
    err = {"beat": {"B_t": (P["Bt"] - P["Bt1"]).pow(2).mean((1, 2))},
           "hr": {"B_t": (P["rt"][:, 0] - P["rt1"][:, 0]).abs()},
           "irr": {"B_t": (P["rt"][:, 1] - P["rt1"][:, 1]).abs()},
           "latent": {}}
    for c, s in states.items():
        beat, rhythm = decoded[c]
        err["beat"][c] = (beat - P["Bt1"]).pow(2).mean((1, 2))
        err["hr"][c] = (rhythm[:, 0] - P["rt1"][:, 0]).abs()
        err["irr"][c] = (rhythm[:, 1] - P["rt1"][:, 1]).abs()
        err["latent"][c] = (s - P["ht1"]).pow(2).mean(1)
    active = (at != 0).any(1)
    groups = {"all": torch.ones_like(active), "active": active, "stable": ~active}
    table = {m: {grp: {c: float(e[mask].mean()) for c, e in errs.items()} for grp, mask in groups.items()}
             for m, errs in err.items()}
    rng = np.random.default_rng(seed)
    paired = {m: {c: _gap(err[m][c][active], err[m]["true"][active], n_boot, rng) for c in ("h_t", "zero", "decoy")}
              for m in ("beat", "hr")}
    metrics = {"n_pairs": len(active), "n_active": int(active.sum()), "table": table, "paired": paired,
               "beat_mse": {c: e.cpu().numpy() for c, e in err["beat"].items()}}
    return metrics, decoded


@torch.no_grad()
def effects(read, dyn, P, decoded, vocab, min_events):
    """Model vs real effect of each label's onset / resolution (see the module docstring); returns (entries with
    their (12, BEAT_LEN) effect beats, summary)."""
    d_dec = decoded["oracle"][0] - decoded["h_t"][0]
    d_raw = P["Bt1"] - P["Bt"]
    hr_dec = decoded["oracle"][1][:, 0] - decoded["h_t"][1][:, 0]
    hr_raw = P["rt1"][:, 0] - P["rt"][:, 0]
    zero_beat, zero_rhythm = decoded["zero"]
    yt, at = P["yt"], P["at"]
    entries = []
    for i, code in enumerate(vocab):
        for s, kind in ((1.0, "onset"), (-1.0, "resolution")):
            ev = at[:, i] == s
            ctrl = (at[:, i] == 0) & (yt[:, i] == float(s == -1.0))     # stays absent (onset) / present
            if ev.sum() < min_events or ctrl.sum() < min_events:
                continue
            a = torch.zeros_like(at[ev])
            a[:, i] = s
            beat, rhythm = read(dyn(P["ht"][ev], a))
            entries.append({
                "label": code, "kind": kind, "n": int(ev.sum()),
                "model": (beat - zero_beat[ev]).mean(0),
                "real": d_dec[ev].mean(0) - d_dec[ctrl].mean(0),
                "raw": d_raw[ev].mean(0) - d_raw[ctrl].mean(0),
                "hr_model": float((rhythm[:, 0] - zero_rhythm[ev, 0]).mean()),
                "hr_real": float(hr_dec[ev].mean() - hr_dec[ctrl].mean()),
                "hr_raw": float(hr_raw[ev].mean() - hr_raw[ctrl].mean()),
            })
    if not entries:
        return [], {"n_entries": 0}
    M, R, W = (F.normalize(torch.stack([e[k] for e in entries]).flatten(1), dim=1) for k in ("model", "real", "raw"))
    cos = M @ R.T                                                    # (E, E): model effect k vs real effect l
    labels, kinds = np.array([e["label"] for e in entries]), np.array([e["kind"] for e in entries])
    # other labels flipping the same way, so a shared "new diagnosis" direction does not count as specific
    other = torch.from_numpy((labels[:, None] != labels[None, :]) & (kinds[:, None] == kinds[None, :])).cuda()
    for k, e in enumerate(entries):
        e["cos_model_real"] = float(cos[k, k])
        e["cos_model_other"] = float(cos[k][other[k]].mean()) if other[k].any() else float("nan")
        e["cos_model_raw"] = float(M[k] @ W[k])
        e["cos_real_raw"] = float(R[k] @ W[k])
        e["norm_ratio"] = float(e["model"].norm() / e["real"].norm())
    summary = {"n_entries": len(entries),
               "cos_model_real": float(np.mean([e["cos_model_real"] for e in entries])),
               "cos_model_other": float(np.nanmean([e["cos_model_other"] for e in entries])),
               "cos_real_raw": float(np.mean([e["cos_real_raw"] for e in entries])),
               "norm_ratio": float(np.median([e["norm_ratio"] for e in entries]))}
    return entries, summary


def _flips(a, vocab, max_shown=4):
    flips = [f"+{vocab[i]}" for i in np.where(a == 1)[0]] + [f"-{vocab[i]}" for i in np.where(a == -1)[0]]
    return " ".join(flips[:max_shown]) + (f" (+{len(flips) - max_shown} more)" if len(flips) > max_shown else "")


def _beat_figure(n_rows):
    fig, axes = plt.subplots(n_rows, len(FIG_LEADS), figsize=(3.3 * len(FIG_LEADS), 1.8 * n_rows + 0.7),
                             sharex=True, squeeze=False)
    t = (np.arange(BEAT_LEN) - PRE) / FS
    for ax, lead in zip(axes.flat, FIG_LEADS * n_rows):
        ax.text(0.02, 0.96, LEADS[lead], transform=ax.transAxes, va="top", color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("time from QRS (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-units")
    return fig, axes, t


def plot_pairs(path, title, P, decoded, beat_mse, idx, vocab):
    lines = [("real X(t)", FAINT, 1.0, lambda k: P["Bt"][k]),
             ("real X(t+1)", INK, 1.5, lambda k: P["Bt1"][k]),
             ("decoded h(t+1), the ceiling", AQUA, 1.5, lambda k: decoded["oracle"][0][k]),
             ("decoded h(t), do-nothing", ORANGE, 1.5, lambda k: decoded["h_t"][0][k]),
             ("decoded Dyn(h(t), a(t))", BLUE, 1.5, lambda k: decoded["true"][0][k])]
    at = P["at"].cpu().numpy()
    with plt.rc_context(STYLE):
        fig, axes, t = _beat_figure(len(idx))
        for r, k in enumerate(idx):
            for label, color, lw, get in lines:
                beat = get(k).cpu().numpy()
                for ax, lead in zip(axes[r], FIG_LEADS):
                    ax.plot(t, beat[lead], color=color, lw=lw, label=label)
            e = {c: beat_mse[c][k] for c in ("true", "h_t", "oracle", "B_t")}
            axes[r, 0].set_title(f"{_flips(at[k], vocab)}   beat MSE: Dyn {e['true']:.3f}, do-nothing {e['h_t']:.3f}, "
                                 f"ceiling {e['oracle']:.3f}, real X(t) {e['B_t']:.3f}", loc="left")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(lines),
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, y=1.04, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_effects(path, title, entries):
    lines = [("real change, real beats", INK, "raw"), ("real change, decoded", AQUA, "real"),
             ("Dyn(h, ±e_i) - Dyn(h, 0), decoded", BLUE, "model")]
    with plt.rc_context(STYLE):
        fig, axes, t = _beat_figure(len(entries))
        for r, e in enumerate(entries):
            for label, color, key in lines:
                beat = e[key].cpu().numpy()
                for ax, lead in zip(axes[r], FIG_LEADS):
                    ax.plot(t, beat[lead], color=color, label=label)
            for ax in axes[r]:
                ax.axhline(0, color=MUTED, lw=0.6)
            axes[r, 0].set_title(f"{e['label']} {e['kind']} (n={e['n']})   cos(model, real) {e['cos_model_real']:+.2f}, "
                                 f"other labels {e['cos_model_other']:+.2f}   heart rate change: model "
                                 f"{e['hr_model']:+.1f}, real {e['hr_raw']:+.1f} (decoded {e['hr_real']:+.1f}) bpm",
                                 loc="left")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(lines),
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, y=1.04, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def print_decoder(rec, best_epoch):
    print(f"  decoder best-val epoch {best_epoch}")
    print(f"  {'split':<6}{'beat R2':>9}{'beat MSE':>10}{'HR MAE':>8}{'HR r':>7}{'nRMSSD MAE':>12}{'nRMSSD r':>10}")
    for split, m in rec.items():
        print(f"  {split:<6}{m['beat_r2']:>9.4f}{m['beat_mse']:>10.4f}{m['hr_mae']:>8.2f}{m['hr_r']:>7.3f}"
              f"{m['irr_mae']:>12.4f}{m['irr_r']:>10.3f}", flush=True)


def print_forward(m):
    print(f"  forward test: {m['n_pairs']:,} test pairs, {m['n_active']:,} active")
    for metric, unit in (("beat", "beat MSE"), ("hr", "|HR error| bpm"), ("latent", "latent MSE")):
        cols = [c for c in ("B_t",) + CONDITIONS if c in m["table"][metric]["all"]]
        print(f"  {unit:<16}" + "".join(f"{c:>9}" for c in cols))
        for grp, row in m["table"][metric].items():
            print(f"    {grp:<14}" + "".join(f"{row[c]:>9.4f}" for c in cols))
    for metric, rows in m["paired"].items():
        for c, r in rows.items():
            print(f"  active, {metric:<4} true vs {c:<6} P(true better) {r['p_true_better']:.3f}  "
                  f"gap {r['gap']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]")


def print_effects(entries, summary, min_events):
    if not entries:
        print(f"  effects: no label flips in >= {min_events} test pairs")
        return
    print(f"  effects ({summary['n_entries']} label/direction entries, >= {min_events} events): cos(model, real) "
          f"{summary['cos_model_real']:+.3f} vs other labels {summary['cos_model_other']:+.3f}; "
          f"cos(real, raw) {summary['cos_real_raw']:+.3f}; median |model|/|real| {summary['norm_ratio']:.2f}")
    print(f"    {'label':<6}{'kind':<11}{'n':>5}{'cos':>8}{'other':>8}{'real~raw':>9}{'|m|/|r|':>8}"
          f"{'dHR model':>11}{'decoded':>9}{'raw':>7}")
    for e in sorted(entries, key=lambda e: -e["n"])[:15]:
        print(f"    {e['label']:<6}{e['kind']:<11}{e['n']:>5}{e['cos_model_real']:>+8.3f}{e['cos_model_other']:>+8.3f}"
              f"{e['cos_real_raw']:>+9.3f}{e['norm_ratio']:>8.2f}{e['hr_model']:>+11.2f}{e['hr_real']:>+9.2f}"
              f"{e['hr_raw']:>+7.2f}", flush=True)


def main():
    args = parse_args()
    paths = data_paths(args.config)
    lance_path, pairs_path = paths["lance_path"], paths["pairs_path"]
    vocab = json.loads(Path(paths["vocabulary_path"]).read_text())
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(lance_path).parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(lance_path, cache_dir=cache_dir, num_workers=args.workers)
    fold, y = lance_table(lance_path)
    usable = cache["n_beats"] > 0
    rows = {"fit": np.where(usable & (fold <= 17))[0], "val": np.where(usable & (fold == 18))[0],
            "test": np.where(usable & (fold == 19))[0]}
    pt, pt1, n_all = test_pairs(pairs_path, usable)
    print({k: len(v) for k, v in rows.items()}, f"usable ECGs; {len(pt):,} of {n_all:,} test pairs", flush=True)

    specs = dict(c.split("=", 1) for c in args.ckpt)
    H = embeddings(specs, np.where(usable)[0], lance_path, cache_dir, args.embed_batch_size, args.workers)

    # Targets on the GPU once, shared by every encoder's decoder
    rz = np.log(cache["rhythm"] + np.array(RHYTHM_OFFSET, np.float32))
    r_mu, r_sd = rz[rows["fit"]].mean(0), rz[rows["fit"]].std(0)
    data = {"r_mu": torch.from_numpy(r_mu).cuda(), "r_sd": torch.from_numpy(r_sd).cuda()}
    for split, r in rows.items():
        data[split] = {"beat": torch.from_numpy(np.asarray(cache["beats"][r])).cuda(),
                       "rhythm": torch.from_numpy(cache["rhythm"][r]).cuda(),
                       "rz": torch.from_numpy((rz[r] - r_mu) / r_sd).cuda()}
    data["mean_beat"] = sum(b.float().sum(0) for b in data["fit"]["beat"].split(65536)) / len(rows["fit"])

    pos, pos1 = np.searchsorted(rows["test"], pt), np.searchsorted(rows["test"], pt1)
    assert (rows["test"][pos] == pt).all() and (rows["test"][pos1] == pt1).all(), "test pair outside fold 19"
    P = {"yt": torch.from_numpy(y[pt]).float().cuda(), "Bt": data["test"]["beat"][pos].float(),
         "Bt1": data["test"]["beat"][pos1].float(), "rt": data["test"]["rhythm"][pos],
         "rt1": data["test"]["rhythm"][pos1]}
    P["at"] = torch.from_numpy(y[pt1]).float().cuda() - P["yt"]
    active = (P["at"] != 0).any(1).cpu().numpy()
    examples = np.sort(np.random.default_rng(args.seed).choice(np.where(active)[0], min(args.n_examples, active.sum()),
                                                               replace=False))

    results = {}
    for name, spec in specs.items():
        print(f"\n== {name} ({spec})", flush=True)
        Hs = {split: torch.from_numpy(H[name][r].astype(np.float32)).cuda() for split, r in rows.items()}
        read, best_epoch, curve = train_decoder(Hs, data, args)
        rec = {split: reconstruction(read, Hs[split], data[split], data["mean_beat"]) for split in rows}
        print_decoder(rec, best_epoch)
        torch.save({"state_dict": read.dec.state_dict(), "spec": spec, "hidden": args.hidden,
                    "h_mean": read.h_mu.cpu(), "h_std": read.h_sd.cpu(), "rhythm_mean": data["r_mu"].cpu(),
                    "rhythm_std": data["r_sd"].cpu(), "rhythm_offset": RHYTHM_OFFSET,
                    "beat": {"fs": FS, "pre": PRE, "post": POST, "leads": LEADS}},
                   out_dir / f"decoder_{name}.pt")
        results[name] = {"spec": spec, "decoder": {"best_epoch": best_epoch, "val_beat_mse": curve, **rec}}

        dyn = load_dynamics(spec)
        if dyn is None:
            print("  no forward model in this checkpoint: decoder only", flush=True)
        else:
            P["ht"] = torch.from_numpy(H[name][pt].astype(np.float32)).cuda()
            P["ht1"] = torch.from_numpy(H[name][pt1].astype(np.float32)).cuda()
            metrics, decoded = forward_test(read, dyn, P, args.bootstrap, args.seed)
            print_forward(metrics)
            entries, summary = effects(read, dyn, P, decoded, vocab, args.min_events)
            print_effects(entries, summary, args.min_events)

            plot_pairs(out_dir / f"pairs_{name}.png", f"{name}: median beats of random active test pairs", P,
                       decoded, metrics.pop("beat_mse"), examples, vocab)
            if entries:
                top = sorted(entries, key=lambda e: -e["n"])[:args.n_examples]
                plot_effects(out_dir / f"effects_{name}.png", f"{name}: effect of a single label flip on the "
                             "median beat (mean over the test pairs where it flips)", top)
                np.savez_compressed(out_dir / f"effects_{name}.npz",
                                    **{k: torch.stack([e[k] for e in entries]).cpu().numpy()
                                       for k in ("model", "real", "raw")},
                                    label=[e["label"] for e in entries], kind=[e["kind"] for e in entries])
            results[name]["forward"] = metrics
            results[name]["effects"] = {"summary": summary, "entries": [
                {k: v for k, v in e.items() if k not in ("model", "real", "raw")} for e in entries]}
        del Hs, read
        torch.cuda.empty_cache()

    with open(out_dir / "signal_decoder.json", "w") as f:
        json.dump({"encoders": specs, "args": vars(args), "n_test_pairs": len(pt), "results": results}, f, indent=2)
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()
