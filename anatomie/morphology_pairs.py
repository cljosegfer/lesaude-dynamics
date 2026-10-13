"""
Morphology-Change Figures for the Forward Model
================================================
Example pairs whose median beat changes shape between X(t) and X(t+1), read through one encoder's signal decoder
(anatomie/signal_decoder.py): does decoded h(t+1) show the new morphology, does decoded h(t) (do-nothing) keep the
old one, and does decoded Dyn(h(t), a(t)) move toward the new one?

Each figure holds two pairs; each pair takes two rows over its drawn leads, by default the --n-leads leads where its
beat changes most (mean squared difference of the per-lead mean-centred beats):
  observed  the real median beats of X(t) and X(t+1): the morphology difference
  decoded   decoded h(t+1) (the decoder's ceiling), decoded h(t) (do-nothing) and decoded Dyn(h(t), a(t))
The two rows of a pair share the y scale per lead, so each decoded trace reads against the observed beat above it.
A lead with fewer than --min-levels distinct values in either real beat is never drawn: z-scoring a nearly flat lead
blows its amplitude quantization up into a staircase that changes "shape" between any two ECGs.

Selection, on the k=1 cross-stay pairs of --split whose two ECGs have usable beats (active pairs only unless
--include-stable). Shape is compared on the drawn leads, by correlation of the per-lead mean-centred beats, so
neither a baseline shift nor an amplitude change counts as a new morphology:
  change   1 - corr(B(t), B(t+1)) between the two real beats; pairs in the top --top-change fraction
  shape    shape(E) = corr(E, B(t+1)) - corr(E, B(t)): whether an estimate E of the next beat looks more like
           X(t+1)'s beat (> 0) or X(t)'s (< 0). A flattened copy of X(t)'s beat still looks like X(t).
  kept     shape(decoded h(t)) <= -margin and shape(decoded h(t+1)) >= margin (--margin): the decoder tells the
           two morphologies apart
  ranked   by shape(decoded Dyn(h(t), a(t))), highest first

The top figures show where Dyn works, not how often: over all test pairs, decoded Dyn is worse than do-nothing
(anatomie/signal_decoder_report.md). The script prints how many pairs pass each filter and how often Dyn takes the
new shape among them, and --contrast adds one figure with the middle- and the last-ranked kept pair, a typical case
and the worst. The CSV lists every drawn pair with its numbers, including those of decoded Dyn(h(t), 0): what the
state map does without the action (drawn with --zero-action).

--decoy-action also draws decoded Dyn(h(t), decoy), the decoy an admissible action strictly different from a(t)
with its onset and resolution counts on random labels (strict_decoys). If Dyn uses which labels changed, the decoy
moves the beat differently from a(t); if it only applies a generic "something changed" shift, the two match. Both
are scored by their effect, decoded Dyn(h(t), .) - decoded Dyn(h(t), 0) on the drawn leads, so the shared state map
does not make them look alike: the correlation of the decoy's effect with a(t)'s (1 = the same change) and its
relative size, per figure title, in the CSV, and summarized over each group of pairs.

Outputs, with <name> from the decoder file (decoder_dyn.pt -> dyn):
  --out-dir/<name>_<split>_morphology_<k>.png         the top figures, ranks 2k-1 and 2k
  --out-dir/<name>_<split>_morphology_middle_bottom.png   with --contrast
  --out-dir/<name>_<split>_selected_pairs.csv         the drawn pairs, by rank, with the figure each is in
  morphology_cache_<name>_<split>.npz           next to the decoder: the split's beats and embeddings, reused while
                                                the checkpoint and the Lance table are unchanged

Example
-------
python anatomie/morphology_pairs.py                         # dyn = archive/pretrain_val.ckpt
python anatomie/morphology_pairs.py --contrast              # plus the middle- and the last-ranked pair
python anatomie/morphology_pairs.py --contrast --decoy-action --disjoint-decoy --out-dir outputs/signal_decoder/morphology_decoy
python anatomie/morphology_pairs.py --n-figures 10 --leads II V1 V5 --zero-action
"""

import argparse
import json
import sys
from pathlib import Path

import lance
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import data_paths, load_dynamics, load_encoder  # noqa: E402
from median_beats import BEAT_LEN, FS, PRE, median_beats  # noqa: E402
from signal_decoder import (AQUA, BLUE, FAINT, INK, LEADS, MUTED, ORANGE, STYLE, BeatDecoder, Readout,  # noqa: E402
                            _cache_key, _flips)

SPLITS = {"val": 18, "test": 19}
YELLOW = "#eda100"             # categorical slot 4 of the dataviz reference palette, for decoded Dyn(h(t), 0)
MAGENTA = "#e87ba4"            # slot 5, for decoded Dyn(h(t), decoy)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", default="outputs/signal_decoder/decoder_dyn.pt",
                   help="decoder saved by anatomie/signal_decoder.py; it names its encoder checkpoint")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--split", default="test", choices=list(SPLITS))
    p.add_argument("--include-stable", action="store_true", help="also select pairs with no diagnosis change")
    p.add_argument("--top-change", type=float, default=0.1,
                   help="fraction of pairs, those whose real beats differ most in shape, to select from")
    p.add_argument("--margin", type=float, default=0.2,
                   help="how clearly decoded h(t) and h(t+1) must each look like their own real beat")
    p.add_argument("--n-figures", type=int, default=5, help="figures to draw, two pairs each")
    p.add_argument("--leads", nargs="+", choices=LEADS, default=None,
                   help="leads to draw (default: per pair, the --n-leads leads whose beat changes most)")
    p.add_argument("--n-leads", type=int, default=3)
    p.add_argument("--min-levels", type=int, default=10,
                   help="fewest distinct values a lead needs in both real beats to be drawn")
    p.add_argument("--zero-action", action="store_true", help="also draw decoded Dyn(h(t), 0)")
    p.add_argument("--contrast", action="store_true",
                   help="also draw one figure with the middle- and the last-ranked kept pair")
    p.add_argument("--decoy-action", action="store_true",
                   help="also draw decoded Dyn(h(t), decoy), a different action with a(t)'s onset/resolution counts")
    p.add_argument("--disjoint-decoy", action="store_true",
                   help="the decoy flips only labels a(t) leaves unchanged (fewer pairs have one)")
    p.add_argument("--seed", type=int, default=0, help="seed of the decoy draws")
    p.add_argument("--batch-size", type=int, default=512, help="ECGs per read and encoder pass")
    p.add_argument("--out-dir", default="outputs/signal_decoder/morphology")
    return p.parse_args()


def beats_and_embeddings(rows, spec, lance_path, path, batch_size):
    """Real median beats, beat counts (0 = unusable) and float16 h, as signal_decoder.py caches it, of the Lance
    rows `rows`; cached in `path`, keyed by the checkpoint file and the Lance table version."""
    key = _cache_key(spec, lance_path)
    if path.exists():
        z = np.load(path)
        if str(z["key"]) == key and np.array_equal(z["rows"], rows):
            return z["beats"], z["n_beats"], z["h"]
    print(f"Reading {len(rows):,} ECGs for beats and embeddings (cached in {path})", flush=True)
    ds, enc = lance.dataset(lance_path), load_encoder(spec)
    beats = np.zeros((len(rows), len(LEADS), BEAT_LEN), np.float16)
    n_beats = np.zeros(len(rows), np.int16)
    h = None
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        x = (ds.take(chunk.tolist(), columns=["waveform"]).column("waveform").combine_chunks().flatten()
             .to_numpy(zero_copy_only=False).reshape(len(chunk), 5000, 12).astype(np.float32).transpose(0, 2, 1))
        x = np.ascontiguousarray(x)
        b, _, k = median_beats(x)
        beats[s:s + len(chunk)], n_beats[s:s + len(chunk)] = b, k
        hc = enc(torch.from_numpy(x).cuda())["h"].cpu().numpy()
        if h is None:
            h = np.zeros((len(rows), hc.shape[1]), np.float16)
        h[s:s + len(chunk)] = hc
    np.savez(path, key=key, rows=rows, beats=beats, n_beats=n_beats, h=h)
    return beats, n_beats, h


def strict_decoys(yt, at, generator, disjoint=False, tries=32):
    """One admissible decoy per pair, different from a(t): a(t)'s onset and resolution counts on random labels
    (onsets only where y(t) = 0, resolutions only where y(t) = 1, drawn as signal_decoder.decoys draws them), the
    first of `tries` draws that differs. With disjoint, the decoy flips only labels a(t) leaves unchanged.
    Returns (decoys (N, C), found (N,)); found is False where no draw qualifies, e.g. when every present label
    resolves and nothing else changes, so a(t) is the only such action."""
    def subsets(allowed, counts):
        scores = torch.rand((tries, *allowed.shape), generator=generator, device=allowed.device)
        scores = scores.masked_fill(~allowed, -1.0)                       # disallowed labels rank last
        return (scores.argsort(-1, descending=True).argsort(-1) < counts.unsqueeze(1)) & allowed
    free = at == 0 if disjoint else torch.ones_like(at, dtype=torch.bool)
    n_ons, n_res = (at == 1).sum(1), (at == -1).sum(1)
    cand = subsets((yt == 0) & free, n_ons).float() - subsets((yt == 1) & free, n_res).float()
    # a draw qualifies if it differs from a(t) and has a(t)'s counts (it falls short when too few labels are allowed)
    ok = (cand != at).any(-1) & ((cand == 1).sum(-1) == n_ons) & ((cand == -1).sum(-1) == n_res)
    first = ok.float().argmax(0)                                          # first qualifying draw
    return cand[first, torch.arange(len(at), device=at.device)], ok.any(0)


def plot_figure(path, title, items, decoded):
    """Two rows per pair (observed, decoded) over the pair's leads; each pair's columns share their y scale."""
    observed = [("real X(t)", FAINT, 1.0, "Bt"), ("real X(t+1)", INK, 1.5, "Bt1")]
    n_leads = len(items[0]["leads"])
    t = (np.arange(BEAT_LEN) - PRE) / FS
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.3 * n_leads, 3.9 * len(items) + 0.8))
        outer = fig.add_gridspec(len(items), 1, hspace=0.32, top=1 - 1.15 / fig.get_figheight(), bottom=0.07)
        for k, it in enumerate(items):
            axes = outer[k].subgridspec(2, n_leads, hspace=0.1).subplots(sharex=True, sharey="col", squeeze=False)
            for row, (label, lines) in enumerate((("observed", observed), ("decoded", decoded))):
                for ax, lead in zip(axes[row], it["leads"]):
                    for _, color, lw, key in lines:
                        ax.plot(t, it[key][lead], color=color, lw=lw)
                    ax.text(0.02, 0.96, LEADS[lead], transform=ax.transAxes, va="top", color=MUTED)
                axes[row, 0].set_ylabel(f"{label}\nz-units")
            for ax in axes[-1]:
                ax.set_xlabel("time from QRS (s)")
            axes[0, 0].set_title(it["title"], loc="left")
        lines = observed + decoded
        fig.legend([Line2D([], [], color=c, lw=lw) for _, c, lw, _ in lines], [label for label, *_ in lines],
                   loc="upper center", ncol=len(lines), bbox_to_anchor=(0.5, 1 - 0.3 / fig.get_figheight()))
        fig.suptitle(title, y=1, color=INK)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    paths = data_paths(args.config)
    lance_path, pairs_path = paths["lance_path"], paths["pairs_path"]
    vocab = json.loads(Path(paths["vocabulary_path"]).read_text())
    ck = torch.load(args.decoder, map_location="cuda", weights_only=False)
    spec, name = ck["spec"], Path(args.decoder).stem.removeprefix("decoder_")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = (lance.dataset(pairs_path)
             .to_table(filter=f"(fold = {SPLITS[args.split]}) AND (pair_type = 'cross_stay')",
                       columns=["idx_t", "idx_t1", "subject_id"]).to_pandas())
    rows = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    cache = Path(args.decoder).parent / f"morphology_cache_{name}_{args.split}.npz"
    beats, n_beats, H = beats_and_embeddings(rows, spec, lance_path, cache, args.batch_size)
    meta = lance.dataset(lance_path).take(rows.tolist(), columns=["icd", "ecg_time"])
    y = meta.column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(rows), -1)
    days = (meta.column("ecg_time").to_numpy() / 86400)                   # ecg_time is in seconds
    i0, i1 = np.searchsorted(rows, pairs["idx_t"].values), np.searchsorted(rows, pairs["idx_t1"].values)

    dec = BeatDecoder(ck["h_mean"].numel(), ck["hidden"]).cuda()
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    read = Readout(dec, ck["h_mean"].cuda(), ck["h_std"].cuda(), ck["rhythm_mean"].cuda(), ck["rhythm_std"].cuda())
    dyn = load_dynamics(spec)
    assert dyn is not None, f"{spec} has no forward model"

    yt = torch.from_numpy(y[i0]).float().cuda()
    at = torch.from_numpy(y[i1]).float().cuda() - yt
    ht = torch.from_numpy(H[i0].astype(np.float32)).cuda()
    Bt, Bt1 = (torch.from_numpy(beats[i]).float().cuda() for i in (i0, i1))
    with torch.no_grad():
        est = {"ceiling": read(torch.from_numpy(H[i1].astype(np.float32)).cuda())[0], "nothing": read(ht)[0],
               "dyn": read(dyn(ht, at))[0], "zero": read(dyn(ht, torch.zeros_like(at)))[0]}
        if args.decoy_action:
            decoy, has_decoy = strict_decoys(yt, at, torch.Generator(device="cuda").manual_seed(args.seed),
                                             disjoint=args.disjoint_decoy)
            est["decoy"] = read(dyn(ht, decoy))[0]
            est["decoy"][~has_decoy] = float("nan")                         # nothing to draw or score

    # Drawn leads per pair, then shape on them: correlation of the per-lead mean-centred beats (the figures and the
    # beat MSEs use the beats as decoded)
    centred = lambda b: b - b.mean(-1, keepdim=True)
    levels = lambda b: (b.sort(-1).values.diff(dim=-1) != 0).sum(-1) + 1   # (N, 12) distinct values per lead
    ok = (levels(Bt) >= args.min_levels) & (levels(Bt1) >= args.min_levels)
    if args.leads:
        lead_idx = torch.tensor([LEADS.index(lead) for lead in args.leads], device="cuda").expand(len(Bt), -1)
        drawable = ok[:, lead_idx[0]].all(1)
    else:
        lead_change = (centred(Bt1) - centred(Bt)).pow(2).mean(-1)          # (N, 12)
        lead_idx = lead_change.masked_fill(~ok, -1).topk(args.n_leads, dim=1).indices.sort(1).values
        drawable = ok.sum(1) >= args.n_leads
    pick = lambda b: centred(b)[torch.arange(len(b), device="cuda")[:, None], lead_idx].flatten(1)
    corr = lambda a, b: F.cosine_similarity(pick(a), pick(b), dim=1)
    change = 1 - corr(Bt, Bt1)
    r_next = {c: corr(e, Bt1) for c, e in est.items()}
    r_prev = {c: corr(e, Bt) for c, e in est.items()}
    shape = {c: r_next[c] - r_prev[c] for c in est}
    mse = {c: (e - Bt1).pow(2).mean((1, 2)) for c, e in est.items()}
    if args.decoy_action:
        # What each action adds to the state map, decoded Dyn(h(t), .) - decoded Dyn(h(t), 0) on the drawn leads:
        # an action-specific model gives the decoy an effect unlike the true action's
        e_true, e_decoy = pick(est["dyn"]) - pick(est["zero"]), pick(est["decoy"]) - pick(est["zero"])
        effect_corr = F.cosine_similarity(e_true, e_decoy, dim=1)
        effect_size = e_decoy.norm(dim=1) / e_true.norm(dim=1).clamp_min(1e-8)

    usable = torch.from_numpy((n_beats[i0] > 0) & (n_beats[i1] > 0)).cuda() & drawable
    pool = usable if args.include_stable else usable & (at != 0).any(1)
    big = pool & (change >= torch.quantile(change[pool], 1 - args.top_change))
    keep = big & (shape["nothing"] <= -args.margin) & (shape["ceiling"] >= args.margin)
    groups = (("usable" if args.include_stable else "active", pool), (f"top {args.top_change:.0%} shape change", big),
              ("decoder tells shapes apart", keep))
    print(f"{len(pairs):,} {args.split} pairs, {int(usable.sum()):,} with usable beats on their drawn leads")
    print(f"{'':<28}{'pairs':>7}{'Dyn MSE < do-nothing':>22}{'Dyn shape > 0':>15}{'Dyn(h(t), 0) shape > 0':>24}")
    for label, m in groups:
        frac = lambda x: f"{x[m].float().mean().item():.0%}" if m.any() else "-"
        print(f"{label:<28}{int(m.sum()):>7,}{frac(mse['dyn'] < mse['nothing']):>22}{frac(shape['dyn'] > 0):>15}"
              f"{frac(shape['zero'] > 0):>24}")
    if keep.any():
        print("mean shape(E) = corr(E, X(t+1)) - corr(E, X(t)) over those: "
              + ", ".join(f"{c} {shape[c][keep].mean().item():+.2f}" for c in ("ceiling", "dyn", "zero", "nothing")))
    if args.decoy_action:
        print(f"decoy vs true action; effect = decoded Dyn(h(t), .) - decoded Dyn(h(t), 0) on the drawn leads")
        print(f"{'':<28}{'pairs':>7}{'effect corr (median)':>22}{'decoy/true size':>17}{'decoy shape > 0':>17}")
        for label, m in groups:
            m = m & has_decoy                                               # pairs with a different decoy
            if m.any():
                print(f"{label:<28}{int(m.sum()):>7,}{effect_corr[m].median().item():>22.2f}"
                      f"{effect_size[m].median().item():>17.2f}{(shape['decoy'][m] > 0).float().mean().item():>17.0%}")

    idx = torch.nonzero(keep).squeeze(1)
    ranked = idx[shape["dyn"][idx].argsort(descending=True)].cpu().numpy()     # every kept pair, best first
    n_top = min(2 * args.n_figures, len(ranked))
    figure_of = {r: f"{name}_{args.split}_morphology_{(r + 1) // 2:02d}.png" for r in range(1, n_top + 1)}
    if args.contrast:   # the middle and the last of the ranking, unless the top figures already show them
        figure_of.update({r: f"{name}_{args.split}_morphology_middle_bottom.png"
                          for r in sorted({(len(ranked) + 1) // 2, len(ranked)}) if r > n_top})
    at_np = at.cpu().numpy()
    decoy_np = decoy.cpu().numpy() if args.decoy_action else None
    items, records = [], []
    for rank, fname in figure_of.items():
        k = ranked[rank - 1]
        leads = lead_idx[k].tolist()
        rn = {c: r_next[c][k].item() for c in est}
        rp = {c: r_prev[c][k].item() for c in est}
        m = {c: mse[c][k].item() for c in mse}
        gap = days[i1[k]] - days[i0[k]]
        flips = _flips(at_np[k], vocab) or "no diagnosis change"
        zero = f", Dyn(h(t), 0) {rn['zero']:.2f}/{rp['zero']:.2f}" if args.zero_action else ""
        title = (f"{rank}. {flips}  ({gap:.0f} days apart)\ncorrelation with X(t+1)/X(t): decoded h(t+1) "
                 f"{rn['ceiling']:.2f}/{rp['ceiling']:.2f}, Dyn {rn['dyn']:.2f}/{rp['dyn']:.2f}{zero}, do-nothing "
                 f"{rn['nothing']:.2f}/{rp['nothing']:.2f}   beat MSE to X(t+1): Dyn {m['dyn']:.3f}, do-nothing "
                 f"{m['nothing']:.3f}, ceiling {m['ceiling']:.3f}")
        extra = {}
        if args.decoy_action and has_decoy[k]:
            extra = {"decoy": _flips(decoy_np[k], vocab, max_shown=len(vocab)),
                     "decoy_effect_corr": effect_corr[k].item(), "decoy_effect_size": effect_size[k].item()}
            title += (f"\ndecoy {_flips(decoy_np[k], vocab)}: correlation with X(t+1)/X(t) {rn['decoy']:.2f}/"
                      f"{rp['decoy']:.2f}   its effect vs a(t)'s, Dyn(h(t), .) - Dyn(h(t), 0): correlation "
                      f"{extra['decoy_effect_corr']:.2f}, size {extra['decoy_effect_size']:.2f}x")
        elif args.decoy_action:
            title += "\nno decoy: a(t) is the only admissible action with its onset and resolution counts"
        items.append({"title": title, "rank": rank, "figure": fname, "contrast": rank > n_top, "leads": leads,
                      "Bt": Bt[k].cpu().numpy(), "Bt1": Bt1[k].cpu().numpy(),
                      **{c: e[k].cpu().numpy() for c, e in est.items()}})
        records.append({"rank": rank, "figure": fname, "idx_t": int(pairs["idx_t"].iloc[k]),
                        "idx_t1": int(pairs["idx_t1"].iloc[k]), "subject_id": int(pairs["subject_id"].iloc[k]),
                        "days_apart": round(float(gap), 1), "flips": _flips(at_np[k], vocab, max_shown=len(vocab)),
                        "leads": " ".join(LEADS[lead] for lead in leads), "shape_change": change[k].item(),
                        **{f"corr_next_{c}": v for c, v in rn.items()}, **{f"corr_prev_{c}": v for c, v in rp.items()},
                        **{f"mse_{c}": v for c, v in m.items()}, **extra})
    if not records:
        print("No pair passes the filters; loosen --top-change or --margin.")
        return
    table = pd.DataFrame(records)
    table.to_csv(out_dir / f"{name}_{args.split}_selected_pairs.csv", index=False)
    for c in est:
        table[f"shape_{c}"] = table[f"corr_next_{c}"] - table[f"corr_prev_{c}"]
    columns = ["rank", "days_apart", "leads", "shape_change", "shape_ceiling", "shape_dyn", "shape_zero",
               "shape_nothing", "mse_dyn", "mse_nothing", "mse_ceiling"]
    if args.decoy_action:
        columns += ["shape_decoy", "decoy_effect_corr", "decoy_effect_size"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(table[[c for c in columns if c in table]].to_string(index=False))

    decoded = [("decoded h(t+1), the ceiling", AQUA, 1.5, "ceiling"), ("decoded h(t), do-nothing", ORANGE, 1.5, "nothing"),
               ("decoded Dyn(h(t), a(t))", BLUE, 1.5, "dyn")]
    if args.zero_action:
        decoded.append(("decoded Dyn(h(t), 0)", YELLOW, 1.5, "zero"))
    if args.decoy_action:
        decoded.append(("decoded Dyn(h(t), decoy)", MAGENTA, 1.5, "decoy"))
    figures = {}
    for it in items:
        figures.setdefault(it["figure"], []).append(it)
    for fname, part in figures.items():
        ranks = ("rank " if len(part) == 1 else "ranks ") + " and ".join(str(it["rank"]) for it in part)
        what = "the middle and the bottom of the ranking" if part[0]["contrast"] else "selected"
        plot_figure(out_dir / fname, f"{name}: {args.split} pairs with a large change in beat morphology, {what} "
                    f"({ranks} of {len(ranked)})", part, decoded)
    print(f"Saved {len(figures)} figures and the pair list to {out_dir}")


if __name__ == "__main__":
    main()
