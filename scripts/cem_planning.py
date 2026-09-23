"""
CEM Planning with the Forward Dynamics Model
==============================================
Diagnosis as inference-time search (papel/metodo.tex, Planning). For each test pair
(Xt, Xt+1, yt), with ht = Encθ(Xt) and ht1 = Encθ(Xt+1), the Cross-Entropy Method
searches admissible transitions a for the lowest

    E(a) = mean_D ||Dynϕ(ht, Projω(a)) - target||^2

where target = ht1 for next-state checkpoints (scripts/pretrain.py) and ht1 - ht for
displacement checkpoints (scripts/pretrain_displacement.py), so E is always the MSE of
the implied next state against ht1. With cem.temperature = τ the search instead
minimizes the negative log-posterior (up to a constant)

    F(a) = E(a) / τ + Σ_i [a_i ≠ 0] · log((1 - π_i) / π_i)

so every flip pays its prior log-odds, π_i being the train base rate of label i's live
head. Plain E(a) is readily exploited: action combinations far from anything seen in
training can fit ht1 better than the true transition.

Only admissible transitions are searched: given yt, label i can only onset (a_i = +1,
needs yt_i = 0) or resolve (a_i = -1, needs yt_i = 1). Mirroring the Inverse Dynamics
onset/resolution heads, the proposal is two masked Bernoulli heads,

    u_ons ~ Bern(p_ons) * (1 - yt),    u_res ~ Bern(p_res) * yt,    a = u_ons - u_res

initialised at the train split's per-class onset / resolution base rates. Candidates
are scored jointly (one population, one elite set) and each head is refit only on its
admissible entries.

AUROC, on the same pairs and bootstrap as scripts/evaluate_inverse.py:
  1. Contextualized (soft) — yt*(1-p_res) + (1-yt)*p_ons from the final CEM marginals
  2. Contextualized (hard) — clip(yt + a*, 0, 1)
  3. Carry-forward         — yt  (reproduces scripts/baseline_carry_forward.py)
  4. Onset                 — p_ons vs (at_i == +1), over entries with yt_i = 0 only
  5. Resolution            — p_res vs (at_i == -1), over entries with yt_i = 1 only

Example
-------
HYDRA_FULL_ERROR=1 python scripts/cem_planning.py \\
    ckpt_path=archive/pretrain_displacement.ckpt cem.temperature=0.01
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import lance
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionProjector, DynamicsPredictor


class OnsetResolutionCEM:
    """Cross-Entropy Method over admissible transitions a ∈ {-1,0,1}^C.

    Given yt, every (pair, label) entry has exactly one live head: onset where
    yt_i = 0, resolution where yt_i = 1. Each head keeps an independent Bernoulli per
    entry; a candidate combines both heads into one action and is scored as a whole,
    so both refits see the same elite set.

    Candidates are ranked by E(a) when temperature is None, else by
    F(a) = E(a) / temperature + prior flip cost (see the module docstring).
    """

    def __init__(self, n_samples: int, n_elites: int, n_iters: int, smoothing: float, p_min: float,
                 temperature: float | None = None):
        assert 0 < n_elites <= n_samples, "need 0 < n_elites <= n_samples"
        assert temperature is None or temperature > 0, "temperature must be positive (or None)"
        self.N = n_samples
        self.K = n_elites
        self.T = n_iters
        self.alpha = smoothing  # weight kept on the previous distribution
        self.p_min = p_min
        self.temperature = temperature

    @staticmethod
    def flip_costs(prior_ons, prior_res):
        """Prior log-odds against flipping each label, log((1 - π) / π), for each head."""
        return torch.log1p(-prior_ons) - torch.log(prior_ons), torch.log1p(-prior_res) - torch.log(prior_res)

    def objective(self, energies, actions, w_ons, w_res):
        """Ranking score of candidates actions (B, N, C) with energies (B, N)."""
        if self.temperature is None:
            return energies
        cost = (actions == 1).float() @ w_ons + (actions == -1).float() @ w_res   # (B, N)
        return energies / self.temperature + cost

    @torch.no_grad()
    def plan(self, energy_fn, yt, prior_ons, prior_res, generator):
        """
        Args:
            energy_fn: actions (B, N, C) in {-1,0,1} -> energies (B, N)
            yt:        (B, C) current labels in {0,1}
            prior_ons: (C,) P(onset)      for entries with yt_i = 0
            prior_res: (C,) P(resolution) for entries with yt_i = 1

        Returns a dict with p_ons, p_res (B, C; NaN where the head is inactive),
        a_star (B, C), its energy e_star and objective f_star, e_zero and f_zero for the
        do-nothing transition (all (B,)), and per-iteration means in history.
        """
        B, C = yt.shape
        m_ons = yt == 0
        m_res = yt == 1
        w_ons, w_res = self.flip_costs(prior_ons, prior_res)
        p_ons = prior_ons.expand(B, C).clamp(self.p_min, 1 - self.p_min)
        p_res = prior_res.expand(B, C).clamp(self.p_min, 1 - self.p_min)

        # The do-nothing transition seeds the incumbent, so F(a*) <= F(0) always holds.
        a_star = torch.zeros(B, C, device=yt.device)
        e_zero = energy_fn(a_star.unsqueeze(1)).squeeze(1)
        f_zero = self.objective(e_zero.unsqueeze(1), a_star.unsqueeze(1), w_ons, w_res).squeeze(1)
        e_star, f_star = e_zero.clone(), f_zero.clone()

        history = {"best_energy": [], "best_objective": [], "entropy": [], "undecided": []}
        for _ in range(self.T):
            u_ons = torch.bernoulli(p_ons.unsqueeze(1).expand(B, self.N, C), generator=generator) * m_ons.unsqueeze(1)
            u_res = torch.bernoulli(p_res.unsqueeze(1).expand(B, self.N, C), generator=generator) * m_res.unsqueeze(1)
            actions = u_ons - u_res                                     # (B, N, C), admissible by construction
            energies = energy_fn(actions)                               # (B, N)
            scores = self.objective(energies, actions, w_ons, w_res)    # (B, N)

            elite = scores.topk(self.K, dim=1, largest=False).indices  # (B, K)
            elite = elite.unsqueeze(-1).expand(B, self.K, C)
            p_ons = torch.where(m_ons, self.alpha * p_ons + (1 - self.alpha) * u_ons.gather(1, elite).mean(1), p_ons)
            p_res = torch.where(m_res, self.alpha * p_res + (1 - self.alpha) * u_res.gather(1, elite).mean(1), p_res)
            p_ons = p_ons.clamp(self.p_min, 1 - self.p_min)
            p_res = p_res.clamp(self.p_min, 1 - self.p_min)

            best_f, best_i = scores.min(dim=1)
            improved = best_f < f_star
            rows = torch.arange(B, device=yt.device)
            f_star = torch.where(improved, best_f, f_star)
            e_star = torch.where(improved, energies[rows, best_i], e_star)
            a_star = torch.where(improved.unsqueeze(1), actions[rows, best_i], a_star)

            p_live = torch.where(m_ons, p_ons, p_res)
            entropy = torch.special.entr(p_live) + torch.special.entr(1 - p_live)
            history["best_energy"].append(e_star.mean().item())
            history["best_objective"].append(f_star.mean().item())
            history["entropy"].append(entropy.mean().item())
            history["undecided"].append(((p_live > 0.1) & (p_live < 0.9)).float().mean().item())

        nan = torch.tensor(float("nan"), device=yt.device)
        return {
            "p_ons":   torch.where(m_ons, p_ons, nan),
            "p_res":   torch.where(m_res, p_res, nan),
            "a_star":  a_star,
            "e_star":  e_star,
            "f_star":  f_star,
            "e_zero":  e_zero,
            "f_zero":  f_zero,
            "history": history,
        }


def _load_model(ckpt_path, embedding_dim, predictor_hidden_dim, action_dim, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    backbone = ResNet1d(in_channels=12, embedding_dim=embedding_dim)
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    )

    projector = ActionProjector(action_dim=action_dim, embed_dim=embedding_dim)
    projector.load_state_dict(
        {k[len("projector.proj."):]: v for k, v in sd.items() if k.startswith("projector.proj.")}
    )

    predictor = DynamicsPredictor(embed_dim=embedding_dim, hidden_dim=predictor_hidden_dim)
    predictor.load_state_dict(
        {k[len("projector.pred."):]: v for k, v in sd.items() if k.startswith("projector.pred.")}
    )

    for module in (backbone, projector, predictor):
        module.to(device).eval()
    return backbone, projector, predictor, ckpt.get("hyper_parameters") or {}


def _resolve_dyn_target(dyn_target, hparams):
    if dyn_target != "auto":
        return dyn_target
    # pretrain.py and pretrain_displacement.py save identical state_dict layouts; the
    # training run's ckpt_path is the only record of which loss produced the weights.
    return "displacement" if "displacement" in str(hparams.get("ckpt_path", "")) else "next_state"


def _transition_counts(lance_path, pairs_path, pair_types, action_dim):
    """Per-class onset / resolution counts over the train split's pairs, from labels only."""
    type_filter = " OR ".join(f"pair_type = '{t}'" for t in pair_types)
    pairs_df = (
        lance.dataset(pairs_path)
        .to_table(filter=f"(fold <= 17) AND ({type_filter})", columns=["idx_t", "idx_t1"])
        .to_pandas()
    )

    unique_idx = np.unique(np.concatenate([pairs_df["idx_t"].values, pairs_df["idx_t1"].values]))
    labels = (
        lance.dataset(lance_path).take(unique_idx.tolist(), columns=["icd"])
        .column("icd").combine_chunks().flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(len(unique_idx), action_dim)
    )
    yt  = labels[np.searchsorted(unique_idx, pairs_df["idx_t"].values)]
    yt1 = labels[np.searchsorted(unique_idx, pairs_df["idx_t1"].values)]

    return {
        "n_pairs":   len(pairs_df),
        "onset":     ((yt == 0) & (yt1 == 1)).sum(0),
        "absent":    (yt == 0).sum(0),
        "resolution": ((yt == 1) & (yt1 == 0)).sum(0),
        "present":   (yt == 1).sum(0),
    }


def _priors(counts, prior):
    """Laplace-smoothed P(onset | yt=0) and P(resolution | yt=1), per class or pooled."""
    if prior == "base_rate":
        p_ons = (counts["onset"] + 1) / (counts["absent"] + 2)
        p_res = (counts["resolution"] + 1) / (counts["present"] + 2)
    elif prior == "pooled":
        n_classes = len(counts["onset"])
        p_ons = np.full(n_classes, (counts["onset"].sum() + 1) / (counts["absent"].sum() + 2))
        p_res = np.full(n_classes, (counts["resolution"].sum() + 1) / (counts["present"].sum() + 2))
    else:
        raise ValueError(f"Unknown prior: {prior!r} (expected base_rate | pooled)")
    return p_ons.astype(np.float32), p_res.astype(np.float32)


@torch.no_grad()
def _encode(backbone, loader, max_batches, device):
    """Embeds both ECGs of every pair; returns ht, ht1 (N, D) and yt, at (N, C) on CPU."""
    ht, ht1, yt, at = [], [], [], []
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    for i, batch in enumerate(tqdm(loader, desc="Encoding", total=n_batches)):
        if i >= n_batches:
            break
        ht.append(backbone(batch["xt"].to(device)).cpu())
        ht1.append(backbone(batch["xt1"].to(device)).cpu())
        yt.append(batch["yt"].float())
        at.append(batch["at"].float())
    return {"ht": torch.cat(ht), "ht1": torch.cat(ht1), "yt": torch.cat(yt), "at": torch.cat(at)}


def _make_energy_fn(projector, predictor, ht, target, chunk_size):
    """E(a) = mean_D ||Dyn(ht, Proj(a)) - target||^2 for every candidate in actions (B, N, C)."""
    def energy_fn(actions):
        B, N, C = actions.shape
        flat = actions.reshape(B * N, C)
        rows = torch.arange(B * N, device=actions.device) // N   # candidate -> pair
        energies = torch.empty(B * N, device=actions.device)
        for s in range(0, B * N, chunk_size):
            r = rows[s:s + chunk_size]
            pred = predictor(ht[r], projector(flat[s:s + chunk_size]))
            energies[s:s + chunk_size] = (pred - target[r]).pow(2).mean(dim=1)
        return energies.view(B, N)
    return energy_fn


@torch.no_grad()
def _run_planning(emb, projector, predictor, planner, dyn_target, prior_ons, prior_res,
                  batch_size, chunk_size, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    w_ons, w_res = planner.flip_costs(prior_ons, prior_res)
    keys = ("p_ons", "p_res", "a_star", "e_star", "f_star", "e_zero", "f_zero", "e_gt", "f_gt",
            "do_nothing", "yt", "at")
    out = {k: [] for k in keys}
    history, n = None, len(emb["yt"])

    for s in tqdm(range(0, n, batch_size), desc="CEM planning"):
        ht  = emb["ht"][s:s + batch_size].to(device)
        ht1 = emb["ht1"][s:s + batch_size].to(device)
        yt  = emb["yt"][s:s + batch_size].to(device)
        at  = emb["at"][s:s + batch_size].to(device)

        target = ht1 - ht if dyn_target == "displacement" else ht1
        energy_fn = _make_energy_fn(projector, predictor, ht, target, chunk_size)

        res = planner.plan(energy_fn, yt, prior_ons, prior_res, generator)

        a_star = res["a_star"]
        assert not ((a_star == 1) & (yt == 1)).any() and not ((a_star == -1) & (yt == 0)).any(), \
            "planner returned an inadmissible action"

        # Diagnostics only — never seen by the search.
        e_gt = energy_fn(at.unsqueeze(1)).squeeze(1)
        f_gt = planner.objective(e_gt.unsqueeze(1), at.unsqueeze(1), w_ons, w_res).squeeze(1)
        do_nothing = (ht1 - ht).pow(2).mean(dim=1)

        values = {**{k: res[k] for k in keys if k in res},
                  "e_gt": e_gt, "f_gt": f_gt, "do_nothing": do_nothing, "yt": yt, "at": at}
        for k in keys:
            out[k].append(values[k].cpu().float().numpy())

        B = yt.shape[0]
        batch_hist = {k: np.array(v) * B for k, v in res["history"].items()}
        history = batch_hist if history is None else {k: history[k] + batch_hist[k] for k in history}

    out = {k: np.concatenate(v) for k, v in out.items()}
    history = {k: (v / n).tolist() for k, v in (history or {}).items()}
    return out, history


def _bootstrap_auroc(preds, targets, n_bootstrap, seed, desc, mask=None):
    """Same resampling as scripts/evaluate_inverse.py; mask (N, C) restricts each class's rows."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n, n_classes = targets.shape
    scores = []
    for _ in tqdm(range(n_bootstrap), desc=desc):
        idx = rng.integers(0, n, size=n)
        t, p = targets[idx], preds[idx]
        m = None if mask is None else mask[idx]
        per_class = []
        for c in range(n_classes):
            tc, pc = (t[:, c], p[:, c]) if m is None else (t[m[:, c], c], p[m[:, c], c])
            if 0 < tc.sum() < len(tc):
                per_class.append(roc_auc_score(tc, pc))
        if per_class:
            scores.append(np.mean(per_class))
    if not scores:
        return float("nan"), float("nan"), float("nan")
    scores = np.array(scores)
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _macro_auroc(preds, targets):
    from sklearn.metrics import roc_auc_score

    per_class = [
        roc_auc_score(targets[:, c], preds[:, c])
        for c in range(targets.shape[1])
        if 0 < targets[:, c].sum() < len(targets)
    ]
    return (float(np.mean(per_class)) if per_class else float("nan")), len(per_class)


def _ratio(num, den):
    return float(num) / float(den) if den else float("nan")


@hydra.main(version_base="1.3", config_path="../configs", config_name="cem_planning")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert cfg.dyn_target in ("auto", "next_state", "displacement"), f"Unknown dyn_target: {cfg.dyn_target!r}"
    pair_types = tuple(cfg.pair_types)

    ckpt_path = Path(get_original_cwd()) / cfg.ckpt_path
    print(f"Checkpoint: {ckpt_path}")
    backbone, projector, predictor, hparams = _load_model(
        ckpt_path, cfg.embedding_dim, cfg.predictor_hidden_dim, cfg.action_dim, device
    )
    dyn_target = _resolve_dyn_target(cfg.dyn_target, hparams)
    print(f"Dynamics target: {dyn_target}  "
          f"(dyn_target={cfg.dyn_target}; trained as {hparams.get('ckpt_path', '?')}, "
          f"pair_types={hparams.get('pair_types', '?')}, pairs={Path(str(hparams.get('pairs_path', '?'))).name})")

    counts = _transition_counts(cfg.lance_path, cfg.pairs_path, pair_types, cfg.action_dim)
    p_ons_np, p_res_np = _priors(counts, cfg.cem.prior)
    print(f"Priors ({cfg.cem.prior}, {counts['n_pairs']:,} train pairs): "
          f"onset mean {p_ons_np.mean():.4f} / median {np.median(p_ons_np):.4f}   "
          f"resolution mean {p_res_np.mean():.4f} / median {np.median(p_res_np):.4f}")
    prior_ons = torch.from_numpy(p_ons_np).to(device)
    prior_res = torch.from_numpy(p_res_np).to(device)

    # Embeddings depend only on the checkpoint and the pairs, so a temperature sweep can
    # reuse them. The key pins both Lance versions: a rewritten pairs table misses the cache.
    cache_path = None
    if cfg.cache_embeddings and cfg.max_batches is None:
        stat = ckpt_path.stat()
        key = (f"{ckpt_path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
               f"_{Path(cfg.pairs_path).stem}_v{lance.dataset(cfg.pairs_path).version}"
               f"_ecg_v{lance.dataset(cfg.lance_path).version}_{cfg.split}_{'+'.join(pair_types)}")
        cache_path = Path(get_original_cwd()) / cfg.cache_dir / f"{key}.pt"

    if cache_path is not None and cache_path.exists():
        emb = torch.load(cache_path)
        print(f"Loaded cached embeddings: {cache_path}")
    else:
        ds = MIMICLanceDataset(
            cfg.lance_path,
            split=cfg.split,
            mode="pair",
            pairs_path=cfg.pairs_path,
            pair_types=pair_types,
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            pin_memory=True,
        )
        emb = _encode(backbone, loader, cfg.max_batches, device)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, cache_path)
    print(f"{cfg.split.capitalize()} pairs: {len(emb['yt'])}")

    planner = OnsetResolutionCEM(
        n_samples=cfg.cem.n_samples,
        n_elites=cfg.cem.n_elites,
        n_iters=cfg.cem.n_iters,
        smoothing=cfg.cem.smoothing,
        p_min=cfg.cem.p_min,
        temperature=cfg.cem.temperature,
    )
    print(f"Objective: {'E(a)' if planner.temperature is None else f'E(a)/{planner.temperature:g} + prior flip cost'}")
    out, history = _run_planning(
        emb, projector, predictor, planner, dyn_target, prior_ons, prior_res,
        cfg.batch_size, cfg.cem.chunk_size, cfg.evaluate.seed, device,
    )

    yt, at, a_star = out["yt"], out["at"], out["a_star"]
    yt1    = np.clip(yt + at, 0.0, 1.0)                          # (N, 76) true next labels
    m_ons  = yt == 0
    m_res  = yt == 1
    p_ons  = np.nan_to_num(out["p_ons"])                        # NaN (inactive head) -> 0; masked below
    p_res  = np.nan_to_num(out["p_res"])
    soft   = np.where(m_res, 1.0 - p_res, p_ons)                 # yt*(1-p_res) + (1-yt)*p_ons
    hard   = np.clip(yt + a_star, 0.0, 1.0)

    n_bootstrap = cfg.evaluate.n_bootstrap
    seed        = cfg.evaluate.seed
    auroc = {}

    print()
    for name, preds, targets, mask in (
        ("contextualized_soft", soft,  yt1,                           None),
        ("contextualized_hard", hard,  yt1,                           None),
        ("carry_forward",       yt,    yt1,                           None),
        ("onset",               p_ons, (at == 1.0).astype(np.float32),  m_ons),
        ("resolution",          p_res, (at == -1.0).astype(np.float32), m_res),
    ):
        mean, lo, hi = _bootstrap_auroc(preds, targets, n_bootstrap, seed, f"{name} bootstrap", mask)
        auroc[name] = {"mean": mean, "lo": lo, "hi": hi}

    print()
    labels = {
        "contextualized_soft": "Contextualized (soft)",
        "contextualized_hard": "Contextualized (hard)",
        "carry_forward":       "Carry-forward",
        "onset":               "Onset      (yt=0 only)",
        "resolution":          "Resolution (yt=1 only)",
    }
    for name, label in labels.items():
        r = auroc[name]
        print(f"{label:<23s} AUROC: {r['mean']:.4f}  (95% CI: {r['lo']:.4f}–{r['hi']:.4f})")

    # ---- Diagnostics ---------------------------------------------------------------
    e_zero, e_star, e_gt = out["e_zero"], out["e_star"], out["e_gt"]
    f_zero, f_star, f_gt = out["f_zero"], out["f_star"], out["f_gt"]
    gt_active   = np.abs(at).sum(1) > 0
    pred_active = np.abs(a_star).sum(1) > 0
    exact       = (a_star == at).all(1)

    sanity = {"do_nothing": float(out["do_nothing"].mean()), "e_zero": float(e_zero.mean())}
    print(f"\nTarget sanity:  do-nothing ||ht1-ht||^2 {sanity['do_nothing']:.4f}   E(0) {sanity['e_zero']:.4f}")
    if sanity["e_zero"] > 2 * sanity["do_nothing"]:
        print("  WARNING: E(0) is far above the do-nothing error — dyn_target is probably wrong "
              "(override with dyn_target=next_state|displacement).")

    act = gt_active
    headroom = {
        "n_active":             int(act.sum()),
        "p_gt_beats_zero":      _ratio((e_gt[act] < e_zero[act]).sum(), act.sum()),
        "mean_gap":             float((e_zero[act] - e_gt[act]).mean()) if act.any() else float("nan"),
        "p_gt_beats_zero_objective": _ratio((f_gt[act] < f_zero[act]).sum(), act.sum()),
    }
    # Judged on the objective actually minimized (F, which is E when temperature is None).
    failures = {
        "search_failure": _ratio((f_star[act] > f_gt[act]).sum(), act.sum()),
        "model_failure":  _ratio(((f_star[act] <= f_gt[act]) & ~exact[act]).sum(), act.sum()),
        "exact":          _ratio(exact[act].sum(), act.sum()),
    }
    print(f"Headroom (GT-active pairs, n={headroom['n_active']}): "
          f"P(E(a_gt) < E(0)) = {headroom['p_gt_beats_zero']:.3f}  (chance 0.5);  "
          f"mean E(0)-E(a_gt) = {headroom['mean_gap']:+.5f}")
    if planner.temperature is not None:
        print(f"  under the objective: P(F(a_gt) < F(0)) = {headroom['p_gt_beats_zero_objective']:.3f}")
    print(f"GT-active outcomes: search failure [F(a*) > F(a_gt)] {failures['search_failure']:.3f}  "
          f"model failure [F(a*) <= F(a_gt), a* != a_gt] {failures['model_failure']:.3f}  "
          f"exact [a* == a_gt] {failures['exact']:.3f}")

    if history.get("best_energy"):
        print(f"\n{'iter':>4}  {'best E':>10}  {'best F':>12}  {'entropy':>8}  {'undecided':>9}")
        for t, (e, f, h, u) in enumerate(zip(history["best_energy"], history["best_objective"],
                                             history["entropy"], history["undecided"])):
            print(f"{t + 1:>4}  {e:>10.5f}  {f:>12.4f}  {h:>8.4f}  {u:>9.4f}")

    tp_ons = ((a_star == 1) & (at == 1)).sum()
    tp_res = ((a_star == -1) & (at == -1)).sum()
    actions = {
        "confusion": {
            "pred_stable_gt_stable": int((~pred_active & ~gt_active).sum()),
            "pred_stable_gt_active": int((~pred_active & gt_active).sum()),
            "pred_active_gt_stable": int((pred_active & ~gt_active).sum()),
            "pred_active_gt_active": int((pred_active & gt_active).sum()),
        },
        "onsets_per_pair":      {"pred": float((a_star == 1).sum(1).mean()),  "gt": float((at == 1).sum(1).mean())},
        "resolutions_per_pair": {"pred": float((a_star == -1).sum(1).mean()), "gt": float((at == -1).sum(1).mean())},
        "onset_precision":      _ratio(tp_ons, (a_star == 1).sum()),
        "onset_recall":         _ratio(tp_ons, (at == 1).sum()),
        "resolution_precision": _ratio(tp_res, (a_star == -1).sum()),
        "resolution_recall":    _ratio(tp_res, (at == -1).sum()),
    }
    c = actions["confusion"]
    print(f"\n{'':22s}{'GT stable':>10}{'GT active':>10}")
    print(f"{'Pred stable (a*=0)':22s}{c['pred_stable_gt_stable']:>10d}{c['pred_stable_gt_active']:>10d}")
    print(f"{'Pred active':22s}{c['pred_active_gt_stable']:>10d}{c['pred_active_gt_active']:>10d}")
    print(f"Per pair: onsets {actions['onsets_per_pair']['pred']:.3f} pred / {actions['onsets_per_pair']['gt']:.3f} GT   "
          f"resolutions {actions['resolutions_per_pair']['pred']:.3f} pred / {actions['resolutions_per_pair']['gt']:.3f} GT")
    print(f"Flips: onset precision {actions['onset_precision']:.3f} recall {actions['onset_recall']:.3f}   "
          f"resolution precision {actions['resolution_precision']:.3f} recall {actions['resolution_recall']:.3f}")

    by_group = {}
    print(f"\n{'Group':<10}{'N':>7}{'classes':>9}{'CEM soft':>10}{'Carry-fwd':>11}  (point estimates)")
    for group, mask in (("gt_stable", ~gt_active), ("gt_active", gt_active)):
        if not mask.any():
            continue
        auroc_cem, n_cls = _macro_auroc(soft[mask], yt1[mask])
        auroc_cf, _      = _macro_auroc(yt[mask], yt1[mask])
        by_group[group] = {"n": int(mask.sum()), "classes": n_cls, "cem_soft": auroc_cem, "carry_forward": auroc_cf}
        print(f"{group:<10}{int(mask.sum()):>7d}{n_cls:>9d}{auroc_cem:>10.4f}{auroc_cf:>11.4f}")

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dyn_target": dyn_target,
        "n_pairs": int(len(yt)),
        "auroc": auroc,
        "priors": {
            "n_train_pairs": int(counts["n_pairs"]),
            "onset": p_ons_np.tolist(),
            "resolution": p_res_np.tolist(),
        },
        "sanity": sanity,
        "headroom": headroom,
        "gt_active_outcomes": failures,
        "actions": actions,
        "auroc_by_group": by_group,
        "history": history,
    }
    with open(out_dir / "cem_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if cfg.save_arrays:
        np.savez_compressed(out_dir / "cem_arrays.npz", **{k: out[k] for k in
                            ("p_ons", "p_res", "a_star", "e_zero", "e_star", "e_gt", "f_zero", "f_star", "f_gt",
                             "yt", "at")})
    print(f"\nSaved results to {out_dir / 'cem_results.json'}")


if __name__ == "__main__":
    main()
