"""
Forward Dynamics Model Diagnostic — notebook cells
====================================================
Loads the pretrained world model (Enc_theta + Proj_omega + Dyn_phi) from
archive/pretrain_val.ckpt, pulls a batch of longitudinal ECG pairs, and computes:

  ht      = Enc_theta(Xt)                       actual current-state embedding
  ht1     = Enc_theta(Xt1)                       actual next-state embedding (target)
  ht1_hat = Dyn_phi(ht, Proj_omega(at))          predicted next-state embedding

Paste each `# %%` cell into the notebook separately.
"""

# %% Imports and repo path setup
import sys
from pathlib import Path

REPO_ROOT = Path("/sonic_home/josefernandes/repo/lesaude-dynamics")
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionProjector, DynamicsPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# %% Config — dims must match configs/pretrain.yaml, paths must match configs/data.yaml
EMBEDDING_DIM = 256
ACTION_DIM = 76
PREDICTOR_HIDDEN_DIM = 512
CHECKPOINT = REPO_ROOT / "archive" / "pretrain_val.ckpt"

with open(REPO_ROOT / "configs" / "data.yaml") as f:
    data_cfg = yaml.safe_load(f)
LANCE_PATH = data_cfg["lance_path"]
PAIRS_PATH = data_cfg["pairs_path"]

print(f"checkpoint: {CHECKPOINT}")
print(f"lance_path: {LANCE_PATH}")
print(f"pairs_path: {PAIRS_PATH}")

# %% Load Enc_theta, Proj_omega, Dyn_phi from the checkpoint's state_dict
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]

encoder = ResNet1d(in_channels=12, embedding_dim=EMBEDDING_DIM)
encoder_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
assert not missing and not unexpected, f"encoder mismatch — missing: {missing}, unexpected: {unexpected}"

projector = ActionProjector(action_dim=ACTION_DIM, embed_dim=EMBEDDING_DIM)
proj_state = {k.removeprefix("projector.proj."): v for k, v in state.items() if k.startswith("projector.proj.")}
missing, unexpected = projector.load_state_dict(proj_state, strict=True)
assert not missing and not unexpected, f"projector mismatch — missing: {missing}, unexpected: {unexpected}"

dynamics = DynamicsPredictor(embed_dim=EMBEDDING_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM)
dyn_state = {k.removeprefix("projector.pred."): v for k, v in state.items() if k.startswith("projector.pred.")}
missing, unexpected = dynamics.load_state_dict(dyn_state, strict=True)
assert not missing and not unexpected, f"dynamics mismatch — missing: {missing}, unexpected: {unexpected}"

for module in (encoder, projector, dynamics):
    module.eval().to(device)
    for p in module.parameters():
        p.requires_grad_(False)

print("Loaded Enc_theta, Proj_omega, Dyn_phi.")

# %% Load a batch of longitudinal ECG pairs (Xt, Xt+1, yt, at)
SPLIT = "val"
BATCH_SIZE = 256
SEED = 0

pair_ds = MIMICLanceDataset(LANCE_PATH, split=SPLIT, mode="pair", pairs_path=PAIRS_PATH)

rng = np.random.default_rng(SEED)
idx = rng.choice(len(pair_ds), size=min(BATCH_SIZE, len(pair_ds)), replace=False).tolist()
batch = pair_ds.__getitems__(idx)

Xt = torch.stack([b["xt"] for b in batch]).to(device)    # (B, 12, T)
Xt1 = torch.stack([b["xt1"] for b in batch]).to(device)  # (B, 12, T)
yt = torch.stack([b["yt"] for b in batch]).to(device)    # (B, 76)
at = torch.stack([b["at"] for b in batch]).to(device)    # (B, 76), in {-1,0,1}

print(f"batch: Xt {tuple(Xt.shape)}, Xt1 {tuple(Xt1.shape)}, yt {tuple(yt.shape)}, at {tuple(at.shape)}")
print(f"active transitions in batch: {(at.abs().sum(dim=1) > 0).sum().item()} / {len(idx)}")

# %% Compute embeddings and the dynamics prediction, using the true action, zero action, and a
# magnitude-matched noise action: same per-pair ||a_t||_0 as the true action, but flipped on random
# label positions with random +-1 signs ("right sparsity, wrong labels"). Vectorized by giving every
# label a random priority score per row, then keeping the top-||a_t||_0 lowest-priority labels as
# flips — so each row's noise action has exactly as many flips as that row's true action.
noise_match_gen = torch.Generator(device=device).manual_seed(SEED)
action_magnitude = at.abs().sum(dim=1, keepdim=True)  # (B, 1) — per-pair ||a_t||_0
priority = torch.rand(at.shape, generator=noise_match_gen, device=device)
flip_rank = priority.argsort(dim=1).argsort(dim=1)     # 0-indexed rank of each label within its row
flip_mask = flip_rank < action_magnitude                # top-||a_t||_0 random labels per row
signs = torch.randint(0, 2, at.shape, generator=noise_match_gen, device=device).float() * 2 - 1  # +-1
at_noise = flip_mask.float() * signs

with torch.no_grad():
    ht = encoder(Xt)                                    # (B, 256)  actual current-state embedding
    ht1 = encoder(Xt1)                                   # (B, 256)  actual next-state embedding (target)

    ht1_hat = dynamics(ht, projector(at.float()))          # (B, 256)  predicted with the TRUE action
    ht1_zero = dynamics(ht, projector(torch.zeros_like(at, dtype=torch.float32)))    # predicted with the ZERO action
    ht1_noise = dynamics(ht, projector(at_noise.float()))                            # predicted with a MAGNITUDE-MATCHED NOISE action

# print(f"ht         {tuple(ht.shape)}")
# print(f"ht1        {tuple(ht1.shape)}")
# print(f"ht1_hat    {tuple(ht1_hat.shape)}")
# print(f"ht1_zero   {tuple(ht1_zero.shape)}")
# print(f"ht1_noise  {tuple(ht1_noise.shape)}")

# %% MSE against ht1, broken down by action magnitude ||a_t||_0 (how many of the 76 label columns
# flipped), for:
#   ht         — "do nothing" baseline in embedding space (no dynamics model involved)
#   ht1_hat    — Dyn_phi(ht, Proj(at))                      true action
#   ht1_zero   — Dyn_phi(ht, Proj(zero action))              zero-action baseline
#   ht1_noise  — Dyn_phi(ht, Proj(magnitude-matched noise))  same ||a_t||_0 as truth, wrong labels
#
# Rows: "all" (no grouping), "different" (all active pairs, ||a_t||_0 > 0, combined), then one row
# per exact magnitude 0, 1, 2, ... (magnitudes with too few pairs are lumped into a trailing ">=k"
# bucket so their mean/var isn't just sampling noise). Note magnitude "0" is exactly the stable pairs.
MIN_GROUP_SIZE = 5
COLS = ["ht", "ht1_hat", "ht1_zero", "ht1_noise"]

action_l0 = at.abs().sum(dim=1).cpu().numpy()   # ||a_t||_0 — number of labels that flipped
mse = {
    "ht":         (ht - ht1).pow(2).mean(dim=1).cpu().numpy(),
    "ht1_hat":    (ht1_hat - ht1).pow(2).mean(dim=1).cpu().numpy(),
    "ht1_zero":   (ht1_zero - ht1).pow(2).mean(dim=1).cpu().numpy(),
    "ht1_noise":  (ht1_noise - ht1).pow(2).mean(dim=1).cpu().numpy(),
}
mse_df = pd.DataFrame({"action_l0": action_l0, **mse})

l0_counts = mse_df["action_l0"].value_counts().sort_index()
cum_from_top = l0_counts[::-1].cumsum()[::-1]  # pairs with action_l0 >= i, for each i
cutoff = next((i for i in l0_counts.index if cum_from_top[i] < MIN_GROUP_SIZE), l0_counts.index.max() + 1)

mse_df["magnitude_bucket"] = np.where(
    mse_df["action_l0"] < cutoff,
    mse_df["action_l0"].astype(str),
    f">={cutoff}",
)

order = sorted(mse_df["magnitude_bucket"].unique(), key=lambda s: cutoff if s.startswith(">=") else int(s))

def mse_table(row_frames: dict, cols: list) -> pd.DataFrame:
    """One row per (label, sub_df) with mean/std per col, plus a single trailing 'count' column."""
    table = pd.DataFrame({label: sub[cols].agg(["mean", "std"]).T.stack() for label, sub in row_frames.items()}).T
    table[("count", "")] = pd.Series({label: len(sub) for label, sub in row_frames.items()})
    return table

def plot_mse_histograms(row_frames: dict, cols: list, title: str, bins: int = 30, xlabel: str = "MSE"):
    """Grid of histograms — one row per discrimination group (same rows as the printed table),
    one column per variable — to check whether each distribution looks unimodal or like a
    mixture of populations, and whether that changes across groups. Shares one x/y scale across
    the whole grid (not just within a column) so magnitudes are directly comparable at a glance."""
    n_rows, n_cols = len(row_frames), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.2 * n_rows),
                              sharex=True, sharey=True, squeeze=False)
    for row_axes, (label, sub) in zip(axes, row_frames.items()):
        for ax, col in zip(row_axes, cols):
            ax.hist(sub[col], bins=bins, color="steelblue", edgecolor="white")
            ax.axvline(sub[col].mean(), color="crimson", lw=1.2)
        row_axes[0].set_ylabel(label, rotation=0, ha="right", va="center")
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

row_frames = {"all": mse_df, "different": mse_df[mse_df["action_l0"] > 0]}
row_frames |= {label: mse_df[mse_df["magnitude_bucket"] == label] for label in order}

by_magnitude = mse_table(row_frames, COLS)
print(by_magnitude.to_string())
by_magnitude.to_csv(REPO_ROOT / "demo" / "dynamics_mse_by_magnitude.csv")

# Plot only {all, different, 0} — the finer per-magnitude groups have too few pairs to visualize.
PLOT_GROUPS = ["all", "different", "0"]
plot_mse_histograms({g: row_frames[g] for g in PLOT_GROUPS}, COLS, "MSE distribution per baseline, by group")

# %% Same breakdown as above (rows: all, different, 0, 1, 2, ..., >=k — grouped by the TRUE action's
# ||a_t||_0, same magnitude_bucket as the main table), but comparing {ht, ht1_hat} against out-of-
# distribution noise actions instead of {ht1_zero, ht1_noise}. Unlike ht1_noise (magnitude-matched to
# each pair's own true action), these synthesize random ternary actions with independent per-label
# +-1 flips at controlled target sparsity levels (mean ||a||_0 = 1, 2, 3, and 76/2 = "half the labels
# flip"), regardless of the pair's true action — testing whether Dyn_phi's prediction keeps degrading
# as an unrelated perturbation grows, rather than just being insensitive to any "wrong" action (which
# ht1_noise alone can't show, since most real actions — and thus most magnitude-matched draws — are
# near-zero).
NOISE_L0_MEANS = [1, 2, 3, ACTION_DIM // 2]
NOISE_COLS = ["ht", "ht1_hat"] + [f"noise{m}" for m in NOISE_L0_MEANS]
noise_gen = torch.Generator(device=device).manual_seed(SEED)

noise_mse_df = mse_df[["action_l0", "magnitude_bucket", "ht", "ht1_hat"]].copy()
actual_l0 = {}
with torch.no_grad():
    for target_l0 in NOISE_L0_MEANS:
        p_flip = target_l0 / ACTION_DIM
        flip_mask = torch.rand(at.shape, generator=noise_gen, device=device) < p_flip
        signs = torch.randint(0, 2, at.shape, generator=noise_gen, device=device).float() * 2 - 1  # +-1
        at_noise_level = flip_mask.float() * signs

        ht1_noise_level = dynamics(ht, projector(at_noise_level.float()))
        noise_mse_df[f"noise{target_l0}"] = (ht1_noise_level - ht1).pow(2).mean(dim=1).cpu().numpy()
        actual_l0[f"noise{target_l0}"] = at_noise_level.abs().sum(dim=1).mean().item()

print("realized mean ||a||_0 per noise level (target vs. actual):")
for target_l0 in NOISE_L0_MEANS:
    print(f"  noise{target_l0}: target={target_l0}  actual={actual_l0[f'noise{target_l0}']:.2f}")
print()

noise_row_frames = {"all": noise_mse_df, "different": noise_mse_df[noise_mse_df["action_l0"] > 0]}
noise_row_frames |= {label: noise_mse_df[noise_mse_df["magnitude_bucket"] == label] for label in order}

by_magnitude_noise = mse_table(noise_row_frames, NOISE_COLS)
print(by_magnitude_noise.to_string())
by_magnitude_noise.to_csv(REPO_ROOT / "demo" / "dynamics_mse_by_magnitude_noise.csv")

plot_mse_histograms({g: noise_row_frames[g] for g in PLOT_GROUPS}, NOISE_COLS, "MSE distribution per noise level, by group")

# %% Same experiment as the first table (COLS = ht, ht1_hat, ht1_zero, ht1_noise), but scoring
# cosine similarity to ht1 instead of MSE (bounded in [-1, 1], higher = more aligned). MSE is
# sensitive to overall vector magnitude; cosine isolates whether the predicted *direction* in
# embedding space is right, regardless of scale.
cos = {
    "ht":         F.cosine_similarity(ht, ht1, dim=1).cpu().numpy(),
    "ht1_hat":    F.cosine_similarity(ht1_hat, ht1, dim=1).cpu().numpy(),
    "ht1_zero":   F.cosine_similarity(ht1_zero, ht1, dim=1).cpu().numpy(),
    "ht1_noise":  F.cosine_similarity(ht1_noise, ht1, dim=1).cpu().numpy(),
}
cos_df = pd.DataFrame({"action_l0": action_l0, "magnitude_bucket": mse_df["magnitude_bucket"].values, **cos})

cos_row_frames = {"all": cos_df, "different": cos_df[cos_df["action_l0"] > 0]}
cos_row_frames |= {label: cos_df[cos_df["magnitude_bucket"] == label] for label in order}

by_magnitude_cos = mse_table(cos_row_frames, COLS)
print(by_magnitude_cos.to_string())
by_magnitude_cos.to_csv(REPO_ROOT / "demo" / "dynamics_cos_by_magnitude.csv")

plot_mse_histograms({g: cos_row_frames[g] for g in PLOT_GROUPS}, COLS,
                     "Cosine similarity to ht1 per baseline, by group", xlabel="cosine similarity")

# %% Same experiment as the noise-level table, but scoring cosine similarity to ht1 instead of MSE.
# Reseeds the noise generator identically to the MSE noise cell above, so this scores the exact same
# noise actions — an apples-to-apples comparison, just under a different metric.
cos_noise_df = pd.DataFrame({
    "action_l0": action_l0,
    "magnitude_bucket": mse_df["magnitude_bucket"].values,
    "ht": cos["ht"],
    "ht1_hat": cos["ht1_hat"],
})

noise_gen_cos = torch.Generator(device=device).manual_seed(SEED)
with torch.no_grad():
    for target_l0 in NOISE_L0_MEANS:
        p_flip = target_l0 / ACTION_DIM
        flip_mask = torch.rand(at.shape, generator=noise_gen_cos, device=device) < p_flip
        signs = torch.randint(0, 2, at.shape, generator=noise_gen_cos, device=device).float() * 2 - 1  # +-1
        at_noise_level = flip_mask.float() * signs

        ht1_noise_level = dynamics(ht, projector(at_noise_level.float()))
        cos_noise_df[f"noise{target_l0}"] = F.cosine_similarity(ht1_noise_level, ht1, dim=1).cpu().numpy()

cos_noise_row_frames = {"all": cos_noise_df, "different": cos_noise_df[cos_noise_df["action_l0"] > 0]}
cos_noise_row_frames |= {label: cos_noise_df[cos_noise_df["magnitude_bucket"] == label] for label in order}

by_magnitude_noise_cos = mse_table(cos_noise_row_frames, NOISE_COLS)
print(by_magnitude_noise_cos.to_string())
by_magnitude_noise_cos.to_csv(REPO_ROOT / "demo" / "dynamics_cos_by_magnitude_noise.csv")

plot_mse_histograms({g: cos_noise_row_frames[g] for g in PLOT_GROUPS}, NOISE_COLS,
                     "Cosine similarity to ht1 per noise level, by group", xlabel="cosine similarity")
