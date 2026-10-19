"""
Frozen-Encoder Adapters for the anatomie/ Evaluation Scripts
=============================================================
anatomie/layer_readout.py, anatomie/refit_inverse_head.py and anatomie/signal_decoder.py reach encoders only
through load_encoder(), which returns a callable mapping a batch of raw ECGs (B, 12, 5000) to per-layer features
{layer: (B, D)}. A new backbone (e.g. a LeNEPA ViT) needs one new branch in load_encoder() and nothing else.
The heads trained with an encoder come from load_inverse_head() and load_dynamics().

Encoder specs
-------------
  random              ResNet1d with seed-0 random weights: the "pretraining adds nothing" floor
  random-vit          the same for ViT1d at its default size (src/models/vit1d.py)
  path/to/model.ckpt  Lightning checkpoint with encoder weights under "backbone." (relative to the repo root);
                      ResNet1d unless its hyper_parameters say backbone: vit1d
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dataset.dataset import MIMICLanceDataset  # noqa: E402
from models.backbones import build_backbone  # noqa: E402
from models.dynamics import ActionPredictor, ActionProjector, DynamicsPredictor  # noqa: E402
from models.resnet1d import ResNet1d  # noqa: E402
from models.vit1d import ViT1d  # noqa: E402


def data_paths(config: str = "configs/data.yaml") -> dict:
    with open(REPO / config) as f:
        return yaml.safe_load(f)


class ResNet1dLayers:
    """Global-average-pooled stem and layer1..layer4 outputs, plus h = the fc output used downstream."""

    layers = ("stem", "layer1", "layer2", "layer3", "layer4", "h")

    def __init__(self, net: ResNet1d):
        self.net = net.eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        net, out = self.net, {}
        with torch.autocast("cuda", dtype=torch.float16):   # pretraining runs in 16-mixed
            z = net.stem(x)
            out["stem"] = z.float().mean(-1)
            for name in ("layer1", "layer2", "layer3", "layer4"):
                z = getattr(net, name)(z)
                out[name] = z.float().mean(-1)
            out["h"] = net.fc(out["layer4"]).float()
        return out


class ViT1dLayers:
    """Mean-pooled tokens after the patch embedding and after every block, plus h (ViT1d.layer_features)."""

    def __init__(self, net: ViT1d):
        self.net = net.eval()
        self.layers = ("patch", *(f"block{i + 1}" for i in range(len(net.blocks))), "h")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        with torch.autocast("cuda", dtype=torch.float16):   # pretraining runs in 16-mixed
            return {k: v.float() for k, v in self.net.layer_features(x).items()}


def _state_dict(spec: str) -> dict:
    return torch.load(REPO / spec, map_location="cpu", weights_only=False)["state_dict"]


def load_encoder(spec: str, device: str = "cuda") -> ResNet1dLayers | ViT1dLayers:
    """'random' (ResNet1d), 'random-vit' (ViT1d at its default size) or a checkpoint path; a checkpoint whose
    hparams say backbone: vit1d is rebuilt with models.backbones.build_backbone."""
    if spec == "random":
        torch.manual_seed(0)
        return ResNet1dLayers(ResNet1d(in_channels=12, embedding_dim=256).to(device))
    if spec == "random-vit":
        torch.manual_seed(0)
        return ViT1dLayers(ViT1d(in_channels=12, embedding_dim=256).to(device))
    ck = torch.load(REPO / spec, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("backbone."): v for k, v in ck["state_dict"].items() if k.startswith("backbone.")}
    hp = ck.get("hyper_parameters") or {}
    if hp.get("backbone") == "vit1d":
        net = build_backbone(hp)
        net.load_state_dict(sd)
        return ViT1dLayers(net.to(device))
    net = ResNet1d(in_channels=12, embedding_dim=256)
    net.load_state_dict(sd)
    return ResNet1dLayers(net.to(device))


def load_inverse_head(spec: str, device: str = "cuda") -> ActionPredictor | None:
    """The inverse head trained with the encoder (reads h), if the checkpoint has one: projector.pred in
    scripts/inverse_pretrain.py, projector.inv in scripts/joint_pretrain.py."""
    if spec in ("random", "random-vit"):
        return None
    sd = _state_dict(spec)
    for prefix in ("projector.inv.", "projector.pred."):
        if f"{prefix}onset_head.weight" in sd:
            w = sd[f"{prefix}input_proj.weight"]
            head = ActionPredictor(embed_dim=w.shape[1], hidden_dim=w.shape[0],
                                   action_dim=sd[f"{prefix}onset_head.weight"].shape[0])
            head.load_state_dict({k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)})
            return head.to(device).eval()
    return None


def load_dynamics(spec: str, device: str = "cuda"):
    """The forward model trained with the encoder, as a function (h_t, a_t) -> predicted h_{t+1} (raw h, float
    a_t), if the checkpoint has one. Layouts, as scripts/cem_planning.py loads them:
      projector.proj + projector.pred  Dyn(h, Proj(a)) of scripts/pretrain.py and scripts/joint_pretrain.py; the
                                       older pretrain_displacement checkpoints (training ckpt_path names
                                       "displacement") predict h_{t+1} - h_t with the same layout
      projector.action                 the displacement g(a) alone, scripts/pretrain_displacement.py
    scripts/inverse_pretrain.py keeps its inverse head under projector.pred, with no projector.proj."""
    if spec in ("random", "random-vit"):
        return None
    ckpt = torch.load(REPO / spec, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sub = lambda prefix: {k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)}
    if g := sub("projector.action."):
        w = g.get("weight", g.get("net.0.weight"))
        net = (nn.Linear(w.shape[1], w.shape[0], bias=False) if set(g) == {"weight"}
               else ActionProjector(action_dim=w.shape[1], embed_dim=w.shape[0], bias=False))
        net.load_state_dict(g)
        net.to(device).eval()
        return lambda h, a: h + net(a)
    if not (p := sub("projector.proj.")):
        return None
    proj = ActionProjector(action_dim=p["net.0.weight"].shape[1], embed_dim=p["net.0.weight"].shape[0])
    proj.load_state_dict(p)
    d = sub("projector.pred.")
    pred = DynamicsPredictor(embed_dim=d["output_proj.weight"].shape[0], hidden_dim=d["output_proj.weight"].shape[1])
    pred.load_state_dict(d)
    proj.to(device).eval()
    pred.to(device).eval()
    if "displacement" in str((ckpt.get("hyper_parameters") or {}).get("ckpt_path", "")):
        return lambda h, a: h + pred(h, proj(a))
    return lambda h, a: pred(h, proj(a))


@torch.no_grad()
def embed(encoders: dict, rows, lance_path: str, layers=None, batch_size: int = 256, num_workers: int = 16):
    """Features of the ECGs at Lance row indices `rows` for every encoder, in one pass over the data.

    Returns ({name: {layer: (N, D) float32 CPU tensor}}, labels (N, 76)), in the order of `rows`.
    Call from under `if __name__ == "__main__":` — the loader uses spawned workers (Lance is not fork-safe).
    """
    ds = MIMICLanceDataset(lance_path, split="train", mode="monitoring")
    ds.rows = np.asarray(rows)          # any rows, any split: monitoring mode reads self.rows directly
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        multiprocessing_context="spawn", pin_memory=True)
    feats = {n: {} for n in encoders}
    labels = []
    for batch in loader:
        x = batch["waveform"].cuda(non_blocking=True)
        labels.append(batch["label"])
        for n, enc in encoders.items():
            for layer, v in enc(x).items():
                if layers is None or layer in layers:
                    feats[n].setdefault(layer, []).append(v.cpu())
    return {n: {l: torch.cat(v) for l, v in d.items()} for n, d in feats.items()}, torch.cat(labels)
