import torch.nn as nn

from models.resnet1d import ResNet1d
from models.vit1d import ViT1d


def build_backbone(cfg) -> nn.Module:
    """Encoder named by cfg["backbone"]: "resnet1d" (default, also for configs and checkpoint hparams that
    predate the key) or "vit1d", configured by cfg["vit"]. Takes a Hydra config or a checkpoint's
    hyper_parameters dict, so loaders can rebuild the trained architecture."""
    name = cfg.get("backbone", "resnet1d")
    if name == "resnet1d":
        return ResNet1d(in_channels=12, embedding_dim=cfg["embedding_dim"])
    if name == "vit1d":
        return ViT1d(in_channels=12, embedding_dim=cfg["embedding_dim"], **cfg["vit"])
    raise ValueError(f"Unknown backbone: {name!r} (expected resnet1d | vit1d)")
