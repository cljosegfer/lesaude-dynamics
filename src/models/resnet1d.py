import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    blocks = [BasicBlock1d(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_blocks):
        blocks.append(BasicBlock1d(out_ch, out_ch))
    return nn.Sequential(*blocks)


class ResNet1d(nn.Module):
    """ResNet-18 equivalent for 1D time-series.

    Input:  (B, in_channels, length)   e.g. (B, 12, 5000)
    Output: (B, embedding_dim)

    The final linear layer is named `fc` so spt.backbone.utils.set_embedding_dim
    can replace it if needed.
    """

    def __init__(
        self,
        in_channels: int = 12,
        embedding_dim: int = 512,
        blocks: tuple[int, ...] = (2, 2, 2, 2),
        base_channels: int = 64,
    ):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(c,     c,     blocks[0], stride=1)
        self.layer2 = _make_layer(c,     c * 2, blocks[1], stride=2)
        self.layer3 = _make_layer(c * 2, c * 4, blocks[2], stride=2)
        self.layer4 = _make_layer(c * 4, c * 8, blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
