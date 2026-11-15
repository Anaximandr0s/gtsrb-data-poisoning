"""CNN for 32x32 RGB traffic signs, with normalisation as its first layer."""

import numpy as np
import torch
from torch import nn


def compute_norm_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of images (N, C, H, W); call it on the training split only."""
    return x.mean(axis=(0, 2, 3)).tolist(), x.std(axis=(0, 2, 3)).tolist()


class Normalize(nn.Module):
    """Standardise [0, 1] images per channel; mean/std are buffers so they travel with the checkpoint."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        """Store ``mean`` and ``std`` (one value per channel) as (1, C, 1, 1) buffers."""
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(x - mean) / std``."""
        return (x - self.mean) / self.std


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    """conv-BN-ReLU-conv-BN-ReLU-MaxPool-Dropout; halves the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class GtsrbCNN(nn.Module):
    """Normalize -> conv blocks -> fully connected feature layer -> classifier."""

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        channels: list[int],
        fc_dim: int,
        num_classes: int,
        image_size: int,
        conv_dropout: float,
        fc_dropout: float,
    ) -> None:
        """Build the network; each conv block halves the image size."""
        super().__init__()
        self.normalize = Normalize(mean, std)
        in_channels = [len(mean)] + channels[:-1]
        self.conv = nn.Sequential(
            *(_conv_block(i, o, conv_dropout) for i, o in zip(in_channels, channels))
        )
        flat_dim = channels[-1] * (image_size // 2 ** len(channels)) ** 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
        )
        self.classifier = nn.Linear(fc_dim, num_classes)

    @property
    def last_conv(self) -> nn.Conv2d:
        """The last convolution of the network, used as the Grad-CAM target layer."""
        return self.conv[-1][3]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Activations of the last hidden layer (after ReLU), shape (N, fc_dim)."""
        return self.fc(self.conv(self.normalize(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits, shape (N, num_classes), for images in [0, 1]."""
        return self.classifier(self.features(x))


def build_model(cfg: dict, mean: list[float], std: list[float]) -> GtsrbCNN:
    """Create the CNN from the ``model`` and ``data`` config sections and the train-split norm stats."""
    return GtsrbCNN(
        mean=mean,
        std=std,
        channels=cfg["model"]["channels"],
        fc_dim=cfg["model"]["fc_dim"],
        num_classes=cfg["data"]["num_classes"],
        image_size=cfg["data"]["image_size"],
        conv_dropout=cfg["model"]["conv_dropout"],
        fc_dropout=cfg["model"]["fc_dropout"],
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
