from __future__ import annotations

"""
VGG-19 encoder wrapper with explicit provenance.

This module is the only intentionally vendored component in the clean-room package.
Its layer layout follows the normalized VGG encoder used by the CAP-VSTNet reference
implementation:

  Reproduce_CAP-VSTNet/CAP-VSTNet/models/VGG.py

Reference paper:
  Wen, Gao, Zou. "CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer."
  CVPR 2023.
"""

from pathlib import Path
from typing import Self

import torch
from torch import nn


def _build_normalized_vgg19() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),
    )


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload  # type: ignore[return-value]
    raise TypeError("Unsupported VGG checkpoint payload type.")


class VGG19Encoder(nn.Module):
    def __init__(self, checkpoint_path: str | Path, *, freeze: bool = True) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "VGG checkpoint not found. Expected normalized VGG-19 weights at "
                f"{checkpoint_path}."
            )

        base = _build_normalized_vgg19()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        base.load_state_dict(_extract_state_dict(checkpoint))
        layers = list(base.children())
        self.enc_1 = nn.Sequential(*layers[:4])
        self.enc_2 = nn.Sequential(*layers[4:11])
        self.enc_3 = nn.Sequential(*layers[11:18])
        self.enc_4 = nn.Sequential(*layers[18:31])
        self.enc_5 = nn.Sequential(*layers[31:45])

        if freeze:
            for name in ("enc_1", "enc_2", "enc_3", "enc_4", "enc_5"):
                for parameter in getattr(self, name).parameters():
                    parameter.requires_grad = False

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, freeze: bool = True) -> Self:
        return cls(checkpoint_path, freeze=freeze)

    def encode_with_intermediate(self, tensor: torch.Tensor, n_layer: int = 4) -> list[torch.Tensor]:
        if n_layer < 1 or n_layer > 5:
            raise ValueError(f"n_layer must be between 1 and 5, got {n_layer}.")
        features: list[torch.Tensor] = []
        current = tensor
        for layer_index in range(n_layer):
            current = getattr(self, f"enc_{layer_index + 1}")(current)
            features.append(current)
        return features

    def encode(self, tensor: torch.Tensor, n_layer: int = 4) -> torch.Tensor:
        return self.encode_with_intermediate(tensor, n_layer=n_layer)[-1]

    def forward(self, tensor: torch.Tensor, n_layer: int = 4) -> list[torch.Tensor]:
        return self.encode_with_intermediate(tensor, n_layer=n_layer)
