from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class IntermediateEncoder(Protocol):
    def encode_with_intermediate(self, tensor: torch.Tensor, n_layer: int = 4) -> list[torch.Tensor]:
        ...


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    if feat.ndim != 4:
        raise ValueError(f"feat must have shape (B, C, H, W), got {tuple(feat.shape)}.")
    variance = feat.flatten(start_dim=2).var(dim=2, unbiased=False) + eps
    std = variance.sqrt().view(feat.shape[0], feat.shape[1], 1, 1)
    mean = feat.flatten(start_dim=2).mean(dim=2).view(feat.shape[0], feat.shape[1], 1, 1)
    return mean, std


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError(f"feat must have shape (B, C, H, W), got {tuple(feat.shape)}.")
    batch, channels, height, width = feat.shape
    reshaped = feat.reshape(batch, channels, height * width)
    return torch.bmm(reshaped, reshaped.transpose(1, 2)) / float(channels * height * width)


class VGGStyleLoss(nn.Module):
    def __init__(self, encoder: IntermediateEncoder, *, n_layer: int = 4) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_layer = n_layer
        self.mse = nn.MSELoss()

    def forward(self, stylized_rgb: torch.Tensor, style_rgb: torch.Tensor) -> torch.Tensor:
        stylized_features = self.encoder.encode_with_intermediate(stylized_rgb, n_layer=self.n_layer)
        style_features = self.encoder.encode_with_intermediate(style_rgb, n_layer=self.n_layer)
        total = stylized_rgb.new_zeros(())
        for stylized_feature, style_feature in zip(stylized_features, style_features):
            stylized_mean, stylized_std = calc_mean_std(stylized_feature)
            style_mean, style_std = calc_mean_std(style_feature)
            total = total + self.mse(stylized_mean, style_mean) + self.mse(stylized_std, style_std)
        return total


class VGGGramLoss(nn.Module):
    def __init__(self, encoder: IntermediateEncoder, *, n_layer: int = 4) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_layer = n_layer
        self.mse = nn.MSELoss()

    def forward(self, stylized_rgb: torch.Tensor, style_rgb: torch.Tensor) -> torch.Tensor:
        stylized_features = self.encoder.encode_with_intermediate(stylized_rgb, n_layer=self.n_layer)
        style_features = self.encoder.encode_with_intermediate(style_rgb, n_layer=self.n_layer)
        total = stylized_rgb.new_zeros(())
        for stylized_feature, style_feature in zip(stylized_features, style_features):
            total = total + self.mse(gram_matrix(stylized_feature), gram_matrix(style_feature))
        return total
