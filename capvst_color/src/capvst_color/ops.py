from __future__ import annotations

import torch
from torch import nn


def split_channels(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(tensor.shape)}.")
    channels = tensor.shape[1]
    if channels % 2 != 0:
        raise ValueError(f"Channel dimension must be even for splitting, got {channels}.")
    midpoint = channels // 2
    return tensor[:, :midpoint], tensor[:, midpoint:]


def merge_channels(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 4 or right.ndim != 4:
        raise ValueError(
            "merge_channels expects two 4D tensors, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.shape[0] != right.shape[0] or left.shape[2:] != right.shape[2:]:
        raise ValueError(
            "Left and right tensors must share batch and spatial dimensions, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    return torch.cat((left, right), dim=1)


class InjectiveChannelPadding(nn.Module):
    def __init__(self, pad_channels: int) -> None:
        super().__init__()
        if pad_channels < 0:
            raise ValueError(f"pad_channels must be non-negative, got {pad_channels}.")
        self.pad_channels = pad_channels

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pad_channels == 0:
            return tensor
        padding = tensor.new_zeros(
            tensor.shape[0],
            self.pad_channels,
            tensor.shape[2],
            tensor.shape[3],
        )
        return torch.cat((tensor, padding), dim=1)

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pad_channels == 0:
            return tensor
        if tensor.shape[1] < self.pad_channels:
            raise ValueError(
                "Cannot remove injective padding because the tensor has fewer channels "
                f"({tensor.shape[1]}) than the configured padding ({self.pad_channels})."
            )
        return tensor[:, : tensor.shape[1] - self.pad_channels]


def squeeze2d(tensor: torch.Tensor, factor: int = 2) -> torch.Tensor:
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}.")
    batch, channels, height, width = tensor.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Spatial dimensions {(height, width)} must be divisible by squeeze factor {factor}."
        )
    new_height = height // factor
    new_width = width // factor
    tensor = tensor.reshape(batch, channels, new_height, factor, new_width, factor)
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    return tensor.reshape(batch, channels * factor * factor, new_height, new_width)


def unsqueeze2d(tensor: torch.Tensor, factor: int = 2) -> torch.Tensor:
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}.")
    batch, channels, height, width = tensor.shape
    divisor = factor * factor
    if channels % divisor != 0:
        raise ValueError(
            f"Channel dimension {channels} must be divisible by {divisor} for unsqueeze."
        )
    new_channels = channels // divisor
    tensor = tensor.reshape(batch, new_channels, factor, factor, height, width)
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    return tensor.reshape(batch, new_channels, height * factor, width * factor)


def collect_spatial_to_channels(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        tensor = squeeze2d(tensor, factor=2)
    return tensor


def spread_channels_to_spatial(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        tensor = unsqueeze2d(tensor, factor=2)
    return tensor
