from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import CAPColorTransferConfig
from .reversible import CAPReversibleBackbone
from .transform import CholeskyWCT


@dataclass(frozen=True, slots=True)
class CAPColorTransferOutput:
    stylized_rgb: torch.Tensor
    content_latent: torch.Tensor
    style_latent: torch.Tensor
    stylized_latent: torch.Tensor


class CAPColorTransferModel(nn.Module):
    def __init__(self, config: CAPColorTransferConfig | None = None) -> None:
        super().__init__()
        self.config = config or CAPColorTransferConfig.from_photo_v1()
        self.backbone = CAPReversibleBackbone(self.config)
        self.transfer_module = CholeskyWCT(
            epsilon=self.config.transfer_epsilon,
            max_attempts=self.config.transfer_max_attempts,
            jitter_growth=self.config.transfer_jitter_growth,
        )

    @property
    def downscale_factor(self) -> int:
        return self.backbone.downscale_factor

    @property
    def latent_channels(self) -> int:
        return self.backbone.latent_channels

    def forward(self, content_rgb: torch.Tensor, style_rgb: torch.Tensor) -> CAPColorTransferOutput:
        return self.stylize(content_rgb, style_rgb)

    def encode(self, content_rgb: torch.Tensor) -> torch.Tensor:
        self._validate_image_tensor(content_rgb, name="content_rgb")
        return self.backbone.encode(content_rgb)

    def transfer(self, content_latent: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent_tensor(content_latent, name="content_latent")
        self._validate_latent_tensor(style_latent, name="style_latent")
        if content_latent.shape[0] != style_latent.shape[0]:
            raise ValueError(
                "content_latent and style_latent must share the same batch size, "
                f"got {content_latent.shape[0]} and {style_latent.shape[0]}."
            )
        return self.transfer_module.transfer(content_latent, style_latent)

    def decode(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        self._validate_latent_tensor(latent_tensor, name="latent_tensor")
        return self.backbone.inverse(latent_tensor)

    def stylize(self, content_rgb: torch.Tensor, style_rgb: torch.Tensor) -> CAPColorTransferOutput:
        self._validate_image_tensor(content_rgb, name="content_rgb")
        self._validate_image_tensor(style_rgb, name="style_rgb")
        if content_rgb.shape[0] != style_rgb.shape[0]:
            raise ValueError(
                "content_rgb and style_rgb must share the same batch size, "
                f"got {content_rgb.shape[0]} and {style_rgb.shape[0]}."
            )

        content_latent = self.backbone.encode(content_rgb)
        style_latent = self.backbone.encode(style_rgb)
        stylized_latent = self.transfer_module.transfer(content_latent, style_latent)
        stylized_rgb = self.backbone.inverse(stylized_latent)
        return CAPColorTransferOutput(
            stylized_rgb=stylized_rgb,
            content_latent=content_latent,
            style_latent=style_latent,
            stylized_latent=stylized_latent,
        )

    def _validate_image_tensor(self, tensor: torch.Tensor, *, name: str) -> None:
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(tensor.shape)}.")
        if tensor.shape[1] != self.config.in_channels:
            raise ValueError(
                f"{name} must have {self.config.in_channels} channels, got {tensor.shape[1]}."
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor, got dtype {tensor.dtype}.")
        self._validate_spatial_divisibility(tensor.shape[-2], tensor.shape[-1], name=name)

    def _validate_latent_tensor(self, tensor: torch.Tensor, *, name: str) -> None:
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(tensor.shape)}.")
        if tensor.shape[1] != self.latent_channels:
            raise ValueError(
                f"{name} must have {self.latent_channels} channels, got {tensor.shape[1]}."
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor, got dtype {tensor.dtype}.")
        self._validate_spatial_divisibility(tensor.shape[-2], tensor.shape[-1], name=name)

    def _validate_spatial_divisibility(self, height: int, width: int, *, name: str) -> None:
        factor = self.downscale_factor
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"{name} height and width must be divisible by the backbone downscale factor "
                f"({factor}), got {(height, width)}."
            )
