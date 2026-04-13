from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import CAPColorTransferConfig
from .ops import (
    InjectiveChannelPadding,
    collect_spatial_to_channels,
    merge_channels,
    spread_channels_to_spatial,
    split_channels,
    squeeze2d,
    unsqueeze2d,
)


TensorPair = tuple[torch.Tensor, torch.Tensor]


def _zero_initialized_conv_out(layer: nn.Conv2d) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ResidualUpdateNet(nn.Module):
    def __init__(self, half_channels: int, *, mult: int, kernel_size: int) -> None:
        super().__init__()
        bottleneck_channels = max(half_channels // mult, 1)
        pad = kernel_size // 2
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(half_channels, bottleneck_channels, kernel_size=kernel_size, bias=True),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(bottleneck_channels, half_channels, kernel_size=kernel_size, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        last_layer = self.layers[-1]
        if isinstance(last_layer, nn.Conv2d):
            _zero_initialized_conv_out(last_layer)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class AdditiveCouplingBlock(nn.Module):
    def __init__(self, half_channels: int, *, mult: int, kernel_size: int) -> None:
        super().__init__()
        self.update = ResidualUpdateNet(half_channels, mult=mult, kernel_size=kernel_size)

    def forward(self, state: TensorPair) -> TensorPair:
        left, right = state
        updated_right = left + self.update(right)
        return right, updated_right

    def inverse(self, state: TensorPair) -> TensorPair:
        right, updated_right = state
        left = updated_right - self.update(right)
        return left, right


class ReversibleStage(nn.Module):
    def __init__(self, half_channels: int, depth: int, *, mult: int, kernel_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            AdditiveCouplingBlock(half_channels, mult=mult, kernel_size=kernel_size)
            for _ in range(depth)
        )

    def forward(self, state: TensorPair) -> TensorPair:
        for block in self.blocks:
            state = block(state)
        return state

    def inverse(self, state: TensorPair) -> TensorPair:
        for block in reversed(self.blocks):
            state = block.inverse(state)
        return state


class ChannelRefinement(nn.Module):
    def __init__(
        self,
        input_half_channels: int,
        target_half_channels: int,
        *,
        spread_steps: int,
        num_blocks: int,
        mult: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.spread_steps = spread_steps
        self.target_half_channels = target_half_channels
        self.internal_half_channels = target_half_channels * (4 ** spread_steps)
        if self.internal_half_channels < input_half_channels:
            raise ValueError(
                "Channel refinement must be injective: "
                f"internal_half_channels={self.internal_half_channels} "
                f"is smaller than input_half_channels={input_half_channels}."
            )
        self.padding = InjectiveChannelPadding(self.internal_half_channels - input_half_channels)
        self.blocks = nn.ModuleList(
            AdditiveCouplingBlock(
                self.internal_half_channels,
                mult=mult,
                kernel_size=kernel_size,
            )
            for _ in range(num_blocks)
        )

    @property
    def latent_channels(self) -> int:
        return self.target_half_channels * 2

    def forward(self, merged_tensor: torch.Tensor) -> torch.Tensor:
        left, right = split_channels(merged_tensor)
        state = (self.padding(left), self.padding(right))
        for block in self.blocks:
            state = block(state)
        merged = merge_channels(*state)
        return spread_channels_to_spatial(merged, self.spread_steps)

    def inverse(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        merged = collect_spatial_to_channels(latent_tensor, self.spread_steps)
        state = split_channels(merged)
        for block in reversed(self.blocks):
            state = block.inverse(state)
        left, right = state
        return merge_channels(self.padding.inverse(left), self.padding.inverse(right))


@dataclass(frozen=True, slots=True)
class BackboneSignature:
    input_channels: int
    latent_channels: int
    downscale_factor: int


class CAPReversibleBackbone(nn.Module):
    def __init__(self, config: CAPColorTransferConfig) -> None:
        super().__init__()
        self.config = config

        merged_input_channels = config.n_channels[0] * 2
        if merged_input_channels < config.in_channels:
            raise ValueError(
                "Injective input lift requires at least as many target channels as source channels, "
                f"got target={merged_input_channels} and source={config.in_channels}."
            )
        self.input_padding = InjectiveChannelPadding(merged_input_channels - config.in_channels)

        self.stage_strides = config.n_strides
        self.stages = nn.ModuleList(
            ReversibleStage(
                half_channels=half_channels,
                depth=depth,
                mult=config.mult,
                kernel_size=config.kernel_size,
            )
            for half_channels, depth in zip(config.n_channels, config.n_blocks)
        )
        self.refinement = ChannelRefinement(
            input_half_channels=config.n_channels[-1],
            target_half_channels=config.hidden_dim,
            spread_steps=config.sp_steps,
            num_blocks=config.refinement_blocks,
            mult=config.mult,
            kernel_size=config.kernel_size,
        )
        self.signature = BackboneSignature(
            input_channels=config.in_channels,
            latent_channels=self.refinement.latent_channels,
            downscale_factor=config.downscale_factor,
        )

    @property
    def latent_channels(self) -> int:
        return self.signature.latent_channels

    @property
    def downscale_factor(self) -> int:
        return self.signature.downscale_factor

    def forward(self, tensor: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        return self.inverse(tensor) if inverse else self.encode(tensor)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        lifted = self.input_padding(tensor)
        state = split_channels(lifted)
        for stage_index, stage in enumerate(self.stages):
            if stage_index > 0 and self.stage_strides[stage_index] == 2:
                state = (squeeze2d(state[0]), squeeze2d(state[1]))
            state = stage(state)
        merged = merge_channels(*state)
        return self.refinement(merged)

    def inverse(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        merged = self.refinement.inverse(latent_tensor)
        state = split_channels(merged)
        for stage_index in reversed(range(len(self.stages))):
            state = self.stages[stage_index].inverse(state)
            if stage_index > 0 and self.stage_strides[stage_index] == 2:
                state = (unsqueeze2d(state[0]), unsqueeze2d(state[1]))
        merged = merge_channels(*state)
        return self.input_padding.inverse(merged)
