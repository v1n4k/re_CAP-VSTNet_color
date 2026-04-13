from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CAPColorTransferConfig:
    n_blocks: tuple[int, int, int] = (10, 10, 10)
    n_strides: tuple[int, int, int] = (1, 2, 2)
    n_channels: tuple[int, int, int] = (16, 64, 256)
    in_channels: int = 3
    hidden_dim: int = 16
    sp_steps: int = 2
    mult: int = 4
    kernel_size: int = 3
    refinement_blocks: int = 2
    transfer_epsilon: float = 2e-5
    transfer_max_attempts: int = 8
    transfer_jitter_growth: float = 10.0
    vgg_checkpoint_path: Path = Path("checkpoints/vgg_normalised.pth")

    def __post_init__(self) -> None:
        if len(self.n_blocks) != len(self.n_strides) or len(self.n_blocks) != len(self.n_channels):
            raise ValueError("n_blocks, n_strides, and n_channels must have the same length.")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        if self.kernel_size % 2 == 0 or self.kernel_size < 1:
            raise ValueError(f"kernel_size must be a positive odd integer, got {self.kernel_size}.")
        if self.sp_steps < 0:
            raise ValueError(f"sp_steps must be non-negative, got {self.sp_steps}.")
        if self.mult <= 0:
            raise ValueError(f"mult must be positive, got {self.mult}.")
        if self.refinement_blocks <= 0:
            raise ValueError(f"refinement_blocks must be positive, got {self.refinement_blocks}.")
        if self.transfer_epsilon <= 0:
            raise ValueError(f"transfer_epsilon must be positive, got {self.transfer_epsilon}.")
        if self.transfer_max_attempts <= 0:
            raise ValueError(
                f"transfer_max_attempts must be positive, got {self.transfer_max_attempts}."
            )
        if self.transfer_jitter_growth <= 1.0:
            raise ValueError(
                "transfer_jitter_growth must be greater than 1.0 "
                f"for bounded jitter escalation, got {self.transfer_jitter_growth}."
            )

        current = self.n_channels[0]
        for index, (stride, next_channels) in enumerate(zip(self.n_strides[1:], self.n_channels[1:]), start=1):
            if stride == 2 and next_channels != current * 4:
                raise ValueError(
                    "Each stride-2 stage must quadruple the half-channel count. "
                    f"Stage {index} expected {current * 4} but got {next_channels}."
                )
            if stride not in (1, 2):
                raise ValueError(f"Only stride values 1 and 2 are supported, got {stride}.")
            current = next_channels

        if self.n_strides[0] != 1:
            raise ValueError("The first stage must start at stride 1 for the paper-faithful photo preset.")

        target_half_channels = self.hidden_dim * (4 ** self.sp_steps)
        if target_half_channels < self.n_channels[-1]:
            raise ValueError(
                "Channel refinement must be injective: hidden_dim * 4**sp_steps "
                f"must be at least {self.n_channels[-1]}, got {target_half_channels}."
            )

    @classmethod
    def from_photo_v1(cls) -> "CAPColorTransferConfig":
        return cls()

    @property
    def downscale_factor(self) -> int:
        return prod(self.n_strides)

    @property
    def latent_channels(self) -> int:
        return self.hidden_dim * 2
