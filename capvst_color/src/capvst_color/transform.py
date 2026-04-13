from __future__ import annotations

import torch
from torch import nn


class CholeskyWCT(nn.Module):
    def __init__(
        self,
        *,
        epsilon: float = 2e-5,
        max_attempts: int = 8,
        jitter_growth: float = 10.0,
        use_double: bool = True,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if max_attempts <= 0:
            raise ValueError(f"max_attempts must be positive, got {max_attempts}.")
        if jitter_growth <= 1.0:
            raise ValueError(f"jitter_growth must be greater than 1.0, got {jitter_growth}.")
        self.epsilon = epsilon
        self.max_attempts = max_attempts
        self.jitter_growth = jitter_growth
        self.use_double = use_double

    def forward(self, content_latent: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        return self.transfer(content_latent, style_latent)

    def transfer(self, content_latent: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        if content_latent.ndim != 4 or style_latent.ndim != 4:
            raise ValueError(
                "content_latent and style_latent must both be 4D tensors, "
                f"got {tuple(content_latent.shape)} and {tuple(style_latent.shape)}."
            )
        if content_latent.shape[0] != style_latent.shape[0]:
            raise ValueError(
                "content_latent and style_latent must share the same batch size, "
                f"got {content_latent.shape[0]} and {style_latent.shape[0]}."
            )
        if content_latent.shape[1] != style_latent.shape[1]:
            raise ValueError(
                "content_latent and style_latent must share the same channel count, "
                f"got {content_latent.shape[1]} and {style_latent.shape[1]}."
            )

        input_dtype = content_latent.dtype
        content_flat = content_latent.flatten(start_dim=2)
        style_flat = style_latent.flatten(start_dim=2)

        if self.use_double:
            content_flat = content_flat.to(dtype=torch.float64)
            style_flat = style_flat.to(dtype=torch.float64)

        whitened = self._whiten(content_flat)
        stylized = self._color(whitened, style_flat)

        if self.use_double:
            stylized = stylized.to(dtype=input_dtype)

        return stylized.reshape_as(content_latent)

    def _whiten(self, features: torch.Tensor) -> torch.Tensor:
        centered, _ = self._center(features)
        covariance = self._covariance(centered)
        chol = self._stable_cholesky(covariance)
        return self._solve_lower(chol, centered)

    def _color(self, whitened_content: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        centered_style, style_mean = self._center(style_features)
        covariance = self._covariance(centered_style)
        chol = self._stable_cholesky(covariance)
        colored = chol @ whitened_content
        return colored + style_mean

    def _center(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = features.mean(dim=-1, keepdim=True)
        return features - mean, mean

    def _covariance(self, centered: torch.Tensor) -> torch.Tensor:
        sample_count = centered.shape[-1]
        denom = max(sample_count - 1, 1)
        return (centered @ centered.transpose(-1, -2)) / denom

    def _stable_cholesky(self, covariance: torch.Tensor) -> torch.Tensor:
        channels = covariance.shape[-1]
        eye = torch.eye(channels, dtype=covariance.dtype, device=covariance.device).expand(
            covariance.shape[0], -1, -1
        )
        jitter = self.epsilon
        for _ in range(self.max_attempts):
            try:
                return torch.linalg.cholesky(covariance + eye * jitter)
            except RuntimeError:
                jitter *= self.jitter_growth
        raise RuntimeError(
            "Cholesky decomposition failed after "
            f"{self.max_attempts} attempts; final jitter was {jitter / self.jitter_growth:.3e}."
        )

    def _solve_lower(self, lower_triangular: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.linalg, "solve_triangular"):
            return torch.linalg.solve_triangular(lower_triangular, rhs, upper=False)
        solution, _ = torch.triangular_solve(rhs, lower_triangular, upper=False)
        return solution
