from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import validate_image_tensor


class LPIPSMetric:
    def __init__(self, *, net: str = "alex") -> None:
        self.net = net
        self._module = None

    @property
    def available(self) -> bool:
        if self._module is not None:
            return True
        try:
            import lpips  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
        prediction, target = _validate_batched_pair(prediction, target)
        module = self._get_module().to(device=prediction.device, dtype=prediction.dtype)
        values = module(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
        return values.flatten().detach().cpu().tolist()

    def _get_module(self):
        if self._module is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("lpips is not installed. Install the optional 'metrics' dependency.") from exc
            self._module = lpips.LPIPS(net=self.net).eval()
            for parameter in self._module.parameters():
                parameter.requires_grad = False
        return self._module


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> list[float]:
    prediction, target = _validate_batched_pair(prediction, target)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    data_term = prediction.new_tensor(float(data_range * data_range))
    psnr = torch.empty_like(mse)
    zero_mask = mse <= 1e-12
    psnr[zero_mask] = float("inf")
    psnr[~zero_mask] = 10.0 * torch.log10(data_term / mse[~zero_mask])
    return psnr.detach().cpu().tolist()


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> list[float]:
    prediction, target = _validate_batched_pair(prediction, target)
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean(dim=(1, 2, 3)).detach().cpu().tolist()


def compute_hcorr(prediction: torch.Tensor, style_rgb: torch.Tensor, *, bins: int = 16) -> list[float]:
    prediction, style_rgb = _validate_batched_pair(prediction, style_rgb)
    try:
        from skimage.color import rgb2lab
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for H-Corr computation. Install the optional 'metrics' dependency."
        ) from exc

    scores: list[float] = []
    pred_np = prediction.detach().cpu().permute(0, 2, 3, 1).numpy()
    style_np = style_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
    bin_ranges = ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))

    for pred_image, style_image in zip(pred_np, style_np):
        pred_lab = rgb2lab(np.clip(pred_image, 0.0, 1.0))
        style_lab = rgb2lab(np.clip(style_image, 0.0, 1.0))
        pred_hist = np.histogramdd(pred_lab.reshape(-1, 3), bins=bins, range=bin_ranges)[0].astype(np.float64)
        style_hist = np.histogramdd(style_lab.reshape(-1, 3), bins=bins, range=bin_ranges)[0].astype(np.float64)
        scores.append(_histogram_correlation(pred_hist.reshape(-1), style_hist.reshape(-1)))
    return scores


def align_bchw_to_reference(ref_bchw: torch.Tensor, x_bchw: torch.Tensor) -> torch.Tensor:
    ref_bchw = validate_image_tensor(ref_bchw, name="ref_bchw")
    x_bchw = validate_image_tensor(x_bchw, name="x_bchw")
    if ref_bchw.ndim == 3:
        ref_bchw = ref_bchw.unsqueeze(0)
    if x_bchw.ndim == 3:
        x_bchw = x_bchw.unsqueeze(0)
    if ref_bchw.shape[0] != x_bchw.shape[0]:
        raise ValueError(
            f"ref_bchw and x_bchw must have the same batch size, got {ref_bchw.shape[0]} and {x_bchw.shape[0]}."
        )
    if ref_bchw.shape[-2:] == x_bchw.shape[-2:]:
        return x_bchw
    return F.interpolate(x_bchw, size=ref_bchw.shape[-2:], mode="bilinear", align_corners=False)


def summarize_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "count"
        }
    )
    summary: dict[str, Any] = {"count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}.")
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def _validate_batched_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = validate_image_tensor(prediction, name="prediction")
    target = validate_image_tensor(target, name="target")
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}."
        )
    return prediction, target


def _histogram_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    left_mean = left.mean()
    right_mean = right.mean()
    left_centered = left - left_mean
    right_centered = right - right_mean
    numerator = float(np.sum(left_centered * right_centered))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return numerator / denominator
