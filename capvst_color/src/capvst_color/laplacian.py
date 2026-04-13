from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
import torch

torch.sparse.check_sparse_tensor_invariants.disable()


def _to_image_array(image: torch.Tensor | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image, dtype=np.float64)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            array = tensor.permute(1, 2, 0).numpy().astype(np.float64, copy=False)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            array = tensor.numpy().astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported tensor shape for matting Laplacian: {tuple(tensor.shape)}.")
    else:
        array = np.asarray(image, dtype=np.float64)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got array with shape {array.shape}.")
    if array.max() > 1.0:
        array = array / 255.0
    return array


def compute_matting_laplacian(
    image: torch.Tensor | np.ndarray | Image.Image,
    *,
    win_rad: int = 1,
    eps: float = 1e-7,
) -> torch.Tensor:
    image_array = _to_image_array(image)
    if win_rad < 0:
        raise ValueError(f"win_rad must be non-negative, got {win_rad}.")
    height, width, channels = image_array.shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}.")

    window_diameter = 2 * win_rad + 1
    window_size = window_diameter * window_diameter
    if height < window_diameter or width < window_diameter:
        raise ValueError(
            f"Image size {(height, width)} is too small for window radius {win_rad}."
        )

    flat_indices = np.arange(height * width, dtype=np.int64).reshape(height, width)
    window_indices = sliding_window_view(flat_indices, (window_diameter, window_diameter))
    window_indices = window_indices.reshape(-1, window_size)

    flat_pixels = image_array.reshape(height * width, channels)
    window_pixels = flat_pixels[window_indices]
    means = window_pixels.mean(axis=1, keepdims=True)
    centered = window_pixels - means
    covariances = np.einsum("nki,nkj->nij", centered, centered) / float(window_size)
    regularizer = (eps / float(window_size)) * np.eye(channels, dtype=np.float64)
    inverse_covariances = np.linalg.inv(covariances + regularizer[None, :, :])

    projected = np.einsum("nki,nij->nkj", centered, inverse_covariances)
    affinities = (1.0 + np.einsum("nki,nli->nkl", projected, centered)) / float(window_size)

    row_indices = np.repeat(window_indices, window_size, axis=1).reshape(-1)
    col_indices = np.tile(window_indices, (1, window_size)).reshape(-1)
    values = affinities.reshape(-1).astype(np.float32, copy=False)

    indices = torch.from_numpy(np.stack((row_indices, col_indices)))
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(values),
        (height * width, height * width),
        dtype=torch.float32,
    ).coalesce()

    degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
    diagonal_index = torch.arange(height * width, dtype=torch.long)
    diagonal = torch.sparse_coo_tensor(
        torch.stack((diagonal_index, diagonal_index)),
        degrees,
        (height * width, height * width),
        dtype=torch.float32,
    )
    return (diagonal - adjacency).coalesce()


def laplacian_quadratic_loss_and_gradient(
    image: torch.Tensor,
    laplacian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must have shape (3, H, W), got {tuple(image.shape)}.")
    if not laplacian.is_sparse:
        raise TypeError("laplacian must be a sparse COO tensor.")

    laplacian = laplacian.coalesce().to(device=image.device, dtype=image.dtype)
    _, height, width = image.shape
    pixel_count = float(height * width)

    total_loss = image.new_zeros(())
    channel_grads: list[torch.Tensor] = []
    for channel in range(image.shape[0]):
        flat = image[channel].reshape(-1, 1)
        grad = torch.sparse.mm(laplacian, flat) / pixel_count
        total_loss = total_loss + torch.matmul(flat.transpose(0, 1), grad).squeeze()
        channel_grads.append((2.0 * grad).reshape(height, width))

    return total_loss, torch.stack(channel_grads, dim=0)
