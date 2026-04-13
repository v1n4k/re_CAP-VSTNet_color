from __future__ import annotations

import torch

from capvst_color.laplacian import compute_matting_laplacian, laplacian_quadratic_loss_and_gradient
from capvst_color.losses import VGGGramLoss, VGGStyleLoss


def test_style_and_gram_losses_are_finite(dummy_encoder) -> None:
    torch.manual_seed(3)
    prediction = torch.rand(1, 3, 16, 16)
    style = torch.rand(1, 3, 16, 16)

    style_loss = VGGStyleLoss(dummy_encoder)(prediction, style)
    gram_loss = VGGGramLoss(dummy_encoder)(prediction, style)

    assert torch.isfinite(style_loss)
    assert torch.isfinite(gram_loss)


def test_matting_laplacian_gradient_is_finite() -> None:
    torch.manual_seed(4)
    image = torch.rand(3, 8, 8)
    laplacian = compute_matting_laplacian(image, win_rad=1)

    loss, gradient = laplacian_quadratic_loss_and_gradient(image, laplacian)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.shape == image.shape
