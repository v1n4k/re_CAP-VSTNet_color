from __future__ import annotations

import torch

from capvst_color import CAPColorTransferConfig, CAPColorTransferModel, CholeskyWCT


def test_default_model_builds_and_exposes_downscale_factor() -> None:
    config = CAPColorTransferConfig.from_photo_v1()
    model = CAPColorTransferModel(config)

    assert model.downscale_factor == 4
    assert model.latent_channels == 32


def test_backbone_round_trip_preserves_shape_and_is_numerically_tight() -> None:
    torch.manual_seed(0)
    model = CAPColorTransferModel()
    image = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        latent = model.encode(image)
        reconstruction = model.decode(latent)

    assert reconstruction.shape == image.shape
    max_error = (reconstruction - image).abs().max().item()
    assert max_error < 1e-6


def test_cholesky_wct_supports_different_spatial_sizes_and_returns_finite_output() -> None:
    torch.manual_seed(1)
    transfer = CholeskyWCT()
    content = torch.rand(2, 32, 64, 64)
    style = torch.rand(2, 32, 32, 48)

    with torch.no_grad():
        output = transfer.transfer(content, style)

    assert output.shape == content.shape
    assert output.dtype == content.dtype
    assert torch.isfinite(output).all()


def test_stylize_returns_populated_output_for_256_square_inputs() -> None:
    torch.manual_seed(2)
    model = CAPColorTransferModel()
    content = torch.rand(1, 3, 256, 256)
    style = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        output = model.stylize(content, style)

    assert output.stylized_rgb.shape == content.shape
    assert output.content_latent.shape == (1, 32, 256, 256)
    assert output.style_latent.shape == (1, 32, 256, 256)
    assert output.stylized_latent.shape == (1, 32, 256, 256)
    assert torch.isfinite(output.stylized_rgb).all()
    assert torch.isfinite(output.stylized_latent).all()


def test_non_divisible_spatial_shape_raises_clear_error() -> None:
    model = CAPColorTransferModel()
    bad_content = torch.rand(1, 3, 255, 256)
    style = torch.rand(1, 3, 256, 256)

    try:
        model.stylize(bad_content, style)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected a ValueError for non-divisible spatial dimensions.")

    assert "divisible by the backbone downscale factor" in message
