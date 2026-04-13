from __future__ import annotations

from pathlib import Path

from capvst_color import CAPColorTransferModel, VGG19Encoder


def test_core_model_does_not_require_vgg_checkpoint_at_construction() -> None:
    model = CAPColorTransferModel()
    assert model is not None


def test_missing_vgg_checkpoint_fails_only_when_vgg_is_instantiated(tmp_path: Path) -> None:
    missing_checkpoint = tmp_path / "missing_vgg.pth"

    try:
        VGG19Encoder.from_checkpoint(missing_checkpoint)
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing VGG checkpoint.")

    assert str(missing_checkpoint) in message
