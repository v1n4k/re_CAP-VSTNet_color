from __future__ import annotations

from pathlib import Path

import torch

from capvst_color import CAPColorTransferModel
from capvst_color.evaluate import run_photoreal_evaluation


def test_evaluation_runner_emits_benchmark_and_fivek_outputs(
    monkeypatch,
    small_fivek_root: Path,
    small_benchmark_root: Path,
    tmp_path: Path,
    dummy_encoder,
) -> None:
    import capvst_color.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "load_vgg_encoder", lambda *args, **kwargs: dummy_encoder)

    model = CAPColorTransferModel()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    config = {
        "device": "cpu",
        "checkpoint": str(checkpoint_path),
        "datasets": {
            "benchmark": {
                "content_dir": str(small_benchmark_root / "content"),
                "style_dir": str(small_benchmark_root / "style"),
                "gt_dir": str(small_benchmark_root / "gt"),
                "max_size": None,
            },
            "fivek": {
                "input_dir": str(small_fivek_root / "input"),
                "style_dir": str(small_fivek_root / "expert_c"),
                "train_count": 3,
                "val_count": 1,
            },
        },
        "evaluation": {
            "output_dir": str(tmp_path / "eval_outputs"),
            "uses_masks": False,
            "fivek_new_size": 32,
            "fivek_crop_size": 16,
            "fivek_crop_mode": "center",
            "benchmark_render_limit": 1,
            "fivek_render_limit": 1,
            "enable_lpips": False,
        },
    }

    result = run_photoreal_evaluation(config)

    output_dir = Path(result["output_dir"])
    assert (output_dir / "photoreal_benchmark" / "metrics.csv").exists()
    assert (output_dir / "fivek_sanity" / "metrics.csv").exists()
    assert result["summary"]["photoreal_benchmark"]["uses_masks"] is False
