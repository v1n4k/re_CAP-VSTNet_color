from __future__ import annotations

from pathlib import Path

import torch

from capvst_color.train import run_training


def test_cpu_training_smoke_run_emits_checkpoints_and_logs(
    monkeypatch,
    small_fivek_root: Path,
    tmp_path: Path,
    dummy_encoder,
) -> None:
    import capvst_color.train as train_module

    monkeypatch.setattr(train_module, "load_vgg_encoder", lambda *args, **kwargs: dummy_encoder)

    config = {
        "device": "cpu",
        "datasets": {
            "fivek": {
                "input_dir": str(small_fivek_root / "input"),
                "style_dir": str(small_fivek_root / "expert_c"),
                "train_count": 3,
                "val_count": 1,
            }
        },
        "training": {
            "batch_size": 1,
            "new_size": 32,
            "crop_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "iterations": 1,
            "checkpoint_every": 1,
            "validation_every": 1,
            "image_every": 1,
            "log_every": 1,
            "output_dir": str(tmp_path / "train_outputs"),
            "validation_render_limit": 1,
        },
        "validation": {
            "new_size": 32,
            "crop_size": 16,
            "crop_mode": "center",
            "enable_lpips": False,
        },
    }

    summary = run_training(config)

    output_dir = Path(summary["output_dir"])
    assert (output_dir / "checkpoints" / "last.pt").exists()
    assert (output_dir / "checkpoints" / "best.pt").exists()
    assert (output_dir / "logs" / "train_metrics.jsonl").exists()
    assert (output_dir / "logs" / "validation_metrics.jsonl").exists()
    assert summary["train_records"] == 3
    assert summary["val_records"] == 1
