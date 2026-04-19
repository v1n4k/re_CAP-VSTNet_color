from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRAIN_CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "datasets": {
        "fivek": {
            "input_dir": "/path/to/fivek/input",
            "style_dir": "/path/to/fivek/expert_c",
            "image_extensions": [".jpg", ".jpeg", ".png"],
            "train_count": 4500,
            "val_count": 500,
        }
    },
    "model": {},
    "training": {
        "batch_size": 2,
        "new_size": 512,
        "crop_size": 256,
        "num_workers": 0,
        "pin_memory": False,
        "win_radius": 1,
        "iterations": 160000,
        "lr": 1e-4,
        "lr_decay": 5e-5,
        "style_weight": 1.0,
        "content_weight": 0.0,
        "lap_weight": 1200.0,
        "rec_weight": 10.0,
        "grad_clip_norm": 5.0,
        "lap_gradient_clip": 0.05,
        "output_dir": "outputs/train/fivek",
        "resume_from": None,
        "checkpoint_every": 10000,
        "image_every": 1000,
        "log_every": 10,
        "validation_every": 10000,
        "validation_render_limit": 4,
    },
    "validation": {
        "new_size": 512,
        "crop_size": 256,
        "crop_mode": "center",
        "enable_lpips": True,
    },
}


EVAL_CONFIG_DEFAULTS: dict[str, Any] = {
    "device": "cuda",
    "checkpoint": "outputs/train_fivek/checkpoints/last.pt",
    "seed": 42,
    "datasets": {
        "benchmark": {
            "content_dir": "/path/to/photoreal/content",
            "style_dir": "/path/to/photoreal/style",
            "gt_dir": "/path/to/photoreal/gt",
            "image_extensions": [".jpg", ".jpeg", ".png"],
            "content_basename_prefix": "",
            "style_basename_prefix": "",
            "gt_basename_prefix": "",
            "exclude_record_keys": [],
            "max_size": None,
        },
        "fivek": {
            "input_dir": "/path/to/fivek/input",
            "style_dir": "/path/to/fivek/expert_c",
            "image_extensions": [".jpg", ".jpeg", ".png"],
            "train_count": 4500,
            "val_count": 500,
        },
    },
    "model": {},
    "evaluation": {
        "output_dir": "outputs/eval/photoreal",
        "uses_masks": False,
        "run_benchmark": True,
        "run_fivek_sanity": True,
        "hcorr_bins": 16,
        "save_images": False,
        "fivek_crop_mode": "center",
        "fivek_new_size": 512,
        "fivek_crop_size": 256,
        "benchmark_render_limit": 50,
        "fivek_render_limit": 50,
        "enable_lpips": True,
    },
}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config_path(path_value: str | Path | None, *, config_path: str | Path | None = None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if config_path is not None:
        return Path(config_path).resolve().parent / path
    return PROJECT_ROOT / path


def load_yaml_config(
    path: str | Path,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    data = _parse_config_text(text)
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must parse to a mapping, got {type(data).__name__}.")
    merged = deep_merge_dicts(defaults or {}, data)
    merged["config_path"] = str(path)
    return merged


def _parse_config_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.loads(text)
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping config, got {type(loaded).__name__}.")
    return loaded
