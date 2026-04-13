from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
import torch

from .preprocess import tensor_to_pil


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def adjust_inverse_time_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    lr_decay: float,
    iteration: int,
) -> float:
    lr = base_lr / (1.0 + lr_decay * iteration)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def iterate_forever(loader: torch.utils.data.DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {path} must load to a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({key for row in rows for key in row})
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.8f}")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_preview_strip(path: str | Path, **images: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [tensor_to_pil(image) for image in images.values()]
    widths = [image.width for image in pil_images]
    heights = [image.height for image in pil_images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(0, 0, 0))
    cursor = 0
    for image in pil_images:
        canvas.paste(image, (cursor, 0))
        cursor += image.width
    canvas.save(path)
