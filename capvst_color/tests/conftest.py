from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DummyEncoder(nn.Module):
    def encode_with_intermediate(self, tensor: torch.Tensor, n_layer: int = 4) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        current = tensor
        for _ in range(n_layer):
            features.append(current)
            if current.shape[-2] > 1 and current.shape[-1] > 1:
                current = F.avg_pool2d(current, kernel_size=2, stride=2, ceil_mode=True)
        return features


@pytest.fixture
def dummy_encoder() -> DummyEncoder:
    return DummyEncoder()


def write_rgb_image(path: Path, *, seed: int, size: tuple[int, int] = (40, 40)) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


@pytest.fixture
def small_fivek_root(tmp_path: Path) -> Path:
    input_dir = tmp_path / "fivek" / "input"
    style_dir = tmp_path / "fivek" / "expert_c"
    for index in range(4):
        basename = f"img_{index:02d}.png"
        write_rgb_image(input_dir / basename, seed=index)
        write_rgb_image(style_dir / basename, seed=index + 100)
    return tmp_path / "fivek"


@pytest.fixture
def small_benchmark_root(tmp_path: Path) -> Path:
    content_dir = tmp_path / "benchmark" / "content"
    style_dir = tmp_path / "benchmark" / "style"
    gt_dir = tmp_path / "benchmark" / "gt"
    for index in range(2):
        basename = f"pair_{index:02d}.png"
        write_rgb_image(content_dir / basename, seed=index + 200, size=(36, 36))
        write_rgb_image(style_dir / basename, seed=index + 300, size=(36, 36))
        write_rgb_image(gt_dir / basename, seed=index + 400, size=(36, 36))
    return tmp_path / "benchmark"


@pytest.fixture
def small_dpst_root(tmp_path: Path) -> Path:
    input_dir = tmp_path / "dpst" / "input"
    style_dir = tmp_path / "dpst" / "style"
    for index in range(2):
        suffix = f"{index:02d}"
        write_rgb_image(input_dir / f"in{suffix}.png", seed=index + 500, size=(36, 36))
        write_rgb_image(style_dir / f"tar{suffix}.png", seed=index + 600, size=(36, 36))
    return tmp_path / "dpst"
