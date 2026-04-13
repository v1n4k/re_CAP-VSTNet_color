from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset

from .laplacian import compute_matting_laplacian
from .preprocess import (
    center_crop,
    load_rgb_image,
    paired_center_crop,
    pil_to_tensor,
    random_crop,
    resize_image_for_benchmark,
    resize_pair_like_content,
    resize_short_edge,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm")


@dataclass(frozen=True, slots=True)
class FiveKRecord:
    basename: str
    input_path: Path
    style_path: Path


@dataclass(frozen=True, slots=True)
class PhotorealBenchmarkRecord:
    basename: str
    content_path: Path
    style_path: Path
    gt_path: Path


def discover_fivek_records(
    input_dir: str | Path,
    style_dir: str | Path,
    *,
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[FiveKRecord]:
    input_files = _discover_files(Path(input_dir), image_extensions)
    style_files = _discover_files(Path(style_dir), image_extensions)
    basenames = sorted(set(input_files) & set(style_files))
    if not basenames:
        raise ValueError("No matched FiveK basenames were found between input_dir and style_dir.")
    return [
        FiveKRecord(
            basename=basename,
            input_path=input_files[basename],
            style_path=style_files[basename],
        )
        for basename in basenames
    ]


def split_fivek_records(
    records: list[FiveKRecord],
    *,
    train_count: int = 4500,
    val_count: int = 500,
    seed: int = 42,
) -> tuple[list[FiveKRecord], list[FiveKRecord]]:
    if train_count < 0 or val_count < 0:
        raise ValueError(f"train_count and val_count must be non-negative, got {train_count} and {val_count}.")
    if len(records) < train_count + val_count:
        raise ValueError(
            f"Need at least {train_count + val_count} FiveK records, got {len(records)}."
        )
    shuffled = list(records)
    shuffled.sort(key=lambda record: record.basename)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count : train_count + val_count]


def discover_photoreal_benchmark_records(
    content_dir: str | Path,
    style_dir: str | Path,
    gt_dir: str | Path,
    *,
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[PhotorealBenchmarkRecord]:
    content_files = _discover_files(Path(content_dir), image_extensions)
    style_files = _discover_files(Path(style_dir), image_extensions)
    gt_files = _discover_files(Path(gt_dir), image_extensions)
    basenames = sorted(set(content_files) & set(style_files) & set(gt_files))
    if not basenames:
        raise ValueError("No matched photoreal benchmark basenames were found.")
    return [
        PhotorealBenchmarkRecord(
            basename=basename,
            content_path=content_files[basename],
            style_path=style_files[basename],
            gt_path=gt_files[basename],
        )
        for basename in basenames
    ]


class FiveKContentTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[FiveKRecord],
        *,
        new_size: int = 512,
        crop_size: int = 256,
        win_radius: int = 1,
        base_seed: int = 42,
        deterministic: bool = False,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty.")
        self.records = list(records)
        self.new_size = new_size
        self.crop_size = crop_size
        self.win_radius = win_radius
        self.base_seed = base_seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = random.Random(self.base_seed + index) if self.deterministic else random
        image = resize_short_edge(load_rgb_image(record.input_path), self.new_size)
        crop = random_crop(image, self.crop_size, rng=rng)
        rgb = pil_to_tensor(crop)
        laplacian = compute_matting_laplacian(rgb, win_rad=self.win_radius)
        return {
            "basename": record.basename,
            "image": rgb,
            "laplacian": laplacian,
        }


class ImagePoolTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        image_paths: list[Path],
        *,
        basenames: list[str],
        new_size: int = 512,
        crop_size: int = 256,
        base_seed: int = 42,
        deterministic: bool = False,
    ) -> None:
        if not image_paths:
            raise ValueError("image_paths must not be empty.")
        if len(image_paths) != len(basenames):
            raise ValueError("image_paths and basenames must have the same length.")
        self.image_paths = list(image_paths)
        self.basenames = list(basenames)
        self.new_size = new_size
        self.crop_size = crop_size
        self.base_seed = base_seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.base_seed + index) if self.deterministic else random
        image = resize_short_edge(load_rgb_image(self.image_paths[index]), self.new_size)
        crop = random_crop(image, self.crop_size, rng=rng)
        return {
            "basename": self.basenames[index],
            "image": pil_to_tensor(crop),
        }


class FiveKPairedValidationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[FiveKRecord],
        *,
        new_size: int = 512,
        crop_size: int = 256,
        crop_mode: str = "center",
    ) -> None:
        if not records:
            raise ValueError("records must not be empty.")
        if crop_mode not in {"center"}:
            raise ValueError(f"Unsupported crop_mode {crop_mode!r}.")
        self.records = list(records)
        self.new_size = new_size
        self.crop_size = crop_size
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        content = resize_short_edge(load_rgb_image(record.input_path), self.new_size)
        target = resize_short_edge(load_rgb_image(record.style_path), self.new_size)
        content_crop, target_crop = paired_center_crop(content, target, self.crop_size)
        return {
            "basename": record.basename,
            "content_rgb": pil_to_tensor(content_crop),
            "style_rgb": pil_to_tensor(target_crop),
            "gt_rgb": pil_to_tensor(target_crop.copy()),
        }


class PhotorealBenchmarkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[PhotorealBenchmarkRecord],
        *,
        downscale_factor: int,
        max_size: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty.")
        self.records = list(records)
        self.downscale_factor = downscale_factor
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        content = load_rgb_image(record.content_path)
        gt = load_rgb_image(record.gt_path)
        style = load_rgb_image(record.style_path)
        content, gt = resize_pair_like_content(
            content,
            gt,
            max_size=self.max_size,
            downscale_factor=self.downscale_factor,
        )
        style = resize_image_for_benchmark(
            style,
            max_size=self.max_size,
            downscale_factor=self.downscale_factor,
        )
        return {
            "basename": record.basename,
            "content_rgb": pil_to_tensor(content),
            "style_rgb": pil_to_tensor(style),
            "gt_rgb": pil_to_tensor(gt),
        }


def collate_content_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "basename": [item["basename"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "laplacian": [item["laplacian"] for item in batch],
    }


def collate_image_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"basename": [item["basename"] for item in batch]}
    for key in ("image", "content_rgb", "style_rgb", "gt_rgb"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result


def _discover_files(directory: Path, image_extensions: Iterable[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")
    allowed = {extension.lower() for extension in image_extensions}
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        files[path.stem] = path
    return files
