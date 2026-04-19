from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from capvst_color.datasets import (
    FiveKContentTrainDataset,
    FiveKPairedValidationDataset,
    discover_photoreal_benchmark_records,
    discover_fivek_records,
    split_fivek_records,
)


def test_fivek_discovery_and_split_are_deterministic(small_fivek_root: Path) -> None:
    records = discover_fivek_records(small_fivek_root / "input", small_fivek_root / "expert_c")
    train_records, val_records = split_fivek_records(records, train_count=3, val_count=1, seed=7)
    train_records_again, val_records_again = split_fivek_records(records, train_count=3, val_count=1, seed=7)

    assert len(records) == 4
    assert [record.basename for record in train_records] == [record.basename for record in train_records_again]
    assert [record.basename for record in val_records] == [record.basename for record in val_records_again]


def test_content_dataset_crop_and_laplacian_shapes_match(small_fivek_root: Path) -> None:
    records = discover_fivek_records(small_fivek_root / "input", small_fivek_root / "expert_c")
    dataset = FiveKContentTrainDataset(
        records,
        new_size=32,
        crop_size=16,
        win_radius=1,
        base_seed=13,
        deterministic=True,
    )

    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 16, 16)
    assert sample["laplacian"].shape == (16 * 16, 16 * 16)


def test_paired_validation_dataset_is_repeatable(small_fivek_root: Path) -> None:
    records = discover_fivek_records(small_fivek_root / "input", small_fivek_root / "expert_c")
    dataset = FiveKPairedValidationDataset(records, new_size=32, crop_size=16, crop_mode="center")

    first = dataset[0]
    second = dataset[0]
    assert first["basename"] == second["basename"]
    assert first["content_rgb"].equal(second["content_rgb"])
    assert first["style_rgb"].equal(second["style_rgb"])


def test_paired_validation_dataset_aligns_mismatched_sizes(tmp_path: Path) -> None:
    root = tmp_path / "fivek"
    input_dir = root / "input"
    style_dir = root / "expert_c"
    input_dir.mkdir(parents=True)
    style_dir.mkdir(parents=True)

    Image.fromarray(np.zeros((2920, 4386, 3), dtype=np.uint8), mode="RGB").save(input_dir / "pair.png")
    Image.fromarray(np.zeros((2912, 4368, 3), dtype=np.uint8), mode="RGB").save(style_dir / "pair.png")

    records = discover_fivek_records(input_dir, style_dir)
    dataset = FiveKPairedValidationDataset(records, new_size=512, crop_size=256, crop_mode="center")

    sample = dataset[0]
    assert sample["basename"] == "pair"
    assert tuple(sample["content_rgb"].shape) == (3, 256, 256)
    assert tuple(sample["style_rgb"].shape) == (3, 256, 256)
    assert sample["style_rgb"].equal(sample["gt_rgb"])


def test_photoreal_benchmark_discovery_can_exclude_record_keys(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    content_dir = root / "content"
    style_dir = root / "style"
    gt_dir = root / "gt"
    content_dir.mkdir(parents=True)
    style_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    for stem in ("keep", "drop"):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(content_dir / f"{stem}.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(style_dir / f"{stem}.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(gt_dir / f"{stem}.png")

    records = discover_photoreal_benchmark_records(
        content_dir,
        style_dir,
        gt_dir,
        exclude_record_keys=("drop",),
    )

    assert [record.basename for record in records] == ["keep"]
