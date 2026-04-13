from __future__ import annotations

from pathlib import Path

from capvst_color.datasets import (
    FiveKContentTrainDataset,
    FiveKPairedValidationDataset,
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
