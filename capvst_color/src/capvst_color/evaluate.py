from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .config_io import (
    EVAL_CONFIG_DEFAULTS,
    PROJECT_ROOT,
    deep_merge_dicts,
    load_yaml_config,
    resolve_config_path,
)
from .datasets import (
    FiveKPairedValidationDataset,
    PhotorealBenchmarkDataset,
    discover_fivek_records,
    discover_photoreal_benchmark_records,
    split_fivek_records,
)
from .losses import VGGGramLoss
from .metrics import LPIPSMetric, align_bchw_to_reference, compute_hcorr, compute_psnr, compute_ssim, summarize_numeric_rows
from .model import CAPColorTransferModel
from .utils import ensure_dir, load_checkpoint, move_to_device, resolve_device, save_preview_strip, write_csv, write_json


def build_model_config(config: dict[str, Any]):
    from .train import build_model_config as _build_model_config

    return _build_model_config(config)


def load_vgg_encoder(*args, **kwargs):
    from .train import load_vgg_encoder as _load_vgg_encoder

    return _load_vgg_encoder(*args, **kwargs)


def run_photoreal_evaluation(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        config = load_yaml_config(config, defaults=EVAL_CONFIG_DEFAULTS)
    else:
        config = deep_merge_dicts(EVAL_CONFIG_DEFAULTS, config)

    device = resolve_device(str(config.get("device", "cuda")))
    model_config = build_model_config(config)
    model = CAPColorTransferModel(model_config).to(device)
    checkpoint_path = resolve_config_path(config["checkpoint"], config_path=config.get("config_path"))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vgg_encoder = load_vgg_encoder(model_config, device=device)
    gram_metric = VGGGramLoss(vgg_encoder)
    evaluation_cfg = config["evaluation"]

    output_root = ensure_dir(resolve_config_path(evaluation_cfg["output_dir"], config_path=config.get("config_path")))
    benchmark = None
    if bool(evaluation_cfg.get("run_benchmark", True)):
        benchmark = evaluate_photoreal_benchmark(
            model,
            gram_metric,
            config,
            device=device,
            output_dir=output_root / "photoreal_benchmark",
            render_limit=int(evaluation_cfg.get("benchmark_render_limit", 50)),
            enable_lpips=bool(evaluation_cfg.get("enable_lpips", True)),
            hcorr_bins=int(evaluation_cfg.get("hcorr_bins", 16)),
            save_images=bool(evaluation_cfg.get("save_images", False)),
        )
    fivek_sanity = None
    if bool(evaluation_cfg.get("run_fivek_sanity", True)):
        fivek_sanity = evaluate_fivek_sanity_split(
            model,
            config,
            device=device,
            output_dir=output_root / "fivek_sanity",
            render_limit=int(evaluation_cfg.get("fivek_render_limit", 50)),
            enable_lpips=bool(evaluation_cfg.get("enable_lpips", True)),
            save_images=bool(evaluation_cfg.get("save_images", False)),
        )
    if benchmark is None and fivek_sanity is None:
        raise ValueError("At least one evaluation target must be enabled.")

    summary: dict[str, Any] = {}
    if benchmark is not None:
        summary["photoreal_benchmark"] = benchmark["summary"]
    if fivek_sanity is not None:
        summary["fivek_sanity"] = fivek_sanity["summary"]
    write_json(output_root / "evaluation_summary.json", summary)
    return {
        "summary": summary,
        "photoreal_benchmark": benchmark,
        "fivek_sanity": fivek_sanity,
        "output_dir": str(output_root),
    }


@torch.inference_mode()
def evaluate_photoreal_benchmark(
    model: CAPColorTransferModel,
    gram_metric: VGGGramLoss,
    config: dict[str, Any],
    *,
    device: torch.device,
    output_dir: str | Path,
    render_limit: int,
    enable_lpips: bool,
    hcorr_bins: int,
    save_images: bool,
) -> dict[str, Any]:
    datasets_cfg = config["datasets"]["benchmark"]
    records = discover_photoreal_benchmark_records(
        resolve_config_path(datasets_cfg["content_dir"], config_path=config.get("config_path")),
        resolve_config_path(datasets_cfg["style_dir"], config_path=config.get("config_path")),
        resolve_config_path(datasets_cfg["gt_dir"], config_path=config.get("config_path")),
        image_extensions=datasets_cfg.get("image_extensions", (".jpg", ".jpeg", ".png")),
        content_basename_prefix=str(datasets_cfg.get("content_basename_prefix", "")),
        style_basename_prefix=str(datasets_cfg.get("style_basename_prefix", "")),
        gt_basename_prefix=str(datasets_cfg.get("gt_basename_prefix", "")),
        exclude_record_keys=tuple(str(key) for key in datasets_cfg.get("exclude_record_keys", ())),
    )
    dataset = PhotorealBenchmarkDataset(
        records,
        downscale_factor=model.downscale_factor,
        max_size=datasets_cfg.get("max_size"),
    )
    output_dir = ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images") if save_images else None
    lpips_metric = LPIPSMetric() if enable_lpips else None
    rows: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        batch = move_to_device(
            {
                "content_rgb": sample["content_rgb"].unsqueeze(0),
                "style_rgb": sample["style_rgb"].unsqueeze(0),
                "gt_rgb": sample["gt_rgb"].unsqueeze(0),
            },
            device,
        )
        output = model.stylize(batch["content_rgb"], batch["style_rgb"])
        gt_for_metrics = align_bchw_to_reference(output.stylized_rgb, batch["gt_rgb"])
        style_for_metrics = align_bchw_to_reference(output.stylized_rgb, batch["style_rgb"])
        row = {
            "basename": sample["basename"],
            "gram_loss": float(gram_metric(output.stylized_rgb, style_for_metrics).detach().cpu().item()),
            "h_corr": compute_hcorr(output.stylized_rgb, style_for_metrics, bins=hcorr_bins)[0],
            "psnr": compute_psnr(output.stylized_rgb, gt_for_metrics)[0],
            "ssim": compute_ssim(output.stylized_rgb, gt_for_metrics)[0],
        }
        if lpips_metric is not None and lpips_metric.available:
            row["lpips"] = lpips_metric(output.stylized_rgb, gt_for_metrics)[0]
        rows.append(row)

        if save_images and image_dir is not None and index < render_limit:
            save_preview_strip(
                image_dir / f"{sample['basename']}.png",
                content=sample["content_rgb"],
                style=sample["style_rgb"],
                stylized=output.stylized_rgb[0].cpu(),
                target=sample["gt_rgb"],
            )

    summary = summarize_numeric_rows(rows)
    summary["uses_masks"] = False
    summary["protocol_note"] = "mask-free approximation of the paper photorealistic benchmark"
    write_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return {"rows": rows, "summary": summary}


@torch.inference_mode()
def evaluate_fivek_sanity_split(
    model: CAPColorTransferModel,
    config: dict[str, Any],
    *,
    device: torch.device,
    output_dir: str | Path,
    render_limit: int,
    enable_lpips: bool,
    save_images: bool,
) -> dict[str, Any]:
    datasets_cfg = config["datasets"]["fivek"]
    records = discover_fivek_records(
        resolve_config_path(datasets_cfg["input_dir"], config_path=config.get("config_path")),
        resolve_config_path(datasets_cfg["style_dir"], config_path=config.get("config_path")),
        image_extensions=datasets_cfg.get("image_extensions", (".jpg", ".jpeg", ".png")),
    )
    _, val_records = split_fivek_records(
        records,
        train_count=int(datasets_cfg.get("train_count", 4500)),
        val_count=int(datasets_cfg.get("val_count", 500)),
        seed=int(config.get("seed", 42)),
    )
    evaluation_cfg = config["evaluation"]
    dataset = FiveKPairedValidationDataset(
        val_records,
        new_size=int(evaluation_cfg.get("fivek_new_size", 512)),
        crop_size=int(evaluation_cfg.get("fivek_crop_size", 256)),
        crop_mode=str(evaluation_cfg.get("fivek_crop_mode", "center")),
    )
    output_dir = ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images") if save_images else None
    lpips_metric = LPIPSMetric() if enable_lpips else None
    rows: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        batch = move_to_device(
            {
                "content_rgb": sample["content_rgb"].unsqueeze(0),
                "style_rgb": sample["style_rgb"].unsqueeze(0),
                "gt_rgb": sample["gt_rgb"].unsqueeze(0),
            },
            device,
        )
        output = model.stylize(batch["content_rgb"], batch["style_rgb"])
        row = {
            "basename": sample["basename"],
            "h_corr": compute_hcorr(output.stylized_rgb, batch["style_rgb"], bins=int(config["evaluation"].get("hcorr_bins", 16)))[0],
            "psnr": compute_psnr(output.stylized_rgb, batch["gt_rgb"])[0],
            "ssim": compute_ssim(output.stylized_rgb, batch["gt_rgb"])[0],
        }
        if lpips_metric is not None and lpips_metric.available:
            row["lpips"] = lpips_metric(output.stylized_rgb, batch["gt_rgb"])[0]
        rows.append(row)

        if save_images and image_dir is not None and index < render_limit:
            save_preview_strip(
                image_dir / f"{sample['basename']}.png",
                content=sample["content_rgb"],
                style=sample["style_rgb"],
                stylized=output.stylized_rgb[0].cpu(),
                target=sample["gt_rgb"],
            )

    summary = summarize_numeric_rows(rows)
    write_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean-room photorealistic CAP-VST evaluation stack.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "eval_photoreal.yaml"),
        help="Path to a JSON-compatible YAML config file.",
    )
    args = parser.parse_args()
    run_photoreal_evaluation(args.config)


if __name__ == "__main__":
    main()
