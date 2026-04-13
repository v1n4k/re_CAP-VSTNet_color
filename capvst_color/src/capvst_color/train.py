from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal local environments
    def tqdm(iterable, **_: Any):
        return iterable

from .config import CAPColorTransferConfig
from .config_io import (
    PROJECT_ROOT,
    TRAIN_CONFIG_DEFAULTS,
    deep_merge_dicts,
    load_yaml_config,
    resolve_config_path,
)
from .datasets import (
    FiveKContentTrainDataset,
    FiveKPairedValidationDataset,
    ImagePoolTrainDataset,
    collate_content_batch,
    collate_image_batch,
    discover_fivek_records,
    split_fivek_records,
)
from .laplacian import laplacian_quadratic_loss_and_gradient
from .losses import VGGStyleLoss
from .metrics import LPIPSMetric, compute_psnr, compute_ssim, summarize_numeric_rows
from .model import CAPColorTransferModel
from .utils import (
    adjust_inverse_time_lr,
    append_jsonl,
    ensure_dir,
    iterate_forever,
    load_checkpoint,
    move_to_device,
    resolve_device,
    save_checkpoint,
    save_preview_strip,
    set_random_seed,
    write_json,
)
from .vgg import VGG19Encoder


def build_fivek_dataloaders(
    config: dict[str, Any],
    *,
    downscale_factor: int,
) -> tuple[dict[str, DataLoader], dict[str, list[Any]]]:
    datasets_cfg = config["datasets"]["fivek"]
    training_cfg = config["training"]
    validation_cfg = config["validation"]

    records = discover_fivek_records(
        resolve_config_path(datasets_cfg["input_dir"], config_path=config.get("config_path")),
        resolve_config_path(datasets_cfg["style_dir"], config_path=config.get("config_path")),
        image_extensions=datasets_cfg.get("image_extensions", (".jpg", ".jpeg", ".png")),
    )
    train_records, val_records = split_fivek_records(
        records,
        train_count=int(datasets_cfg.get("train_count", 4500)),
        val_count=int(datasets_cfg.get("val_count", 500)),
        seed=int(config.get("seed", 42)),
    )

    new_size = int(training_cfg.get("new_size", 512))
    crop_size = int(training_cfg.get("crop_size", 256))
    if new_size < crop_size:
        raise ValueError(f"training.new_size must be >= crop_size, got {new_size} and {crop_size}.")
    if crop_size % downscale_factor != 0:
        raise ValueError(
            f"training.crop_size must be divisible by the model downscale factor {downscale_factor}, got {crop_size}."
        )

    batch_size = int(training_cfg.get("batch_size", 2))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))
    base_seed = int(config.get("seed", 42))

    content_train = FiveKContentTrainDataset(
        train_records,
        new_size=new_size,
        crop_size=crop_size,
        win_radius=int(training_cfg.get("win_radius", 1)),
        base_seed=base_seed,
        deterministic=False,
    )
    style_train = ImagePoolTrainDataset(
        image_paths=[record.style_path for record in train_records],
        basenames=[record.basename for record in train_records],
        new_size=new_size,
        crop_size=crop_size,
        base_seed=base_seed + 100000,
        deterministic=False,
    )
    fivek_validation = FiveKPairedValidationDataset(
        val_records,
        new_size=int(validation_cfg.get("new_size", 512)),
        crop_size=int(validation_cfg.get("crop_size", 256)),
        crop_mode=str(validation_cfg.get("crop_mode", "center")),
    )

    loaders = {
        "content_train": DataLoader(
            content_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_content_batch,
        ),
        "style_train": DataLoader(
            style_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_image_batch,
        ),
        "fivek_validation": DataLoader(
            fivek_validation,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_image_batch,
        ),
    }
    return loaders, {"train_records": train_records, "val_records": val_records}


def load_vgg_encoder(
    model_config: CAPColorTransferConfig,
    *,
    device: torch.device,
) -> nn.Module:
    encoder = VGG19Encoder.from_checkpoint(model_config.vgg_checkpoint_path)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.inference_mode()
def evaluate_fivek_sanity(
    model: CAPColorTransferModel,
    dataset_loader: DataLoader,
    *,
    device: torch.device,
    enable_lpips: bool = True,
    render_dir: str | Path | None = None,
    render_limit: int = 0,
) -> dict[str, Any]:
    model.eval()
    lpips_metric = LPIPSMetric() if enable_lpips else None
    rows: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(dataset_loader):
        batch = move_to_device(batch, device)
        output = model.stylize(batch["content_rgb"], batch["style_rgb"])
        row = {
            "basename": batch["basename"][0],
            "psnr": compute_psnr(output.stylized_rgb, batch["gt_rgb"])[0],
            "ssim": compute_ssim(output.stylized_rgb, batch["gt_rgb"])[0],
        }
        if lpips_metric is not None and lpips_metric.available:
            row["lpips"] = lpips_metric(output.stylized_rgb, batch["gt_rgb"])[0]
        rows.append(row)

        if render_dir is not None and batch_index < render_limit:
            save_preview_strip(
                Path(render_dir) / f"{batch['basename'][0]}.png",
                content=batch["content_rgb"][0].cpu(),
                style=batch["style_rgb"][0].cpu(),
                stylized=output.stylized_rgb[0].cpu(),
                target=batch["gt_rgb"][0].cpu(),
            )

    return {"rows": rows, "summary": summarize_numeric_rows(rows)}


def run_training(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        config = load_yaml_config(config, defaults=TRAIN_CONFIG_DEFAULTS)
    else:
        config = deep_merge_dicts(TRAIN_CONFIG_DEFAULTS, config)

    set_random_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cuda")))
    model_config = build_model_config(config)
    model = CAPColorTransferModel(model_config).to(device)
    style_encoder = load_vgg_encoder(model_config, device=device)
    style_loss_fn = VGGStyleLoss(style_encoder)
    reconstruction_loss = nn.L1Loss()

    dataloaders, split_info = build_fivek_dataloaders(config, downscale_factor=model.downscale_factor)
    content_iterator = iterate_forever(dataloaders["content_train"])
    style_iterator = iterate_forever(dataloaders["style_train"])

    training_cfg = config["training"]
    validation_cfg = config["validation"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("lr", 1e-4)))

    output_root = ensure_dir(resolve_config_path(training_cfg["output_dir"], config_path=config.get("config_path")))
    checkpoint_dir = ensure_dir(output_root / "checkpoints")
    image_dir = ensure_dir(output_root / "images")
    log_dir = ensure_dir(output_root / "logs")

    start_iteration = 0
    skipped_batches = 0
    resume_from = training_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_config_path(resume_from, config_path=config.get("config_path"))
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = int(checkpoint.get("iteration", -1)) + 1
        skipped_batches = int(checkpoint.get("skipped_batches", 0))

    iterations = int(training_cfg.get("iterations", 160000))
    best_val_ssim = float("-inf")
    progress = tqdm(range(start_iteration, iterations), desc="Training", leave=False)
    for iteration in progress:
        content_batch = next(content_iterator)
        style_batch = next(style_iterator)

        content_batch = move_to_device(
            {"basename": content_batch["basename"], "image": content_batch["image"]},
            device,
        ) | {"laplacian": [lap.coalesce() for lap in content_batch["laplacian"]]}
        style_batch = move_to_device(style_batch, device)

        optimizer.zero_grad(set_to_none=True)
        current_lr = adjust_inverse_time_lr(
            optimizer,
            base_lr=float(training_cfg.get("lr", 1e-4)),
            lr_decay=float(training_cfg.get("lr_decay", 5e-5)),
            iteration=iteration,
        )

        try:
            output = model.stylize(content_batch["image"], style_batch["image"])
        except RuntimeError as error:
            if "Cholesky decomposition failed" in str(error):
                skipped_batches += 1
                append_jsonl(
                    log_dir / "train_metrics.jsonl",
                    {"iteration": iteration + 1, "event": "skip", "reason": str(error)},
                )
                continue
            raise

        style_loss = style_loss_fn(output.stylized_rgb, style_batch["image"])
        reconstructed_rgb = output.stylized_rgb.new_zeros(output.stylized_rgb.shape)
        rec_loss = output.stylized_rgb.new_zeros(())
        if float(training_cfg.get("rec_weight", 10.0)) > 0:
            try:
                reencoded = model.encode(output.stylized_rgb)
                rec_latent = model.transfer(reencoded, output.content_latent)
                reconstructed_rgb = model.decode(rec_latent)
            except RuntimeError as error:
                if "Cholesky decomposition failed" in str(error):
                    skipped_batches += 1
                    append_jsonl(
                        log_dir / "train_metrics.jsonl",
                        {"iteration": iteration + 1, "event": "skip", "reason": str(error)},
                    )
                    continue
                raise
            rec_loss = reconstruction_loss(reconstructed_rgb, content_batch["image"])

        if float(training_cfg.get("lap_weight", 1200.0)) > 0:
            lap_losses = []
            lap_grads = []
            for stylized_image, laplacian in zip(output.stylized_rgb, content_batch["laplacian"]):
                loss_value, grad_value = laplacian_quadratic_loss_and_gradient(
                    stylized_image,
                    laplacian.to(device),
                )
                lap_losses.append(loss_value)
                lap_grads.append(grad_value)
            lap_grad = torch.stack(lap_grads, dim=0) * float(training_cfg.get("lap_weight", 1200.0))
            lap_grad = lap_grad.clamp(
                -float(training_cfg.get("lap_gradient_clip", 0.05)),
                float(training_cfg.get("lap_gradient_clip", 0.05)),
            )
            output.stylized_rgb.backward(lap_grad, retain_graph=True)
            lap_loss = torch.stack(lap_losses, dim=0).mean()
        else:
            lap_loss = output.stylized_rgb.new_zeros(())

        total_loss = (
            float(training_cfg.get("style_weight", 1.0)) * style_loss
            + float(training_cfg.get("content_weight", 0.0)) * output.stylized_rgb.new_zeros(())
            + float(training_cfg.get("rec_weight", 10.0)) * rec_loss
        )
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip_norm", 5.0)))
        optimizer.step()

        metrics = {
            "iteration": iteration + 1,
            "lr": current_lr,
            "style_loss": float(style_loss.detach().cpu().item()),
            "rec_loss": float(rec_loss.detach().cpu().item()),
            "lap_loss": float(lap_loss.detach().cpu().item()),
            "skipped_batches": skipped_batches,
        }
        if (iteration + 1) % int(training_cfg.get("log_every", 10)) == 0:
            append_jsonl(log_dir / "train_metrics.jsonl", metrics)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix({"style": f"{metrics['style_loss']:.4f}", "rec": f"{metrics['rec_loss']:.4f}"})

        if (iteration + 1) % int(training_cfg.get("image_every", 1000)) == 0:
            render_count = int(training_cfg.get("validation_render_limit", 4))
            for preview_index in range(min(render_count, output.stylized_rgb.shape[0])):
                save_preview_strip(
                    image_dir / f"train_{iteration + 1:08d}_{preview_index}.png",
                    content=content_batch["image"][preview_index].detach().cpu(),
                    style=style_batch["image"][preview_index].detach().cpu(),
                    stylized=output.stylized_rgb[preview_index].detach().cpu(),
                    reconstructed=reconstructed_rgb[preview_index].detach().cpu()
                    if reconstructed_rgb.numel() > 0
                    else content_batch["image"][preview_index].detach().cpu(),
                )

        if (iteration + 1) % int(training_cfg.get("checkpoint_every", 10000)) == 0 or (iteration + 1) == iterations:
            checkpoint_payload = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "skipped_batches": skipped_batches,
            }
            save_checkpoint(checkpoint_dir / "last.pt", checkpoint_payload)

        if (iteration + 1) % int(training_cfg.get("validation_every", 10000)) == 0 or (iteration + 1) == iterations:
            sanity = evaluate_fivek_sanity(
                model,
                dataloaders["fivek_validation"],
                device=device,
                enable_lpips=bool(validation_cfg.get("enable_lpips", True)),
                render_dir=image_dir / f"fivek_sanity_{iteration + 1:08d}",
                render_limit=int(training_cfg.get("validation_render_limit", 4)),
            )
            val_summary = {"iteration": iteration + 1, **sanity["summary"]}
            append_jsonl(log_dir / "validation_metrics.jsonl", val_summary)
            if float(sanity["summary"].get("ssim", float("-inf"))) > best_val_ssim:
                best_val_ssim = float(sanity["summary"].get("ssim", float("-inf")))
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "best_val_ssim": best_val_ssim,
                        "skipped_batches": skipped_batches,
                    },
                )

    summary = {
        "output_dir": str(output_root),
        "checkpoint_dir": str(checkpoint_dir),
        "last_iteration": iterations,
        "best_val_ssim": best_val_ssim,
        "skipped_batches": skipped_batches,
        "train_records": len(split_info["train_records"]),
        "val_records": len(split_info["val_records"]),
    }
    write_json(output_root / "training_summary.json", summary)
    return summary


def build_model_config(config: dict[str, Any]) -> CAPColorTransferConfig:
    base = CAPColorTransferConfig.from_photo_v1()
    model_overrides = dict(config.get("model", {}))
    vgg_path = model_overrides.pop("vgg_checkpoint_path", None)
    if vgg_path is None:
        resolved_vgg = PROJECT_ROOT / base.vgg_checkpoint_path
    else:
        resolved_vgg = resolve_config_path(vgg_path, config_path=config.get("config_path"))
    return replace(base, vgg_checkpoint_path=Path(resolved_vgg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the clean-room CAP-VST image-stage model on MIT-Adobe FiveK.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_fivek.yaml"),
        help="Path to a JSON-compatible YAML config file.",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
