from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


WORKSPACE_ROOT = Path("/home/jhua0805/5405")
CAPVSTNET_ROOT = WORKSPACE_ROOT / "CAP-VSTNet"
RE_CAPVST_COLOR_SRC = WORKSPACE_ROOT / "re_CAP-VSTNet_color" / "capvst_color" / "src"
RE_CAPVST_COLOR_ROOT = WORKSPACE_ROOT / "re_CAP-VSTNet_color" / "capvst_color"
PST50_ROOT = WORKSPACE_ROOT / "Reproduce_SA-LUT" / "data" / "pst50" / "paired"
DEFAULT_EVAL_OUTPUT_ROOT = RE_CAPVST_COLOR_ROOT / "outputs" / "eval" / "original"
ORIGINAL_CHECKPOINT = CAPVSTNET_ROOT / "checkpoints" / "Photo_Image.pt"
VGG_CHECKPOINT = WORKSPACE_ROOT / "re_CAP-VSTNet_color" / "capvst_color" / "checkpoints" / "vgg_normalised.pth"
MAX_SIZE = 1280
DEFAULT_DPST_EXCLUDED_KEYS = ("23",)

sys.path.insert(0, str(RE_CAPVST_COLOR_SRC))
sys.path.insert(0, str(CAPVSTNET_ROOT))

from capvst_color.metrics import (  # noqa: E402
    LPIPSMetric,
    align_bchw_to_reference,
    compute_hcorr,
    compute_psnr,
    compute_ssim,
    summarize_numeric_rows,
)
from models.RevResNet import RevResNet  # noqa: E402
from models.cWCT import cWCT  # noqa: E402
from models.VGG import build_vgg  # noqa: E402
from utils.utils import img_resize  # noqa: E402


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_image_tensor(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim not in (3, 4):
        raise ValueError(f"{name} must be 3D or 4D, got shape {tuple(tensor.shape)}.")
    if tensor.ndim == 4 and tensor.shape[1] != 3:
        raise ValueError(f"{name} must have 3 channels, got shape {tuple(tensor.shape)}.")
    if tensor.ndim == 3 and tensor.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3, H, W), got {tuple(tensor.shape)}.")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.clamp(0.0, 1.0)


class GramMetric(torch.nn.Module):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        vgg = build_vgg()
        vgg.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        layers = list(vgg.children())
        self.encoders = torch.nn.ModuleList(
            [
                torch.nn.Sequential(*layers[:4]),
                torch.nn.Sequential(*layers[4:11]),
                torch.nn.Sequential(*layers[11:18]),
                torch.nn.Sequential(*layers[18:31]),
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def encode_with_intermediate(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        current = tensor
        for encoder in self.encoders:
            current = encoder(current)
            outputs.append(current)
        return outputs

    def forward(self, stylized_rgb: torch.Tensor, style_rgb: torch.Tensor) -> torch.Tensor:
        stylized_features = self.encode_with_intermediate(stylized_rgb)
        style_features = self.encode_with_intermediate(style_rgb)
        total = stylized_rgb.new_zeros(())
        for stylized_feature, style_feature in zip(stylized_features, style_features):
            total = total + F.mse_loss(gram_matrix(stylized_feature), gram_matrix(style_feature))
        return total


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feat.shape
    reshaped = feat.reshape(batch, channels, height * width)
    return torch.bmm(reshaped, reshaped.transpose(1, 2)) / float(channels * height * width)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = validate_image_tensor(tensor.detach().cpu(), name="tensor")
    if tensor.ndim == 4:
        tensor = tensor[0]
    return transforms.ToPILImage()(tensor)


def build_model(device: torch.device) -> tuple[torch.nn.Module, cWCT]:
    model = RevResNet(
        nBlocks=[10, 10, 10],
        nStrides=[1, 2, 2],
        nChannels=[16, 64, 256],
        in_channel=3,
        mult=4,
        hidden_dim=16,
        sp_steps=2,
    )
    checkpoint = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    transfer = cWCT()
    return model, transfer


def load_resized_tensor(path: Path, *, down_scale: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    image = img_resize(image, MAX_SIZE, down_scale=down_scale)
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image, tensor


@torch.inference_mode()
def stylize_pair(
    model: torch.nn.Module,
    transfer: cWCT,
    *,
    content_path: Path,
    style_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, Image.Image, Image.Image]:
    content_image, content_tensor = load_resized_tensor(content_path, down_scale=model.down_scale)
    style_image, style_tensor = load_resized_tensor(style_path, down_scale=model.down_scale)
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)

    content_latent = model(content_tensor, forward=True)
    style_latent = model(style_tensor, forward=True)
    stylized_latent = transfer.transfer(content_latent, style_latent)
    stylized = model(stylized_latent, forward=False)
    return stylized, content_image, style_image


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    ensure_dir(path.parent)
    headers = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize(rows: list[dict[str, float | str]]) -> dict[str, object]:
    summary = summarize_numeric_rows(rows)
    summary["protocol_note"] = "mask-free evaluation using original CAP-VSTNet inference"
    summary["uses_masks"] = False
    return summary


def build_index(directory: Path, prefix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if prefix and not stem.startswith(prefix):
            continue
        key = stem[len(prefix) :] if prefix else stem
        index[key] = path
    return index


def discover_records(
    *,
    content_dir: Path,
    style_dir: Path,
    gt_dir: Path,
    content_prefix: str,
    style_prefix: str,
    gt_prefix: str,
    exclude_keys: set[str],
) -> list[tuple[str, Path, Path, Path]]:
    content_index = build_index(content_dir, content_prefix)
    style_index = build_index(style_dir, style_prefix)
    gt_index = build_index(gt_dir, gt_prefix)
    keys = sorted((set(content_index) & set(style_index) & set(gt_index)) - exclude_keys)
    if not keys:
        raise ValueError("No matched benchmark files were found across content/style/gt.")
    return [(key, content_index[key], style_index[key], gt_index[key]) for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original CAP-VSTNet benchmark evaluation.")
    parser.add_argument("--content-dir", type=Path, default=PST50_ROOT / "content")
    parser.add_argument("--style-dir", type=Path, default=PST50_ROOT / "style")
    parser.add_argument("--gt-dir", type=Path, default=PST50_ROOT / "gt")
    parser.add_argument("--content-prefix", type=str, default="")
    parser.add_argument("--style-prefix", type=str, default="")
    parser.add_argument("--gt-prefix", type=str, default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--exclude-keys", nargs="*", default=None)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    if (
        args.content_prefix == "in"
        and args.style_prefix == "tar"
        and args.gt_prefix == "in"
        and args.content_dir.resolve() == args.gt_dir.resolve()
    ):
        return DEFAULT_EVAL_OUTPUT_ROOT / "dpst"
    if (
        args.content_dir.resolve() == (PST50_ROOT / "content").resolve()
        and args.style_dir.resolve() == (PST50_ROOT / "style").resolve()
        and args.gt_dir.resolve() == (PST50_ROOT / "gt").resolve()
    ):
        return DEFAULT_EVAL_OUTPUT_ROOT / "pst50"
    return DEFAULT_EVAL_OUTPUT_ROOT / "custom"


def main() -> None:
    args = parse_args()
    device = resolve_device()
    model, transfer = build_model(device)
    gram_metric = GramMetric(VGG_CHECKPOINT).to(device)
    lpips_metric = LPIPSMetric()

    output_root = ensure_dir(resolve_output_root(args))
    benchmark_root = ensure_dir(output_root / "photoreal_benchmark")
    image_dir = ensure_dir(benchmark_root / "stylized") if args.save_images else None

    rows: list[dict[str, float | str]] = []
    exclude_keys = set(args.exclude_keys or [])
    if (
        args.content_prefix == "in"
        and args.style_prefix == "tar"
        and args.gt_prefix == "in"
        and args.content_dir.resolve() == args.gt_dir.resolve()
    ):
        exclude_keys.update(DEFAULT_DPST_EXCLUDED_KEYS)
    records = discover_records(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        gt_dir=args.gt_dir,
        content_prefix=args.content_prefix,
        style_prefix=args.style_prefix,
        gt_prefix=args.gt_prefix,
        exclude_keys=exclude_keys,
    )

    for index, (basename, content_path, style_path, gt_path) in enumerate(records, start=1):
        stylized, content_image, style_image = stylize_pair(
            model,
            transfer,
            content_path=content_path,
            style_path=style_path,
            device=device,
        )
        gt_image = Image.open(gt_path).convert("RGB")
        stylized_pil = tensor_to_pil(stylized)
        stylized_tensor = transforms.ToTensor()(stylized_pil).unsqueeze(0).to(device)
        style_tensor = transforms.ToTensor()(style_image).unsqueeze(0).to(device)
        gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0).to(device)
        gt_for_metrics = align_bchw_to_reference(stylized_tensor, gt_tensor)
        style_for_metrics = align_bchw_to_reference(stylized_tensor, style_tensor)

        row: dict[str, float | str] = {
            "basename": basename,
            "gram_loss": float(gram_metric(stylized_tensor, style_for_metrics).detach().cpu().item()),
            "h_corr": compute_hcorr(stylized_tensor, style_for_metrics)[0],
            "psnr": compute_psnr(stylized_tensor, gt_for_metrics)[0],
            "ssim": compute_ssim(stylized_tensor, gt_for_metrics)[0],
        }
        if lpips_metric.available:
            row["lpips"] = lpips_metric(stylized_tensor, gt_for_metrics)[0]
        rows.append(row)

        if image_dir is not None:
            tensor_to_pil(stylized_tensor[0]).save(image_dir / f"{basename}.png")
        print(f"[{index:03d}/{len(records):03d}] {basename}")

    summary = summarize(rows)
    write_csv(benchmark_root / "metrics.csv", rows)
    write_json(benchmark_root / "summary.json", summary)
    write_json(output_root / "evaluation_summary.json", {"photoreal_benchmark": summary})
    print(json.dumps({"output_dir": str(output_root), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
