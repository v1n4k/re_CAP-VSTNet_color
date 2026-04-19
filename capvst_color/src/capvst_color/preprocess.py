from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = validate_image_tensor(tensor, name="tensor")
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"tensor_to_pil expects a single image, got batch shape {tuple(tensor.shape)}.")
        tensor = tensor[0]
    if tensor.shape[0] != 3:
        raise ValueError(f"tensor_to_pil expects 3 channels, got {tensor.shape[0]}.")
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_rgb_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def validate_image_tensor(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape (3, H, W) or (B, 3, H, W), got {tuple(tensor.shape)}.")
    if tensor.ndim == 3 and tensor.shape[0] != 3:
        raise ValueError(f"{name} must have 3 channels, got {tensor.shape[0]}.")
    if tensor.ndim == 4 and tensor.shape[1] != 3:
        raise ValueError(f"{name} must have 3 channels, got {tensor.shape[1]}.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating-point, got dtype {tensor.dtype}.")
    return tensor


def resize_short_edge(image: Image.Image, short_edge: int) -> Image.Image:
    if short_edge <= 0:
        raise ValueError(f"short_edge must be positive, got {short_edge}.")
    width, height = image.size
    if min(width, height) == short_edge:
        return image
    scale = short_edge / float(min(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


def resize_long_edge_max(image: Image.Image, max_size: int | None) -> Image.Image:
    if max_size is None:
        return image
    if max_size <= 0:
        raise ValueError(f"max_size must be positive when provided, got {max_size}.")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


def snap_to_downscale_factor(image: Image.Image, factor: int) -> Image.Image:
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}.")
    width, height = image.size
    snapped_width = max(factor, width - (width % factor))
    snapped_height = max(factor, height - (height % factor))
    if (snapped_width, snapped_height) == (width, height):
        return image
    return image.resize((snapped_width, snapped_height), Image.BICUBIC)


def random_crop(image: Image.Image, crop_size: int, *, rng: random.Random) -> Image.Image:
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Cannot random crop {crop_size} from image of size {(width, height)}."
        )
    if width == crop_size and height == crop_size:
        return image
    left = rng.randint(0, width - crop_size)
    top = rng.randint(0, height - crop_size)
    return image.crop((left, top, left + crop_size, top + crop_size))


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Cannot center crop {crop_size} from image of size {(width, height)}.")
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def center_crop_to_size(image: Image.Image, width: int, height: int) -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {(width, height)}.")
    image_width, image_height = image.size
    if image_width < width or image_height < height:
        raise ValueError(
            f"Cannot center crop target size {(width, height)} from image of size {image.size}."
        )
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def paired_center_crop(
    content_image: Image.Image,
    gt_image: Image.Image,
    crop_size: int,
) -> tuple[Image.Image, Image.Image]:
    common_width = min(content_image.width, gt_image.width)
    common_height = min(content_image.height, gt_image.height)
    aligned_content = center_crop_to_size(content_image, common_width, common_height)
    aligned_gt = center_crop_to_size(gt_image, common_width, common_height)
    return center_crop(aligned_content, crop_size), center_crop(aligned_gt, crop_size)


def resize_pair_like_content(
    content_image: Image.Image,
    gt_image: Image.Image,
    *,
    max_size: int | None,
    downscale_factor: int,
) -> tuple[Image.Image, Image.Image]:
    if content_image.size != gt_image.size:
        raise ValueError(
            f"content_image and gt_image must share the same size, got {content_image.size} and {gt_image.size}."
        )
    resized_content = resize_long_edge_max(content_image, max_size)
    resized_gt = resize_long_edge_max(gt_image, max_size)
    return (
        snap_to_downscale_factor(resized_content, downscale_factor),
        snap_to_downscale_factor(resized_gt, downscale_factor),
    )


def resize_image_for_benchmark(
    image: Image.Image,
    *,
    max_size: int | None,
    downscale_factor: int,
) -> Image.Image:
    return snap_to_downscale_factor(resize_long_edge_max(image, max_size), downscale_factor)
