# capvst_color

Clean-room phase-1 implementation of the photorealistic, image-only CAP-VST-style color transfer core.

Scope in this package:

- reversible image backbone,
- Cholesky-based feature transfer,
- typed config and outputs,
- cited VGG wrapper for later perceptual losses,
- image-stage training on MIT-Adobe FiveK,
- photorealistic benchmark and FiveK sanity evaluation,
- CPU tests for model, data, training, and evaluation paths.

Not included yet:

- segmentation,
- video,
- artistic mode,
- segmentation-aware benchmark parity.

## Entry Points

Train the photorealistic image-stage model:

`python -m capvst_color.train --config configs/train_fivek.yaml`

Run evaluation:

`python -m capvst_color.evaluate --config configs/eval_photoreal.yaml`

The bundled config files are JSON-compatible YAML so they can be parsed even in minimal local environments.

## Evaluation Note

The photorealistic benchmark runner is intentionally **mask-free** in this phase. It reports the same metric family as the paper-facing benchmark (`SSIM` and `Gram loss`), but it does not attempt semantic-mask parity yet.

## VGG Checkpoint Placeholder

The default config expects a normalized VGG-19 checkpoint at:

`checkpoints/vgg_normalised.pth`

This file is intentionally not included. The clean-room model core does not require it unless you explicitly instantiate `VGG19Encoder`.
