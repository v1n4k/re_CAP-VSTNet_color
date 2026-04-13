# Checkpoints

Place the normalized VGG-19 weights for later perceptual-loss work at:

`checkpoints/vgg_normalised.pth`

This phase does not load the checkpoint unless `VGG19Encoder.from_checkpoint(...)` is called.
