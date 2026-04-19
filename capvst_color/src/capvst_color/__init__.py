from .config import CAPColorTransferConfig
from .evaluate import run_photoreal_evaluation
from .model import CAPColorTransferModel, CAPColorTransferOutput
from .transform import CholeskyWCT
from .vgg import VGG19Encoder

__all__ = [
    "CAPColorTransferConfig",
    "CAPColorTransferModel",
    "CAPColorTransferOutput",
    "CholeskyWCT",
    "run_photoreal_evaluation",
    "run_training",
    "VGG19Encoder",
]

def run_training(*args, **kwargs):
    from .train import run_training as _run_training
    return _run_training(*args, **kwargs)