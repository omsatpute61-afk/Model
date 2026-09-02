"""Model definitions: edge-sized backbones and the multi-head detector."""

from .backbones import AVAILABLE_BACKBONES, BackboneInfo, build_backbone
from .detector import (
    CropGuardNet,
    ExportWrapper,
    ModelConfig,
    ModelEMA,
    ModelOutput,
    MultiHeadLoss,
    create_model,
)

__all__ = [
    "CropGuardNet",
    "ModelConfig",
    "ModelOutput",
    "ExportWrapper",
    "MultiHeadLoss",
    "ModelEMA",
    "create_model",
    "build_backbone",
    "BackboneInfo",
    "AVAILABLE_BACKBONES",
]
