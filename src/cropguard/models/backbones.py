"""Edge-sized backbones.

Constraints that drove the choices here: the model runs on a solar-powered
field box (Raspberry Pi 4 / Jetson Nano class, sometimes an Android phone),
must answer in well under a second on CPU, and must survive INT8 quantisation
without falling apart.

That rules out ViTs and the larger ResNets and leaves the depthwise-separable
CNN family. ``mobilenet_v3_small`` is the default: ~2.5 M parameters, ~2.5 MB
after INT8, and comfortably real-time on a Pi 4 core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

#: name -> (torchvision factory, weights enum name, feature dim)
_BACKBONES: dict[str, tuple[str, str | None, int]] = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights", 576),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", 960),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", 1280),
    "shufflenet_v2_x1_0": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights", 1024),
    "resnet18": ("resnet18", "ResNet18_Weights", 512),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights", 512),
}

AVAILABLE_BACKBONES = tuple(_BACKBONES)


@dataclass
class BackboneInfo:
    name: str
    feature_dim: int
    pretrained: bool
    note: str = ""


def _strip_classifier(name: str, model: nn.Module) -> nn.Module:
    """Return the feature trunk, ending just before global pooling."""
    if name.startswith(("mobilenet", "efficientnet")):
        return model.features
    if name.startswith("shufflenet"):
        return nn.Sequential(
            model.conv1, model.maxpool, model.stage2, model.stage3, model.stage4, model.conv5
        )
    if name.startswith("resnet"):
        return nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
    if name.startswith("squeezenet"):
        return model.features
    raise KeyError(f"no trunk rule for backbone {name!r}")


def build_backbone(
    name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    width_mult: float | None = None,
) -> tuple[nn.Module, BackboneInfo]:
    """Create a feature trunk.

    ``pretrained`` downloads ImageNet weights. Transfer learning is not
    optional in practice: with a few thousand field photos per class, training
    from scratch loses double-digit accuracy. But a field/CI machine may be
    offline, so a failed download degrades to random init with a loud warning
    rather than crashing the run.
    """
    from torchvision import models as tvm

    if name not in _BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; available: {AVAILABLE_BACKBONES}")
    factory_name, weights_enum, feature_dim = _BACKBONES[name]
    factory: Callable[..., nn.Module] = getattr(tvm, factory_name)

    note = ""
    weights = None
    if pretrained and weights_enum is not None:
        try:
            weights = getattr(tvm, weights_enum).DEFAULT
        except Exception as exc:  # noqa: BLE001
            note = f"weights enum unavailable ({exc})"

    try:
        model = factory(weights=weights)
        got_pretrained = weights is not None
    except Exception as exc:  # noqa: BLE001 - offline, cache miss, hash mismatch
        import warnings

        warnings.warn(
            f"could not load pretrained weights for {name} ({exc}). "
            "Falling back to random initialisation - expect materially lower "
            "accuracy. Pre-download the weights on a connected machine and set "
            "TORCH_HOME to reuse them offline.",
            RuntimeWarning,
            stacklevel=2,
        )
        model = factory(weights=None)
        got_pretrained = False
        note = f"pretrained download failed: {exc}"

    trunk = _strip_classifier(name, model)
    return trunk, BackboneInfo(name=name, feature_dim=feature_dim, pretrained=got_pretrained, note=note)


def freeze_backbone(trunk: nn.Module, freeze: bool = True) -> None:
    for p in trunk.parameters():
        p.requires_grad = not freeze


def unfreeze_last_blocks(trunk: nn.Module, n_blocks: int) -> int:
    """Progressive unfreezing: thaw only the last ``n_blocks`` stages.

    Early layers are edges and colour blobs and transfer perfectly; late layers
    encode ImageNet's object semantics and need to be re-learned for lesions.
    Thawing only the tail is faster and, on small datasets, more accurate.
    """
    children = [m for m in trunk.children()]
    if not children:
        return 0
    n_blocks = max(0, min(n_blocks, len(children)))
    for module in children[: len(children) - n_blocks]:
        for p in module.parameters():
            p.requires_grad = False
    for module in children[len(children) - n_blocks :]:
        for p in module.parameters():
            p.requires_grad = True
    return n_blocks


__all__ = ["build_backbone", "BackboneInfo", "AVAILABLE_BACKBONES", "freeze_backbone", "unfreeze_last_blocks"]
