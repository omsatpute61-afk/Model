"""The CropGuard multi-head classifier.

One shared trunk, three heads:

``label``     the fine-grained diagnosis (70-way in the bundled taxonomy)
``category``  the coarse group - healthy / disease / pest / deficiency / abiotic
``severity``  a 4-level ordinal estimate

Why not just the fine head? Because of what happens when it is wrong. A model
that confuses *early blight* with *target spot* has still told the farmer
"fungal leaf spot, protectant spray" and is useful. A model that confuses a
pest with a nutrient deficiency sends them to buy the wrong input. The coarse
head is trained on an easier problem, is far more reliable, and lets the
runtime fall back to category-level advice when the fine head is uncertain
instead of abstaining completely.

Severity is masked wherever the dataset has no severity label, so it costs
nothing on datasets that lack it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxonomy import CATEGORIES, SEVERITY_LEVELS
from .backbones import build_backbone

IGNORE_INDEX = -100


class ModelOutput(NamedTuple):
    """NamedTuple (not dict) so ONNX and TorchScript export cleanly."""

    label_logits: torch.Tensor
    category_logits: torch.Tensor
    severity_logits: torch.Tensor
    embedding: torch.Tensor


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v3_small"
    num_classes: int = 70
    num_categories: int = len(CATEGORIES)
    num_severity: int = len(SEVERITY_LEVELS)
    embedding_dim: int = 128
    dropout: float = 0.2
    pretrained: bool = True
    image_size: int = 224


class CropGuardNet(nn.Module):
    """Shared-trunk classifier sized for a Raspberry-Pi-class device."""

    def __init__(self, cfg: ModelConfig | None = None, **kwargs):
        super().__init__()
        cfg = cfg or ModelConfig(**kwargs)
        self.cfg = cfg

        self.trunk, self.backbone_info = build_backbone(cfg.backbone, cfg.pretrained)
        feat = self.backbone_info.feature_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        # BatchNorm before each activation is not decoration. Without it the
        # neck's Hardswish units can all saturate early in training and the
        # embedding collapses to a constant vector - the label head then
        # predicts one class for everything, and the OOD detector fitted on
        # those embeddings gets a threshold of ~0 and rejects every photo.
        self.neck = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat, cfg.embedding_dim * 2, bias=False),
            nn.BatchNorm1d(cfg.embedding_dim * 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim, bias=False),
            nn.BatchNorm1d(cfg.embedding_dim),
            nn.Hardswish(inplace=True),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.label_head = nn.Linear(cfg.embedding_dim, cfg.num_classes)
        self.category_head = nn.Linear(cfg.embedding_dim, cfg.num_categories)
        self.severity_head = nn.Linear(cfg.embedding_dim, cfg.num_severity)

        for head in (self.label_head, self.category_head, self.severity_head):
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=0.01)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.pool(self.trunk(x)))

    def forward(self, x: torch.Tensor) -> ModelOutput:
        emb = self.features(x)
        h = self.dropout(emb)
        return ModelOutput(
            label_logits=self.label_head(h),
            category_logits=self.category_head(h),
            severity_logits=self.severity_head(h),
            embedding=emb,
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        """Calibrated probabilities, for eval and for the torch inference path."""
        self.eval()
        out = self(x)
        return {
            "label": F.softmax(out.label_logits / temperature, dim=1),
            "category": F.softmax(out.category_logits / temperature, dim=1),
            "severity": F.softmax(out.severity_logits, dim=1),
            "embedding": out.embedding,
        }

    def param_groups(self, lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Discriminative learning rates.

        Pretrained features already encode texture and colour; hitting them
        with the head's learning rate destroys that in the first few hundred
        steps. The trunk moves at a fraction of the head's rate.
        """
        trunk_params = [p for p in self.trunk.parameters() if p.requires_grad]
        head_params = [
            p
            for m in (self.neck, self.label_head, self.category_head, self.severity_head)
            for p in m.parameters()
            if p.requires_grad
        ]
        groups = [{"params": head_params, "lr": lr, "name": "head"}]
        if trunk_params:
            groups.append(
                {"params": trunk_params, "lr": lr * backbone_lr_scale, "name": "trunk"}
            )
        return groups

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )


class ExportWrapper(nn.Module):
    """Inference-only view: calibrated probabilities plus the embedding.

    Exporting this rather than the raw model means the edge runtime never has
    to reimplement softmax-with-temperature and can never get it wrong.

    The embedding is exported too, and it is not dead weight: it is what the
    out-of-distribution check in ``cropguard.ood`` scores. Softmax confidence
    alone cannot reject a photo of the sky - the network happily reports 0.99
    for some class - so the runtime needs a signal computed in feature space.
    """

    def __init__(self, model: CropGuardNet, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return (
            F.softmax(out.label_logits / self.temperature, dim=1),
            F.softmax(out.category_logits / self.temperature, dim=1),
            F.softmax(out.severity_logits, dim=1),
            out.embedding,
        )


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Down-weight the easy majority so rare pests still drive gradients.

    A district archive is dominated by healthy leaves and a handful of common
    diseases. Plain cross-entropy spends its capacity there; focal loss keeps
    pushing on the tail classes that the farmer actually needs warned about.
    """
    ce = F.cross_entropy(
        logits, targets, weight=weight, reduction="none", label_smoothing=label_smoothing
    )
    pt = torch.exp(-ce.clamp(max=20.0))
    return ((1.0 - pt) ** gamma * ce).mean()


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against soft targets (mixup / cutmix)."""
    return torch.sum(-targets * F.log_softmax(logits, dim=1), dim=1).mean()


class MultiHeadLoss(nn.Module):
    """Weighted sum over the three heads, with masking for missing labels."""

    def __init__(
        self,
        label_weight: float = 1.0,
        category_weight: float = 0.3,
        severity_weight: float = 0.2,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.label_weight = label_weight
        self.category_weight = category_weight
        self.severity_weight = severity_weight
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.empty(0)
        )

    def _weights(self) -> torch.Tensor | None:
        w = self.class_weights
        return w if w.numel() else None

    def forward(
        self,
        out: ModelOutput,
        label: torch.Tensor,
        category: torch.Tensor,
        severity: torch.Tensor,
        soft_label: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, float] = {}

        if valid is not None and valid.numel():
            keep = valid.bool()
            if not keep.any():  # whole batch failed to decode - skip it
                zero = out.label_logits.sum() * 0.0
                return zero, {"label": 0.0, "category": 0.0, "severity": 0.0, "total": 0.0}
            if not keep.all():
                out = ModelOutput(
                    out.label_logits[keep], out.category_logits[keep],
                    out.severity_logits[keep], out.embedding[keep],
                )
                label, category, severity = label[keep], category[keep], severity[keep]
                if soft_label is not None:
                    soft_label = soft_label[keep]

        if soft_label is not None:
            label_loss = soft_cross_entropy(out.label_logits, soft_label)
        elif self.use_focal:
            label_loss = focal_loss(
                out.label_logits, label, self.focal_gamma, self._weights(), self.label_smoothing
            )
        else:
            label_loss = F.cross_entropy(
                out.label_logits, label,
                weight=self._weights(), label_smoothing=self.label_smoothing,
            )
        parts["label"] = float(label_loss.detach())

        cat_loss = F.cross_entropy(out.category_logits, category, label_smoothing=self.label_smoothing)
        parts["category"] = float(cat_loss.detach())

        # Severity is unlabelled in most public data - mask instead of inventing.
        sev_mask = severity != IGNORE_INDEX
        if sev_mask.any():
            sev_loss = F.cross_entropy(out.severity_logits[sev_mask], severity[sev_mask])
        else:
            sev_loss = out.severity_logits.sum() * 0.0
        parts["severity"] = float(sev_loss.detach())

        total = (
            self.label_weight * label_loss
            + self.category_weight * cat_loss
            + self.severity_weight * sev_loss
        )
        parts["total"] = float(total.detach())
        return total, parts


class ModelEMA:
    """Exponential moving average of weights.

    Field datasets are small and noisy; the last-epoch weights bounce. The EMA
    copy is consistently 1-2 points better and is what actually gets shipped.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, tau: float = 2000.0):
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = max(1.0, float(tau))
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Ramp the decay up from 0 so early steps are not dominated by the
        # random initialisation. ``tau`` sets how fast. The usual constant of
        # 2000 assumes a long run: on a district dataset of a few hundred
        # steps it never leaves the ramp, and the "EMA" is just a copy of the
        # live weights doing nothing. Callers should scale tau to the run
        # length - cropguard.train sets it from the step budget.
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.tau)).item())
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def create_model(cfg: ModelConfig | None = None, **kwargs) -> CropGuardNet:
    return CropGuardNet(cfg, **kwargs)


__all__ = [
    "CropGuardNet",
    "ModelConfig",
    "ModelOutput",
    "ExportWrapper",
    "MultiHeadLoss",
    "ModelEMA",
    "focal_loss",
    "soft_cross_entropy",
    "create_model",
]
