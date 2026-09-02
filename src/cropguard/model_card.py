"""The contract between a trained model and everything that consumes it.

A pest/disease model is dangerous when its metadata drifts: resize a field
image to 224 when the model was trained at 192, or pair logits with a
reordered label list, and the system confidently recommends the wrong spray.

So every artefact - checkpoint, ONNX file, TorchScript bundle - is accompanied
by a card that pins the label order, the exact preprocessing, the calibration
temperature and the decision thresholds. The edge runtime refuses to run
without one. Pure standard library: the field device parses this, not torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CARD_FILENAME = "model_card.json"
CARD_SCHEMA_VERSION = 1


@dataclass
class PreprocessSpec:
    """Exactly what the pixels must go through before the first conv."""

    image_size: int = 224
    crop_pct: float = 0.90
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    layout: str = "NCHW"
    colour_space: str = "RGB"
    scale: float = 1.0 / 255.0

    @property
    def resize_to(self) -> int:
        return int(round(self.image_size / self.crop_pct))


@dataclass
class DecisionPolicy:
    """How a probability vector becomes an action.

    ``min_confidence`` is not a hyperparameter to leave at 0.5. It is chosen on
    the validation set (see ``cropguard.evaluate.select_threshold``) so that the
    rate of confident-but-wrong diagnoses stays under an operating budget - a
    wrong spray costs a smallholder real money.
    """

    min_confidence: float = 0.55
    temperature: float = 1.0
    margin: float = 0.0          # required gap between top-1 and top-2
    abstain_label: str = "unknown"
    topk: int = 3


@dataclass
class ModelCard:
    name: str = "cropguard-pest-disease"
    version: str = "0.1.0"
    schema_version: int = CARD_SCHEMA_VERSION
    backbone: str = "mobilenet_v3_small"
    class_ids: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    severity_levels: list[str] = field(default_factory=list)
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    pretrained: bool = True
    trained_on: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class_thresholds: dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    git_commit: str = ""
    notes: str = ""
    limitations: list[str] = field(
        default_factory=lambda: [
            "Diagnoses only the classes listed in class_ids; anything else should "
            "fall below min_confidence and be reported as 'unknown', not forced "
            "into the nearest class.",
            "Trained on leaf-level close-ups. Whole-field or drone imagery must be "
            "tiled before inference.",
            "Visual symptoms overlap between causes (nutrient deficiency vs early "
            "viral infection). A high-cost action should be confirmed by an "
            "extension officer.",
            "Severity is a coarse 4-level estimate from a single frame; it is not "
            "a substitute for a scouting count against the economic threshold.",
        ]
    )

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["preprocess"]["mean"] = list(self.preprocess.mean)
        d["preprocess"]["std"] = list(self.preprocess.std)
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / CARD_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCard":
        d = dict(d)
        pre = d.pop("preprocess", {}) or {}
        pol = d.pop("policy", {}) or {}
        known = {f for f in cls.__dataclass_fields__}
        card = cls(**{k: v for k, v in d.items() if k in known})
        card.preprocess = PreprocessSpec(
            **{
                **{k: v for k, v in pre.items() if k in PreprocessSpec.__dataclass_fields__},
                **({"mean": tuple(pre["mean"])} if "mean" in pre else {}),
                **({"std": tuple(pre["std"])} if "std" in pre else {}),
            }
        )
        card.policy = DecisionPolicy(
            **{k: v for k, v in pol.items() if k in DecisionPolicy.__dataclass_fields__}
        )
        return card

    @classmethod
    def load(cls, path: str | Path) -> "ModelCard":
        path = Path(path)
        if path.is_dir():
            path = path / CARD_FILENAME
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- validation ------------------------------------------------------
    def validate(self, num_outputs: int | None = None) -> None:
        """Fail loudly on the mismatches that cause silent misdiagnosis."""
        if not self.class_ids:
            raise ValueError("model card has no class_ids - labels would be guesswork")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("model card class_ids contain duplicates")
        if num_outputs is not None and num_outputs != len(self.class_ids):
            raise ValueError(
                f"model outputs {num_outputs} logits but the card lists "
                f"{len(self.class_ids)} classes - refusing to guess the mapping"
            )
        if not 0.0 < self.policy.temperature <= 20.0:
            raise ValueError(f"implausible calibration temperature {self.policy.temperature}")
        if not 0.0 <= self.policy.min_confidence < 1.0:
            raise ValueError(f"min_confidence must be in [0, 1): {self.policy.min_confidence}")
        if self.preprocess.image_size <= 0:
            raise ValueError("preprocess.image_size must be positive")


__all__ = ["ModelCard", "PreprocessSpec", "DecisionPolicy", "CARD_FILENAME"]
