"""Run configuration: dataclasses with YAML/CLI overrides.

Every field that affects the trained model ends up in the model card, so a
checkpoint can always be traced back to the settings that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    manifest: str = "artifacts/data/manifest.csv"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    balanced_sampling: bool = True
    aug_strength: float = 1.0
    taxonomy: str | None = None          # None -> bundled taxonomy


@dataclass
class ModelSettings:
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    dropout: float = 0.2
    embedding_dim: int = 128


@dataclass
class OptimConfig:
    epochs: int = 30
    lr: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr_scale: float = 0.02
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    use_focal: bool = False
    focal_gamma: float = 2.0
    category_weight: float = 0.3
    severity_weight: float = 0.2
    life_stage_weight: float = 0.15
    mixup_alpha: float = 0.0
    ema_decay: float = 0.999
    use_ema: bool = True
    freeze_epochs: int = 1               # head-only warm start
    unfreeze_blocks: int = 0             # 0 = thaw the whole trunk after warm start
    early_stopping_patience: int = 8
    amp: bool = True


@dataclass
class PolicyConfig:
    max_selective_error: float = 0.10
    min_coverage: float = 0.50
    min_threshold: float = 0.30
    calibrate: bool = True
    fit_ood: bool = True
    ood_percentile: float = 97.5


@dataclass
class TrainConfig:
    run_name: str = "cropguard"
    output_dir: str = "artifacts/runs"
    seed: int = 1337
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelSettings = field(default_factory=ModelSettings)
    optim: OptimConfig = field(default_factory=OptimConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return _build(cls, d or {})

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pyyaml is required to read YAML configs") from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def apply_overrides(self, overrides: list[str] | None) -> "TrainConfig":
        """Apply ``a.b=c`` style CLI overrides (``--set optim.lr=1e-3``)."""
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override {item!r} must look like path.key=value")
            path, raw = item.split("=", 1)
            _set_path(self, path.split("."), _coerce(raw))
        return self


def _build(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        elif isinstance(value, dict) and f.name in _NESTED:
            kwargs[f.name] = _build(_NESTED[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_NESTED = {
    "data": DataConfig,
    "model": ModelSettings,
    "optim": OptimConfig,
    "policy": PolicyConfig,
}


def _set_path(obj: Any, path: list[str], value: Any) -> None:
    for key in path[:-1]:
        if not hasattr(obj, key):
            raise KeyError(f"unknown config section {key!r}")
        obj = getattr(obj, key)
    if not hasattr(obj, path[-1]):
        raise KeyError(f"unknown config key {'.'.join(path)!r}")
    setattr(obj, path[-1], value)


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


__all__ = ["TrainConfig", "DataConfig", "ModelSettings", "OptimConfig", "PolicyConfig"]
