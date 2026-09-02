"""Torch datasets and loaders built on top of a manifest.

Two things here are not boilerplate:

* **Corrupt files do not kill a run.** Scouting archives are full of truncated
  JPEGs. An 8-hour training job must not die at hour 6 because one farmer's
  upload was cut short - it substitutes a neutral image, marks the sample
  invalid so the loss ignores it, and reports the count at the end.
* **Missing severity is a first-class state.** Almost no public dataset labels
  severity. Those samples carry ``-100`` and are masked out of the severity
  loss rather than being forced into a fake "moderate".
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..taxonomy import CATEGORIES, SEVERITY_LEVELS, Taxonomy, load_taxonomy
from .manifest import Record, read_manifest
from .transforms import AugmentationConfig, build_eval_transform, build_train_transform

# A truncated JPEG should degrade, not explode.
ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class Batch:
    """Typed view over a collated batch (documentation more than machinery)."""

    image: torch.Tensor          # (B, 3, H, W)
    label: torch.Tensor          # (B,)   fine-grained class index
    category: torch.Tensor       # (B,)   coarse group index
    severity: torch.Tensor       # (B,)   ordinal severity or IGNORE_INDEX
    valid: torch.Tensor          # (B,)   0 where the image failed to load


class CropDiagnosisDataset(Dataset):
    """Image -> (class, category, severity) for the multi-head model."""

    def __init__(
        self,
        records: Sequence[Record],
        taxonomy: Taxonomy,
        transform: Callable | None = None,
        image_size: int = 224,
    ):
        self.records = list(records)
        self.taxonomy = taxonomy
        self.transform = transform or build_eval_transform(image_size)
        self.image_size = image_size
        self._failed: Counter[str] = Counter()

        unknown = {r.class_id for r in self.records} - set(taxonomy.class_ids)
        if unknown:
            raise KeyError(
                f"manifest contains classes not in the taxonomy: {sorted(unknown)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def failed_images(self) -> dict[str, int]:
        return dict(self._failed)

    def class_distribution(self) -> Counter:
        return Counter(r.class_id for r in self.records)

    def _load(self, path: str) -> tuple[Image.Image, bool]:
        try:
            with Image.open(path) as im:
                return im.convert("RGB"), True
        except Exception as exc:  # noqa: BLE001 - any decode failure is the same to us
            if path not in self._failed:
                LOGGER.warning("unreadable image %s (%s) - substituting", path, exc)
            self._failed[path] += 1
            return Image.new("RGB", (self.image_size, self.image_size), (124, 124, 124)), False

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img, ok = self._load(rec.path)
        tensor = self.transform(img)

        severity = (
            SEVERITY_LEVELS.index(rec.severity)
            if rec.severity in SEVERITY_LEVELS
            else IGNORE_INDEX
        )
        label = self.taxonomy.index_of(rec.class_id)
        return {
            "image": tensor,
            "label": label,
            "category": CATEGORIES.index(self.taxonomy[rec.class_id].category),
            "severity": severity,
            "valid": int(ok),
            "path": rec.path,
        }


def collate(samples: list[dict]) -> dict:
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "label": torch.tensor([s["label"] for s in samples], dtype=torch.long),
        "category": torch.tensor([s["category"] for s in samples], dtype=torch.long),
        "severity": torch.tensor([s["severity"] for s in samples], dtype=torch.long),
        "valid": torch.tensor([s["valid"] for s in samples], dtype=torch.long),
        "path": [s["path"] for s in samples],
    }


def make_sampler(records: Sequence[Record], taxonomy: Taxonomy, beta: float = 0.999):
    """Balanced sampling for a long-tailed archive.

    A district archive typically holds thousands of healthy frames and a few
    dozen of the pest that actually matters. Without rebalancing, the model
    learns to answer "healthy" and scores well on accuracy while being useless.
    """
    from .manifest import class_weights

    weights = class_weights(records, taxonomy.class_ids, beta=beta)
    per_sample = [weights[taxonomy.index_of(r.class_id)] for r in records]
    return WeightedRandomSampler(
        weights=torch.tensor(per_sample, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
    )


def build_dataloaders(
    manifest_path: str | Path,
    taxonomy: Taxonomy | None = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    aug: AugmentationConfig | None = None,
    balanced_sampling: bool = True,
    pin_memory: bool | None = None,
) -> tuple[dict[str, DataLoader], Taxonomy]:
    """Build train/val/test loaders and the taxonomy actually present in the data.

    The returned taxonomy is restricted to the classes in the manifest, in
    taxonomy order. Training a 70-way head when the district only has 12 classes
    wastes capacity and invents confusions that cannot happen.
    """
    full = taxonomy or load_taxonomy()
    all_records = read_manifest(manifest_path)
    if not all_records:
        raise ValueError(f"manifest {manifest_path} is empty")

    present = [c for c in full.class_ids if any(r.class_id == c for r in all_records)]
    tax = full.subset(present)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        records = [r for r in all_records if r.split == split]
        if not records:
            continue
        is_train = split == "train"
        transform = (
            build_train_transform(aug or AugmentationConfig(image_size=image_size))
            if is_train
            else build_eval_transform(image_size)
        )
        ds = CropDiagnosisDataset(records, tax, transform=transform, image_size=image_size)
        sampler = make_sampler(records, tax) if (is_train and balanced_sampling) else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(is_train and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train and len(ds) > batch_size,
            collate_fn=collate,
            persistent_workers=num_workers > 0,
        )
    if "train" not in loaders:
        raise ValueError("manifest has no rows with split=train")
    return loaders, tax


def build_eval_loader(
    manifest_path: str | Path,
    taxonomy: Taxonomy,
    split: str = "train",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader | None:
    """Un-augmented, un-shuffled loader over one split.

    Needed wherever the *representation* is being measured rather than trained:
    fitting the OOD detector on training embeddings, exporting calibration
    images, mining hard examples. Using the augmented training loader for any
    of those measures the augmentation, not the data.
    """
    records = [r for r in read_manifest(manifest_path) if r.split == split]
    if not records:
        return None
    ds = CropDiagnosisDataset(
        records, taxonomy, transform=build_eval_transform(image_size), image_size=image_size
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, drop_last=False,
    )


def mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.2,
    cutmix_prob: float = 0.5,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mixed images and *soft* targets.

    Useful on small district datasets, where the model otherwise memorises the
    handful of images of a rare pest. Returns soft targets so the caller uses a
    soft cross-entropy for the label head only - category and severity heads
    keep their hard targets on the dominant sample.
    """
    if alpha <= 0:
        onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        return images, _smooth(onehot, label_smoothing)

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(images.size(0), device=images.device)

    if torch.rand(1).item() < cutmix_prob:
        _, _, h, w = images.shape
        cut = (1.0 - lam) ** 0.5
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        images = images.clone()
        images[:, :, y0:y1, x0:x1] = images[perm][:, :, y0:y1, x0:x1]
        lam = 1.0 - ((y1 - y0) * (x1 - x0) / (h * w))
    else:
        images = lam * images + (1.0 - lam) * images[perm]

    onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)
    targets = lam * onehot + (1.0 - lam) * onehot[perm]
    return images, _smooth(targets, label_smoothing)


def _smooth(targets: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return targets
    n = targets.size(1)
    return targets * (1.0 - eps) + eps / n


__all__ = [
    "CropDiagnosisDataset",
    "build_eval_loader",
    "Batch",
    "IGNORE_INDEX",
    "collate",
    "build_dataloaders",
    "make_sampler",
    "mixup_cutmix",
]
