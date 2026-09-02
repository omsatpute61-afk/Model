"""Augmentation aimed at the gap between lab photos and field photos.

Models trained on PlantVillage-style images - one flat leaf, even studio light,
grey background - routinely lose 30-40 points of accuracy on a phone photo taken
at noon in a standing crop. The failure is almost never the disease features; it
is harsh shadow, blown highlights, motion blur, cluttered canopy background and
aggressive phone JPEG.

So the training pipeline simulates those conditions rather than only geometric
jitter. Each transform below corresponds to a specific field failure mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

#: ImageNet statistics - kept here as the single source of truth. The exported
#: edge runtime reads these from the model card so preprocessing can never
#: silently drift between training and the device.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomShadow:
    """Hard-edged shade from leaves above, or from the farmer's own body.

    A common field failure: half the leaf is in direct sun and half in deep
    shade, and a model that never saw it calls the shaded half a lesion.
    """

    def __init__(self, p: float = 0.35, strength: tuple[float, float] = (0.35, 0.75)):
        self.p, self.strength = p, strength

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        mask = Image.new("L", (w, h), 255)
        from PIL import ImageDraw

        d = ImageDraw.Draw(mask)
        factor = random.uniform(*self.strength)
        pts = [
            (random.uniform(-0.2, 1.2) * w, random.uniform(-0.2, 1.2) * h)
            for _ in range(random.randint(3, 5))
        ]
        d.polygon(pts, fill=int(255 * factor))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, w / 24)))
        dark = Image.new("RGB", (w, h), (0, 0, 0))
        return Image.composite(img, dark, mask)


class RandomSunGlare:
    """Blown highlight / specular reflection off a wet or waxy leaf."""

    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        from PIL import ImageDraw

        glare = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(glare)
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        r = random.uniform(w * 0.12, w * 0.45)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=random.randint(90, 190))
        glare = glare.filter(ImageFilter.GaussianBlur(radius=r / 2.2))
        white = Image.new("RGB", (w, h), (255, 255, 250))
        return Image.composite(white, img, glare)


class RandomMotionBlur:
    """A hand-held phone in wind: directional smear, not gaussian defocus."""

    def __init__(self, p: float = 0.20, max_kernel: int = 9):
        self.p, self.max_kernel = p, max_kernel

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice([x for x in range(3, self.max_kernel + 1, 2)])
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        arr = np.asarray(img, dtype=np.float32)
        out = np.empty_like(arr)
        pad = k // 2
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        for c in range(arr.shape[2]):
            # separable in practice (kernel is a single row or column)
            acc = np.zeros_like(arr[..., c])
            for i in range(k):
                for j in range(k):
                    if kernel[i, j]:
                        acc += kernel[i, j] * padded[i : i + arr.shape[0], j : j + arr.shape[1], c]
            out[..., c] = acc
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class RandomJPEGArtifacts:
    """Phone/WhatsApp recompression - the state most field photos arrive in."""

    def __init__(self, p: float = 0.30, quality: tuple[int, int] = (28, 72)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomCanopyOcclusion:
    """Another leaf, a stem or a finger crossing the frame."""

    def __init__(self, p: float = 0.25, max_boxes: int = 3):
        self.p, self.max_boxes = p, max_boxes

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        from PIL import ImageDraw

        img = img.copy()
        d = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(random.randint(1, self.max_boxes)):
            colour = (
                random.randint(20, 90),
                random.randint(60, 140),
                random.randint(20, 80),
            )
            x0, y0 = random.uniform(0, w), random.uniform(0, h)
            d.line(
                [(x0, y0), (x0 + random.uniform(-w, w), y0 + random.uniform(-h, h))],
                fill=colour,
                width=random.randint(int(w * 0.03), int(w * 0.12)),
            )
        return img.filter(ImageFilter.GaussianBlur(radius=0.4))


@dataclass
class AugmentationConfig:
    """Knobs for how hard the field simulation pushes.

    ``strength=0`` gives clean resize+normalise (useful when fine-tuning on a
    small, already field-collected set); ``1.0`` is the default field recipe.
    """

    image_size: int = 224
    strength: float = 1.0
    hflip: bool = True
    vflip: bool = True          # a leaf photo has no canonical "up"
    max_rotation: float = 25.0
    scale: tuple[float, float] = (0.55, 1.0)
    colour_jitter: float = 0.30
    grayscale_p: float = 0.03
    erasing_p: float = 0.20


def build_train_transform(cfg: AugmentationConfig | None = None) -> T.Compose:
    cfg = cfg or AugmentationConfig()
    s = max(0.0, min(1.5, cfg.strength))
    jitter = cfg.colour_jitter * s

    stages: list = [
        T.RandomResizedCrop(
            cfg.image_size,
            scale=(1.0 - (1.0 - cfg.scale[0]) * s, cfg.scale[1]),
            ratio=(0.8, 1.25),
        )
    ]
    if cfg.hflip:
        stages.append(T.RandomHorizontalFlip())
    if cfg.vflip:
        stages.append(T.RandomVerticalFlip(p=0.3))
    if cfg.max_rotation:
        stages.append(T.RandomRotation(cfg.max_rotation * s, expand=False))

    if s > 0:
        stages += [
            RandomShadow(p=0.35 * s),
            RandomSunGlare(p=0.25 * s),
            RandomCanopyOcclusion(p=0.25 * s),
            RandomMotionBlur(p=0.20 * s),
            T.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.08, jitter / 4)
            ),
            T.RandomGrayscale(p=cfg.grayscale_p * s),
            RandomJPEGArtifacts(p=0.30 * s),
        ]

    stages += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if cfg.erasing_p and s > 0:
        stages.append(T.RandomErasing(p=cfg.erasing_p * s, scale=(0.02, 0.12)))
    return T.Compose(stages)


def build_eval_transform(image_size: int = 224, crop_pct: float = 0.90) -> T.Compose:
    """Deterministic eval preprocessing - mirrored exactly by the edge runtime."""
    resize = int(round(image_size / crop_pct))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Undo normalisation, for writing debug/QA image grids."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "AugmentationConfig",
    "build_train_transform",
    "build_eval_transform",
    "denormalise",
    "RandomShadow",
    "RandomSunGlare",
    "RandomMotionBlur",
    "RandomJPEGArtifacts",
    "RandomCanopyOcclusion",
]
