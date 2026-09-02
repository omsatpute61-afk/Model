"""Preprocessing for the field device, in numpy + Pillow only.

This is a deliberate reimplementation of the torchvision eval transform. It
exists because installing torch on a Raspberry Pi to resize an image is
absurd - but it means there are now two implementations of the same thing, and
if they drift the model silently sees inputs it was never trained on.

Two defences: both read their parameters from the same ``PreprocessSpec`` in
the model card, and ``tests/test_preprocess_parity.py`` asserts the two paths
produce the same tensor within tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..model_card import PreprocessSpec


def resize_shorter_side(img: Image.Image, size: int, resample=Image.BILINEAR) -> Image.Image:
    """Match ``torchvision.transforms.Resize(int)``: shorter side -> ``size``."""
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    # torchvision truncates the long side (int(), not round()). A one-pixel
    # difference here shifts every subsequent pixel and shows up as a large
    # tensor mismatch, so match it exactly rather than approximately.
    if w <= h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return img.resize((new_w, new_h), resample)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:  # pad rather than fail on a small image
        canvas = Image.new("RGB", (max(w, size), max(h, size)), (124, 116, 104))
        canvas.paste(img, ((max(w, size) - w) // 2, (max(h, size) - h) // 2))
        img = canvas
        w, h = img.size
    # torchvision rounds the crop offset; integer division would be off by one
    # on an odd difference.
    left, top = int(round((w - size) / 2.0)), int(round((h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def to_tensor(img: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    """HWC uint8 PIL image -> NCHW float32 normalised array."""
    arr = np.asarray(img, dtype=np.float32) * spec.scale
    mean = np.array(spec.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(spec.std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    if spec.layout == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def preprocess_image(img: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    img = img.convert(spec.colour_space)
    img = resize_shorter_side(img, spec.resize_to)
    img = center_crop(img, spec.image_size)
    return to_tensor(img, spec)


def load_and_preprocess(path: str | Path, spec: PreprocessSpec) -> np.ndarray:
    with Image.open(path) as img:
        return preprocess_image(img, spec)


def preprocess_batch(images: Iterable[Image.Image], spec: PreprocessSpec) -> np.ndarray:
    batch = [preprocess_image(im, spec) for im in images]
    if not batch:
        raise ValueError("no images to preprocess")
    return np.concatenate(batch, axis=0)


def tile_image(
    img: Image.Image, tile: int, overlap: float = 0.2, max_tiles: int = 24
) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Split a wide canopy shot into overlapping tiles.

    A single 4000x3000 canopy photo downscaled to 224x224 loses every pest that
    matters - a whitefly is a few pixels. Tiling keeps the pest at a resolution
    the model was trained on, and the tile coordinates come back so a detection
    can be pointed at a place in the frame.
    """
    w, h = img.size
    if w <= tile and h <= tile:
        return [(img, (0, 0, w, h))]

    step = max(1, int(tile * (1.0 - overlap)))
    boxes: list[tuple[int, int, int, int]] = []
    for top in range(0, max(1, h - tile + step), step):
        for left in range(0, max(1, w - tile + step), step):
            l, t = min(left, max(0, w - tile)), min(top, max(0, h - tile))
            box = (l, t, min(l + tile, w), min(t + tile, h))
            if box not in boxes:
                boxes.append(box)

    if len(boxes) > max_tiles:  # keep latency bounded on a field device
        stride = len(boxes) / max_tiles
        boxes = [boxes[int(i * stride)] for i in range(max_tiles)]
    return [(img.crop(b), b) for b in boxes]


__all__ = [
    "preprocess_image",
    "preprocess_batch",
    "load_and_preprocess",
    "resize_shorter_side",
    "center_crop",
    "to_tensor",
    "tile_image",
]
