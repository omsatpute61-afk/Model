"""Cleaning a real crop-image corpus before anything is trained on it.

Scraped agricultural datasets carry a predictable set of defects, and each one
inflates a metric rather than announcing itself:

* **Truncated / non-image files.** A run that dies at hour six is the good
  outcome; the bad one is a decoder silently returning grey.
* **Exact duplicates.** The same photo appears under two classes, or twice in
  one class. Duplicates across a split boundary leak the test set into
  training. Duplicates under *different* labels are label noise that caps
  achievable accuracy.
* **Near-duplicates.** Burst frames of one leaf, or the same image rescaled and
  re-uploaded. A random split scatters them across train and val and the model
  scores on images it has memorised.
* **Non-crop images.** Logos, charts, screenshots and reference diagrams are
  common in scraped sets and teach the model nothing useful.
* **Degenerate images.** Thumbnails too small to show a lesion, extreme aspect
  ratios, and blank or single-colour frames.

Everything here is *reported first and applied second*: cleaning silently is
how a dataset loses half its rare-pest images without anyone noticing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageFile

from .manifest import Record

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger("cropguard.clean")

#: Reasons a record can be dropped. Every dropped image carries one.
DROP_REASONS = (
    "unreadable",
    "too_small",
    "extreme_aspect",
    "blank",
    "exact_duplicate",
    "near_duplicate",
    "cross_label_duplicate",
)


@dataclass
class CleaningConfig:
    min_side: int = 64          # below this a lesion is not resolvable
    max_aspect: float = 4.0     # banners, strips, screenshots
    min_std: float = 6.0        # flat/blank frames
    phash_size: int = 8         # 64-bit perceptual hash
    near_duplicate_distance: int = 4   # Hamming distance treated as "same image"
    drop_cross_label_duplicates: bool = True
    keep_one_per_duplicate_group: bool = True
    max_workers: int = 0        # 0 = serial; set >0 to thread the decode


@dataclass
class ImageFacts:
    """What one decode pass learned about an image."""

    path: str
    ok: bool
    width: int = 0
    height: int = 0
    mode: str = ""
    bytes: int = 0
    mean: float = 0.0
    std: float = 0.0
    sharpness: float = 0.0
    content_hash: str = ""
    phash: int = 0
    error: str = ""

    @property
    def aspect(self) -> float:
        if not self.width or not self.height:
            return 0.0
        return max(self.width, self.height) / min(self.width, self.height)


@dataclass
class CleaningReport:
    kept: list[Record] = field(default_factory=list)
    dropped: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    facts: dict[str, ImageFacts] = field(default_factory=dict)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    cross_label_pairs: list[tuple[str, str, str, str]] = field(default_factory=list)

    @property
    def dropped_count(self) -> int:
        return sum(len(v) for v in self.dropped.values())

    def summary(self) -> dict:
        per_class_dropped: Counter = Counter()
        for reason, items in self.dropped.items():
            for path, cls in items:
                per_class_dropped[cls] += 1
        return {
            "kept": len(self.kept),
            "dropped": self.dropped_count,
            "dropped_by_reason": {k: len(v) for k, v in sorted(self.dropped.items())},
            "duplicate_groups": len(self.duplicate_groups),
            "cross_label_duplicates": len(self.cross_label_pairs),
            "dropped_per_class": dict(per_class_dropped.most_common()),
            "kept_per_class": dict(Counter(r.class_id for r in self.kept).most_common()),
        }

    def format(self, top: int = 12) -> str:
        s = self.summary()
        lines = [
            f"kept {s['kept']} images, dropped {s['dropped']}",
            "  by reason:",
        ]
        for reason, n in s["dropped_by_reason"].items():
            lines.append(f"    {reason:<24} {n}")
        if self.cross_label_pairs:
            lines.append(
                f"  identical images under DIFFERENT labels: {len(self.cross_label_pairs)}"
            )
            lines.append("    (label noise - these cap the accuracy any model can reach)")
            for a, ca, b, cb in self.cross_label_pairs[:6]:
                lines.append(f"    {ca}  vs  {cb}")
        worst = sorted(s["dropped_per_class"].items(), key=lambda kv: -kv[1])[:top]
        if worst:
            lines.append("  classes losing the most images:")
            for cls, n in worst:
                kept = s["kept_per_class"].get(cls, 0)
                share = n / max(1, n + kept)
                flag = "  <-- check this class" if share > 0.4 else ""
                lines.append(f"    {cls:<38} -{n:<6} ({share:.0%} of it){flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# per-image inspection
# ---------------------------------------------------------------------------
def _phash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: robust to rescaling and mild recompression."""
    small = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def inspect_image(path: str | Path, cfg: CleaningConfig | None = None) -> ImageFacts:
    """One decode, all the statistics the cleaner and the EDA need."""
    cfg = cfg or CleaningConfig()
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return ImageFacts(path=str(p), ok=False, error=f"unreadable file: {exc}")

    try:
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
            width, height = rgb.size
            mode = im.mode
            small = np.asarray(rgb.resize((64, 64), Image.BILINEAR), dtype=np.float32)
            grey = small.mean(axis=2)
            # variance of the Laplacian: the standard cheap blur proxy
            lap = (
                -4 * grey[1:-1, 1:-1]
                + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
            )
            facts = ImageFacts(
                path=str(p), ok=True, width=width, height=height, mode=mode,
                bytes=len(raw), mean=float(grey.mean()), std=float(grey.std()),
                sharpness=float(lap.var()),
                content_hash=hashlib.blake2b(raw, digest_size=16).hexdigest(),
                phash=_phash(rgb, cfg.phash_size),
            )
            return facts
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same to us
        return ImageFacts(path=str(p), ok=False, bytes=len(raw), error=str(exc)[:200])


def inspect_all(
    paths: Sequence[str | Path],
    cfg: CleaningConfig | None = None,
    progress_every: int = 5000,
) -> dict[str, ImageFacts]:
    cfg = cfg or CleaningConfig()
    out: dict[str, ImageFacts] = {}
    if cfg.max_workers and cfg.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            for i, facts in enumerate(pool.map(lambda p: inspect_image(p, cfg), paths), 1):
                out[facts.path] = facts
                if progress_every and i % progress_every == 0:
                    LOGGER.info("inspected %d/%d images", i, len(paths))
    else:
        for i, p in enumerate(paths, 1):
            facts = inspect_image(p, cfg)
            out[facts.path] = facts
            if progress_every and i % progress_every == 0:
                LOGGER.info("inspected %d/%d images", i, len(paths))
    return out


# ---------------------------------------------------------------------------
# duplicate detection
# ---------------------------------------------------------------------------
def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(
    facts: Iterable[ImageFacts], max_distance: int = 4, bucket_bits: int = 16
) -> list[list[str]]:
    """Group perceptually identical images.

    Comparing every pair is O(n^2) and impossible at 200k images. Images are
    bucketed by several rotations of the hash prefix first, so only plausible
    matches are compared - the standard trick, and it keeps this linear enough
    to run on a laptop.
    """
    items = [f for f in facts if f.ok]
    if not items:
        return []

    buckets: dict[tuple[int, int], list[ImageFacts]] = defaultdict(list)
    rotations = (0, 16, 32, 48)
    for f in items:
        for r in rotations:
            key = (r, (f.phash >> r) & ((1 << bucket_bits) - 1))
            buckets[key].append(f)

    parent: dict[str, str] = {f.path: f.path for f in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bucket in buckets.values():
        if len(bucket) < 2 or len(bucket) > 2000:  # a huge bucket is degenerate
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if _hamming(bucket[i].phash, bucket[j].phash) <= max_distance:
                    union(bucket[i].path, bucket[j].path)

    groups: dict[str, list[str]] = defaultdict(list)
    for f in items:
        groups[find(f.path)].append(f.path)
    return [sorted(g) for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# the cleaner
# ---------------------------------------------------------------------------
def clean_records(
    records: Sequence[Record],
    cfg: CleaningConfig | None = None,
    facts: dict[str, ImageFacts] | None = None,
) -> CleaningReport:
    """Inspect, deduplicate and filter a set of records.

    Returns a report; the caller decides whether to write the cleaned manifest.
    """
    cfg = cfg or CleaningConfig()
    report = CleaningReport()
    facts = facts if facts is not None else inspect_all([r.path for r in records], cfg)
    report.facts = facts

    by_class = {r.path: r.class_id for r in records}
    survivors: list[Record] = []

    # 1. per-image defects
    for rec in records:
        f = facts.get(rec.path)
        if f is None or not f.ok:
            report.dropped["unreadable"].append((rec.path, rec.class_id))
            continue
        if min(f.width, f.height) < cfg.min_side:
            report.dropped["too_small"].append((rec.path, rec.class_id))
            continue
        if f.aspect > cfg.max_aspect:
            report.dropped["extreme_aspect"].append((rec.path, rec.class_id))
            continue
        if f.std < cfg.min_std:
            report.dropped["blank"].append((rec.path, rec.class_id))
            continue
        survivors.append(rec)

    # 2. exact duplicates, by content hash
    seen_hash: dict[str, Record] = {}
    after_exact: list[Record] = []
    for rec in survivors:
        h = facts[rec.path].content_hash
        first = seen_hash.get(h)
        if first is None:
            seen_hash[h] = rec
            after_exact.append(rec)
            continue
        if first.class_id != rec.class_id:
            report.cross_label_pairs.append((first.path, first.class_id, rec.path, rec.class_id))
            if not cfg.drop_cross_label_duplicates:
                after_exact.append(rec)
                continue
        report.dropped[
            "cross_label_duplicate" if first.class_id != rec.class_id else "exact_duplicate"
        ].append((rec.path, rec.class_id))

    # 3. near-duplicates, by perceptual hash
    groups = find_near_duplicates(
        (facts[r.path] for r in after_exact), cfg.near_duplicate_distance
    )
    report.duplicate_groups = groups

    drop_paths: set[str] = set()
    if cfg.keep_one_per_duplicate_group:
        for group in groups:
            labels = {by_class.get(p) for p in group}
            if len(labels) > 1:
                # Same picture, two labels: at most one can be right. Keeping
                # either would be a guess, so drop the lot and report it.
                for p in group:
                    report.cross_label_pairs.append(
                        (group[0], by_class.get(group[0], "?"), p, by_class.get(p, "?"))
                    )
                drop_paths.update(group)
                continue
            # Keep the sharpest, largest copy of the group.
            best = max(group, key=lambda p: (facts[p].sharpness, facts[p].width * facts[p].height))
            drop_paths.update(p for p in group if p != best)

    for rec in after_exact:
        if rec.path in drop_paths:
            reason = (
                "cross_label_duplicate"
                if any(rec.path == p for _, _, p, _ in report.cross_label_pairs)
                else "near_duplicate"
            )
            report.dropped[reason].append((rec.path, rec.class_id))
        else:
            report.kept.append(rec)

    return report


def drop_rare_classes(
    records: Sequence[Record], min_images: int = 25
) -> tuple[list[Record], dict[str, int]]:
    """Remove classes with too few images to learn or to evaluate.

    A class with 6 images cannot support a train/val/test split: whatever the
    model does on those two validation images is noise, and a per-class F1
    computed from them is not a measurement. Better to say the class is not
    covered than to publish a number that means nothing.
    """
    counts = Counter(r.class_id for r in records)
    dropped = {c: n for c, n in counts.items() if n < min_images}
    kept = [r for r in records if r.class_id not in dropped]
    return kept, dropped


def cap_class_size(
    records: Sequence[Record], max_images: int, seed: int = 1337
) -> tuple[list[Record], dict[str, int]]:
    """Cap over-represented classes to blunt an extreme long tail.

    Balanced sampling handles moderate imbalance. When one class has 40 000
    images and another 200, capping the head is cheaper than reweighting it and
    keeps epochs a sensible length.
    """
    import random

    rng = random.Random(seed)
    by_class: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_class[r.class_id].append(r)

    kept: list[Record] = []
    capped: dict[str, int] = {}
    for cls, items in by_class.items():
        if len(items) > max_images:
            capped[cls] = len(items) - max_images
            items = rng.sample(items, max_images)
        kept.extend(items)
    return kept, capped


__all__ = [
    "CleaningConfig",
    "CleaningReport",
    "ImageFacts",
    "DROP_REASONS",
    "inspect_image",
    "inspect_all",
    "clean_records",
    "find_near_duplicates",
    "drop_rare_classes",
    "cap_class_size",
]
