"""Adapters for the real datasets: DLCPD-25 and AP162.

Two public datasets, two quite different shapes, neither of which matches the
flat ``root/<class>/*.jpg`` layout the rest of the pipeline assumes:

**DLCPD-25** (232k images, 25 crops, ~282 classes) is nested as
``root/<Crop>/<Class>/*.jpg``. A single-level scanner reads that as 25 classes
named after crops, which is silently wrong - every disease of a crop collapses
into one label and the model looks like it is training fine.

**AP162** (194k images, 162 pest classes) is a flat species-level set that
splits larva from adult into separate classes. We merge each pair into one pest
class and keep the stage, per ``resources/dataset_maps/ap162.json``.

Both are distributed through Baidu Netdisk and neither can be fetched
automatically; see ``docs/real_datasets.md``. This module assumes the archive is
already unpacked somewhere local.

Nothing here silently drops data. Directories that cannot be mapped are
returned as a report, because on a 282-class dataset an unmapped folder is
usually a class worth adding to the taxonomy rather than a mistake.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..taxonomy import LIFE_STAGES, RESOURCE_DIR, Taxonomy, load_taxonomy, normalise_alias
from .manifest import IMAGE_SUFFIXES, Record, default_group_key

LOGGER = logging.getLogger("cropguard.ingest")

DATASET_MAP_DIR = RESOURCE_DIR / "dataset_maps"

#: Folder-name fragments that mean "this level is a split, not a class".
SPLIT_DIR_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing", "images"}

_LIFE_STAGE_TOKENS = (
    ("larva", "larva"), ("larvae", "larva"), ("caterpillar", "larva"), ("grub", "larva"),
    ("nymph", "nymph"), ("egg", "egg"), ("adult", "adult"), ("imago", "adult"),
)


@dataclass
class IngestReport:
    """What was read, what was skipped, and why."""

    source: str
    records: list[Record] = field(default_factory=list)
    unmapped: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    empty_dirs: list[str] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def class_count(self) -> int:
        return len({r.class_id for r in self.records})

    def summary(self) -> dict:
        return {
            "source": self.source,
            "images": self.image_count,
            "classes": self.class_count,
            "unmapped_folders": len(self.unmapped),
            "unmapped_images": sum(self.unmapped.values()),
            "excluded_folders": len(self.excluded),
            "excluded_images": sum(self.excluded.values()),
            "empty_dirs": len(self.empty_dirs),
            "per_class": dict(Counter(r.class_id for r in self.records)),
        }

    def format(self, top_unmapped: int = 25) -> str:
        lines = [
            f"{self.source}: {self.image_count} images in {self.class_count} classes",
        ]
        if self.excluded:
            lines.append(
                f"  deliberately excluded: {sum(self.excluded.values())} images "
                f"in {len(self.excluded)} folders"
            )
        if self.unmapped:
            lines.append(
                f"  UNMAPPED: {sum(self.unmapped.values())} images in "
                f"{len(self.unmapped)} folders - add an alias or a taxonomy class:"
            )
            for name, n in sorted(self.unmapped.items(), key=lambda kv: -kv[1])[:top_unmapped]:
                lines.append(f"    {n:>7}  {name}")
            if len(self.unmapped) > top_unmapped:
                lines.append(f"    ... and {len(self.unmapped) - top_unmapped} more")
        return "\n".join(lines)


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _looks_like_split(name: str) -> bool:
    return name.strip().lower() in SPLIT_DIR_NAMES


def detect_layout(root: Path, max_probe: int = 40) -> str:
    """Return ``"flat"``, ``"nested"`` or ``"split"`` for an unpacked dataset.

    Guessing this wrong is the expensive mistake: reading DLCPD-25's
    ``Crop/Class`` tree as flat produces 25 crop-shaped labels and a model that
    trains to a good-looking accuracy on the wrong problem entirely.
    """
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not subdirs:
        return "flat"
    if all(_looks_like_split(d.name) for d in subdirs[:max_probe]):
        return "split"

    nested = 0
    for d in subdirs[:max_probe]:
        children = [c for c in d.iterdir() if c.is_dir()]
        direct = any(p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir() if p.is_file())
        if children and not direct:
            nested += 1
    return "nested" if nested > len(subdirs[:max_probe]) / 2 else "flat"


def parse_life_stage(name: str) -> str:
    """Pull ``larva`` / ``adult`` / ``nymph`` / ``egg`` out of a class name."""
    low = f" {normalise_alias(name)} "
    for token, stage in _LIFE_STAGE_TOKENS:
        if f" {token} " in low or low.strip().endswith(f" {token}"):
            return stage
    return "unknown"


def strip_life_stage(name: str) -> str:
    """The species name with any life-stage token removed."""
    out = name
    for token, _ in _LIFE_STAGE_TOKENS:
        out = re.sub(rf"(?i)\b{token}\b", " ", out)
    return " ".join(out.split())


# ---------------------------------------------------------------------------
# generic nested scanner
# ---------------------------------------------------------------------------
def scan_nested(
    root: str | Path,
    taxonomy: Taxonomy | None = None,
    source: str = "",
    group_key: Callable[[Path], str] | None = None,
    crops: Sequence[str] | None = None,
) -> IngestReport:
    """Read ``root/<Crop>/<Class>/*.jpg`` (DLCPD-25 shape).

    Resolution is tried against, in order: ``"Crop Class"``, the class name on
    its own, and ``"Crop___Class"``. That covers the taxonomy's own ids, the
    PlantVillage-style aliases already in the table, and bare class names that
    are only unambiguous once the crop is known.
    """
    tax = taxonomy or load_taxonomy()
    keyer = group_key or default_group_key
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    report = IngestReport(source=source or root.name)
    wanted = {c.lower() for c in crops} if crops else None

    for crop_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        crop_name = crop_dir.name
        if wanted is not None and _crop_key(crop_name) not in wanted:
            images = _images_in(crop_dir)
            if images:
                report.excluded[f"{crop_name}/*"] = len(images)
            continue

        class_dirs = [d for d in sorted(crop_dir.iterdir()) if d.is_dir()]
        if not class_dirs:  # crop folder holding images directly
            class_dirs = [crop_dir]

        for class_dir in class_dirs:
            images = _images_in(class_dir)
            if not images:
                report.empty_dirs.append(str(class_dir.relative_to(root)))
                continue

            label = class_dir.name if class_dir != crop_dir else crop_name
            crop_class = _resolve_nested(tax, crop_name, label)
            if crop_class is None:
                report.unmapped[f"{crop_name}/{label}"] = len(images)
                continue

            for img in images:
                report.records.append(
                    Record(
                        path=str(img),
                        class_id=crop_class.id,
                        category=crop_class.category,
                        severity="unknown",
                        life_stage=parse_life_stage(label),
                        group=keyer(img),
                        source=report.source,
                    )
                )
    return report


def _crop_key(name: str) -> str:
    """Normalise a dataset's crop folder name onto our crop vocabulary."""
    key = normalise_alias(name)
    synonyms = {
        "corn": "maize", "corn maize": "maize", "zea mays": "maize",
        "vitis": "grape", "grapes": "grape", "vitis vinifera": "grape",
        "bell pepper": "chilli", "pepper": "chilli", "pepper bell": "chilli",
        "capsicum": "chilli", "chili": "chilli", "chile": "chilli",
        "paddy": "rice", "solanum lycopersicum": "tomato",
        "citrus fruits": "citrus", "orange": "citrus", "mandarin": "citrus",
        "soya": "soybean", "soya bean": "soybean", "glycine max": "soybean",
    }
    return synonyms.get(key, key)


def _resolve_nested(tax: Taxonomy, crop: str, label: str):
    """Try the plausible spellings of "this crop, this condition".

    Datasets are inconsistent about whether the class folder repeats the crop:
    the same disease appears as ``Corn/Corn___Common_rust_`` in one release and
    ``Corn/Common_rust`` in another. Stripping a repeated crop prefix and
    re-prefixing with our own crop name lets both land on ``maize__common_rust``
    without needing an alias for every spelling.
    """
    crop_key = _crop_key(crop)
    bare = _strip_crop_prefix(label, crop, crop_key)

    # Crop-qualified spellings first. A bare label is tried last and only
    # accepted if the class belongs to this crop: "Rust" is an alias of both
    # sugarcane rust and soybean rust, and resolving Soybean/Rust to
    # sugarcane__rust would mislabel an entire class without any error.
    qualified = (
        f"{crop_key} {bare}",
        f"{crop_key} {label}",
        f"{crop} {label}",
        f"{crop}___{label}",
    )
    for candidate in qualified:
        found = tax.resolve(candidate)
        if found is not None:
            return found

    for candidate in (label, bare):
        found = tax.resolve(candidate)
        if found is not None and _crop_compatible(found, crop_key):
            return found

    state = normalise_alias(bare)
    if state in {"healthy", "health", "normal", "fresh"}:
        return tax.resolve(f"{crop_key} healthy")
    return None


def _crop_compatible(crop_class, crop_key: str) -> bool:
    """Guard against a bare class name matching a different crop's disease."""
    return crop_class.crop in ("any", crop_key)


def _strip_crop_prefix(label: str, *crops: str) -> str:
    """Drop a leading crop name from a class folder ("Corn___Common_rust_")."""
    words = normalise_alias(label).split()
    for crop in crops:
        crop_words = normalise_alias(crop).split()
        n = len(crop_words)
        if n and words[:n] == crop_words and len(words) > n:
            words = words[n:]
    return " ".join(words)


# ---------------------------------------------------------------------------
# AP162
# ---------------------------------------------------------------------------
def load_dataset_map(name: str = "ap162") -> dict:
    with open(DATASET_MAP_DIR / f"{name}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ap162_classes(path: str | Path) -> dict[int, str]:
    """Read the dataset's own ``classes.txt`` (``index<TAB>name`` per line)."""
    out: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\t ]+", line, maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            out[int(parts[0])] = parts[1].strip()
    return out


def scan_ap162(
    root: str | Path,
    taxonomy: Taxonomy | None = None,
    classes_file: str | Path | None = None,
    source: str = "AP162",
    group_key: Callable[[Path], str] | None = None,
    dataset_map: dict | None = None,
) -> IngestReport:
    """Read AP162, merging larva/adult pairs and dropping out-of-scope species.

    Class folders may be named by index (``0``, ``1``, ...) or by species
    (``Spodoptera frugiperda larva``); both are handled. A ``train/``-style
    split layer above the classes is descended into automatically, since the
    dataset's own split is discarded - we build our own grouped, stratified one.
    """
    tax = taxonomy or load_taxonomy()
    keyer = group_key or default_group_key
    root = Path(root)
    spec = dataset_map or load_dataset_map("ap162")
    mapping = {int(k): tuple(v) for k, v in spec["map"].items()}
    excluded_reasons = {int(k): v for k, v in spec["excluded"].items()}

    names = load_ap162_classes(classes_file) if classes_file else {}
    by_name = {normalise_alias(v): k for k, v in names.items()}

    report = IngestReport(source=source)
    for class_dir in _class_dirs(root):
        images = _images_in(class_dir)
        if not images:
            report.empty_dirs.append(class_dir.name)
            continue

        index = _ap162_index(class_dir.name, by_name)
        if index is None:
            report.unmapped[class_dir.name] = len(images)
            continue
        if index in excluded_reasons:
            report.excluded[f"{index}:{class_dir.name}"] = len(images)
            continue
        if index not in mapping:
            report.unmapped[f"{index}:{class_dir.name}"] = len(images)
            continue

        class_id, stage = mapping[index]
        crop_class = tax.get(class_id)
        if crop_class is None:
            report.unmapped[f"{class_id} (not in taxonomy)"] = len(images)
            continue
        if stage not in LIFE_STAGES:
            stage = "unknown"

        for img in images:
            report.records.append(
                Record(
                    path=str(img),
                    class_id=class_id,
                    category=crop_class.category,
                    severity="unknown",
                    life_stage="unknown" if stage == "any" else stage,
                    group=keyer(img),
                    source=source,
                )
            )
    return report


def _class_dirs(root: Path) -> Iterable[Path]:
    """Yield every class directory, descending through a train/val/test layer.

    The dataset's own split is deliberately ignored: we re-split by group so
    that near-duplicate images of one insect cannot straddle train and val.
    """
    tops = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if tops and all(_looks_like_split(d.name) for d in tops):
        for split_dir in tops:
            yield from (d for d in sorted(split_dir.iterdir()) if d.is_dir())
        return
    yield from tops


def _ap162_index(folder: str, by_name: dict[str, int]) -> int | None:
    name = folder.strip()
    if name.isdigit():
        return int(name)
    m = re.match(r"^(\d+)[_\-. ]", name)
    if m:
        return int(m.group(1))
    return by_name.get(normalise_alias(name))


__all__ = [
    "IngestReport",
    "scan_nested",
    "scan_ap162",
    "detect_layout",
    "load_dataset_map",
    "load_ap162_classes",
    "parse_life_stage",
    "strip_life_stage",
]
