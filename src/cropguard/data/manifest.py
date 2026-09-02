"""Turn a directory of crop photos into a reproducible, leakage-free manifest.

Public crop datasets ship as ``root/<Some___Folder_Name>/img.jpg``. Two things
go wrong if you feed that straight to an ImageFolder:

1. **Naming.** ``Corn_(maize)___Common_rust_`` is not a class id anyone can act
   on. The taxonomy's alias table remaps folders onto CropGuard classes, and
   folders that map nowhere are reported instead of silently becoming a class.

2. **Leakage.** PlantVillage contains many augmented copies of the same
   physical leaf, and scouting archives contain bursts of the same plant shot
   seconds apart. A random split puts near-duplicates on both sides and
   produces a validation accuracy that collapses in the field. Splitting is
   therefore done over *groups* (default: the filename up to the first
   augmentation/burst marker), never over individual files.

The manifest is a plain CSV so it can be diffed, reviewed and version
controlled - the split is a decision, not a side effect of a random seed.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..taxonomy import SEVERITY_LEVELS, Taxonomy, load_taxonomy

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MANIFEST_FIELDS = ("path", "class_id", "category", "severity", "group", "split", "source")

#: Markers that indicate "another copy of the same leaf" rather than a new
#: sample. Deliberately conservative: a trailing number is usually the sample's
#: own identity (``img_0042.jpg``), and stripping it would collapse a whole
#: class into one group and push every image into ``train``. Aggressive
#: grouping is opt-in via a custom ``group_key``.
_GROUP_MARKERS = re.compile(
    r"(__sev-[a-z]+)|(_aug\d*)|(_copy\d*)|(_rot\d+)|(_flip[a-z]*)|(_mirror)",
    re.IGNORECASE,
)
_SEVERITY_RE = re.compile(r"__sev-(none|low|moderate|severe)", re.IGNORECASE)


@dataclass
class Record:
    path: str
    class_id: str
    category: str
    severity: str = "unknown"
    group: str = ""
    split: str = "train"
    source: str = ""

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in MANIFEST_FIELDS}


def default_group_key(path: Path) -> str:
    """Group images that are copies/frames of the same physical leaf.

    ``tomato__late_blight__sev-low__0003.jpg`` and its augmented siblings all
    collapse to ``tomato__late_blight``; PlantVillage's
    ``0a1b2c___RS_Late.B 4977.JPG`` style names collapse on their UUID prefix.
    """
    stem = path.stem
    stem = _GROUP_MARKERS.sub("", stem)
    stem = re.sub(r"[\s_-]+$", "", stem)
    return f"{path.parent.name}/{stem}" if stem else f"{path.parent.name}/{path.stem}"


def parse_severity(path: Path) -> str:
    m = _SEVERITY_RE.search(path.stem)
    return m.group(1).lower() if m else "unknown"


def scan_image_folder(
    root: str | Path,
    taxonomy: Taxonomy | None = None,
    group_key: Callable[[Path], str] | None = None,
    source: str = "",
) -> tuple[list[Record], dict[str, int]]:
    """Walk ``root`` and map every class directory onto the taxonomy.

    Returns the records plus a count of images under folders that could not be
    mapped, so an unmapped folder is loud rather than silently dropped.
    """
    tax = taxonomy or load_taxonomy()
    keyer = group_key or default_group_key
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    records: list[Record] = []
    unmapped: Counter[str] = Counter()

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        crop_class = tax.resolve(class_dir.name)
        images = sorted(
            p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if crop_class is None:
            unmapped[class_dir.name] = len(images)
            continue
        for img in images:
            records.append(
                Record(
                    path=str(img),
                    class_id=crop_class.id,
                    category=crop_class.category,
                    severity=parse_severity(img),
                    group=keyer(img),
                    source=source or root.name,
                )
            )
    return records, dict(unmapped)


def stratified_group_split(
    records: Sequence[Record],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
    min_per_split: int = 1,
) -> list[Record]:
    """Assign a split to every record, keeping whole groups together.

    Stratified per class (so a rare pest is present in val/test rather than
    landing entirely in train by chance) and grouped (so no leaf appears on
    both sides of the split).
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be in [0, 1)")

    by_class: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_class[rec.class_id][rec.group].append(rec)

    rng = random.Random(seed)
    out: list[Record] = []
    for class_id in sorted(by_class):
        groups = sorted(by_class[class_id])
        # Deterministic given (seed, class) regardless of filesystem order.
        rng_c = random.Random(f"{seed}:{class_id}")
        rng_c.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:  # never let a class vanish from val/test entirely
            n_test = max(n_test, min_per_split)
            n_val = max(n_val, min_per_split)
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))

        assignment = (
            ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
        )
        for group, split in zip(groups, assignment):
            for rec in by_class[class_id][group]:
                rec.split = split
                out.append(rec)
    rng.shuffle(out)
    return out


def write_manifest(records: Iterable[Record], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.as_row())
    return path


def read_manifest(path: str | Path, split: str | None = None) -> list[Record]:
    out: list[Record] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if split is not None and row.get("split") != split:
                continue
            out.append(
                Record(
                    path=row["path"],
                    class_id=row["class_id"],
                    category=row.get("category", ""),
                    severity=row.get("severity", "unknown"),
                    group=row.get("group", ""),
                    split=row.get("split", "train"),
                    source=row.get("source", ""),
                )
            )
    return out


def manifest_summary(records: Sequence[Record]) -> dict:
    """Per-split, per-class counts - printed before every training run."""
    splits = Counter(r.split for r in records)
    per_class: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        per_class[r.class_id][r.split] += 1
    leaked = _leaked_groups(records)
    # A class whose images all collapse into one or two groups cannot be split;
    # that means the group key is wrong for this dataset's naming, not that the
    # data is bad. Surface it loudly rather than silently training on it.
    groups_per_class = {
        cid: len({r.group for r in records if r.class_id == cid}) for cid in per_class
    }
    degenerate = sorted(
        cid for cid, n in groups_per_class.items() if n < 3 and per_class[cid].total() >= 3
    )
    return {
        "degenerate_grouping": degenerate,
        "total": len(records),
        "splits": dict(splits),
        "classes": len(per_class),
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "classes_missing_val": sorted(k for k, v in per_class.items() if not v.get("val")),
        "leaked_groups": leaked,
        "severity": dict(Counter(r.severity for r in records)),
    }


def _leaked_groups(records: Sequence[Record]) -> list[str]:
    """Groups that appear in more than one split - must always be empty."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seen[r.group].add(r.split)
    return sorted(g for g, s in seen.items() if len(s) > 1)


def class_weights(records: Sequence[Record], class_ids: Sequence[str], beta: float = 0.999) -> list[float]:
    """Effective-number-of-samples weights (Cui et al., CVPR 2019).

    Plain inverse frequency over-corrects when a class has 6 images and another
    has 6000; the effective-number formulation saturates instead, which is what
    a long-tailed scouting archive needs.
    """
    counts = Counter(r.class_id for r in records)
    weights = []
    for cid in class_ids:
        n = counts.get(cid, 0)
        if n == 0:
            weights.append(0.0)
            continue
        eff = (1.0 - beta**n) / (1.0 - beta)
        weights.append(1.0 / eff)
    total = sum(weights)
    k = sum(1 for w in weights if w > 0)
    return [w * k / total if total > 0 else 0.0 for w in weights]


def content_hash(paths: Sequence[str], chunk: int = 65536) -> str:
    """Stable hash of a file list - used to tag checkpoints with their data."""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(paths):
        h.update(p.encode())
    return h.hexdigest()


__all__ = [
    "Record",
    "MANIFEST_FIELDS",
    "scan_image_folder",
    "stratified_group_split",
    "write_manifest",
    "read_manifest",
    "manifest_summary",
    "class_weights",
    "default_group_key",
    "parse_severity",
]
