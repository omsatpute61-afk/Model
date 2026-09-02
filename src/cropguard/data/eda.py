"""Exploratory analysis of a crop-image corpus, aimed at training decisions.

This is not a gallery of pretty histograms. Every section answers a question
that changes what we do next:

* **How long is the tail?** Decides balanced sampling, focal loss, class caps,
  and which classes we should refuse to claim we support at all.
* **Which classes are too small to evaluate?** A per-class F1 from four
  validation images is not a measurement, and reporting one is misleading.
* **How much duplication is there?** Sets how much of the reported accuracy is
  memorisation.
* **Do splits leak?** A single leaked group makes every number optimistic.
* **What do the images actually look like?** Studio-clean or field-messy decides
  how hard the augmentation has to push. If the corpus is all 256x256 studio
  crops, a model trained on it will fall over on a phone photo, and we need to
  know that before training rather than after deployment.
* **Which crops and categories are covered?** The advisory layer promises
  coverage; the data has to back it.

Output is a markdown report plus a JSON blob, so it can be committed, diffed
and argued with.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..taxonomy import Taxonomy, load_taxonomy
from .clean import ImageFacts
from .manifest import Record


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def gini(counts: Sequence[int]) -> float:
    """0.0 = perfectly balanced, 1.0 = one class holds everything."""
    if not counts:
        return 0.0
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


@dataclass
class EDAResult:
    payload: dict

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, default=str)

    def save(self, directory: str | Path, stem: str = "eda") -> dict[str, Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        json_path = d / f"{stem}.json"
        md_path = d / f"{stem}.md"
        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(format_report(self.payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}


def analyse(
    records: Sequence[Record],
    facts: dict[str, ImageFacts] | None = None,
    taxonomy: Taxonomy | None = None,
    min_train: int = 25,
    min_eval: int = 10,
) -> EDAResult:
    """Build the full report payload."""
    tax = taxonomy or load_taxonomy()
    counts = Counter(r.class_id for r in records)
    total = len(records)

    payload: dict = {
        "totals": {
            "images": total,
            "classes": len(counts),
            "sources": dict(Counter(r.source for r in records)),
            "splits": dict(Counter(r.split for r in records)),
        }
    }

    # -- class balance ---------------------------------------------------
    ordered = counts.most_common()
    values = [n for _, n in ordered]
    head, tail = ordered[0] if ordered else ("", 0), ordered[-1] if ordered else ("", 0)
    payload["balance"] = {
        "largest_class": {"class_id": head[0], "images": head[1]},
        "smallest_class": {"class_id": tail[0], "images": tail[1]},
        "imbalance_ratio": (head[1] / tail[1]) if tail[1] else None,
        "gini": gini(values),
        "median_class_size": float(np.median(values)) if values else 0.0,
        "classes_under_min_train": sorted(c for c, n in counts.items() if n < min_train),
        "classes_under_min_eval": sorted(
            c for c, n in counts.items() if n < min_eval * 3
        ),
        "top_10": ordered[:10],
        "bottom_10": ordered[-10:][::-1],
    }

    # -- agronomic coverage ----------------------------------------------
    by_crop: Counter = Counter()
    by_category: Counter = Counter()
    crop_classes: dict[str, set] = defaultdict(set)
    unknown_classes = []
    for r in records:
        c = tax.get(r.class_id)
        if c is None:
            unknown_classes.append(r.class_id)
            continue
        by_crop[c.crop] += 1
        by_category[c.category] += 1
        crop_classes[c.crop].add(c.id)
    payload["coverage"] = {
        "images_per_crop": dict(by_crop.most_common()),
        "classes_per_crop": {k: len(v) for k, v in sorted(crop_classes.items())},
        "images_per_category": dict(by_category.most_common()),
        "classes_not_in_taxonomy": sorted(set(unknown_classes)),
        "crops_without_healthy_class": sorted(
            crop for crop, ids in crop_classes.items()
            if crop != "any" and not any(tax[i].category == "healthy" for i in ids)
        ),
    }

    # -- life stage (pest data only) --------------------------------------
    pest_records = [r for r in records if (tax.get(r.class_id) or None) and tax[r.class_id].category == "pest"]
    payload["life_stage"] = {
        "pest_images": len(pest_records),
        "distribution": dict(Counter(r.life_stage for r in pest_records).most_common()),
        "labelled_fraction": (
            sum(1 for r in pest_records if r.life_stage != "unknown") / len(pest_records)
            if pest_records else 0.0
        ),
    }

    # -- severity ---------------------------------------------------------
    payload["severity"] = {
        "distribution": dict(Counter(r.severity for r in records).most_common()),
        "labelled_fraction": (
            sum(1 for r in records if r.severity not in ("unknown", "")) / total if total else 0.0
        ),
    }

    # -- image properties --------------------------------------------------
    if facts:
        good = [f for f in facts.values() if f.ok]
        if good:
            widths = [f.width for f in good]
            heights = [f.height for f in good]
            sides = [min(f.width, f.height) for f in good]
            sharp = [f.sharpness for f in good]
            payload["images"] = {
                "decoded": len(good),
                "failed": sum(1 for f in facts.values() if not f.ok),
                "width": {"p05": _pct(widths, 5), "p50": _pct(widths, 50), "p95": _pct(widths, 95)},
                "height": {"p05": _pct(heights, 5), "p50": _pct(heights, 50), "p95": _pct(heights, 95)},
                "min_side": {"p05": _pct(sides, 5), "p50": _pct(sides, 50), "p95": _pct(sides, 95)},
                "aspect_p95": _pct([f.aspect for f in good], 95),
                "brightness_p50": _pct([f.mean for f in good], 50),
                "sharpness": {"p05": _pct(sharp, 5), "p50": _pct(sharp, 50), "p95": _pct(sharp, 95)},
                "share_below_224px": float(np.mean([s < 224 for s in sides])),
                "share_greyscale": float(np.mean([f.mode in ("L", "1") for f in good])),
                "uniform_resolution": len({(f.width, f.height) for f in good}) <= 3,
                "distinct_resolutions": len({(f.width, f.height) for f in good}),
            }

    # -- split integrity ---------------------------------------------------
    groups: dict[str, set] = defaultdict(set)
    for r in records:
        groups[r.group].add(r.split)
    leaked = sorted(g for g, s in groups.items() if len(s) > 1)
    per_class_split: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        per_class_split[r.class_id][r.split] += 1
    payload["splits"] = {
        "leaked_groups": len(leaked),
        "leaked_examples": leaked[:10],
        "classes_missing_val": sorted(c for c, s in per_class_split.items() if not s.get("val")),
        "classes_missing_test": sorted(c for c, s in per_class_split.items() if not s.get("test")),
        "groups": len(groups),
        "images_per_group_p95": _pct([len([r for r in records if r.group == g]) for g in list(groups)[:2000]], 95)
        if groups else 0.0,
    }

    payload["recommendations"] = _recommend(payload)
    return EDAResult(payload)


def _recommend(p: dict) -> list[str]:
    """Turn the numbers into the decisions they imply."""
    out: list[str] = []
    bal = p.get("balance", {})
    ratio = bal.get("imbalance_ratio")
    if ratio and ratio > 20:
        out.append(
            f"Imbalance ratio is {ratio:.0f}:1 (Gini {bal['gini']:.2f}). Keep "
            "balanced_sampling on, and consider optim.use_focal=true and a class cap."
        )
    elif ratio and ratio > 5:
        out.append(f"Imbalance ratio {ratio:.0f}:1 - balanced sampling is sufficient.")

    under = bal.get("classes_under_min_train", [])
    if under:
        out.append(
            f"{len(under)} class(es) have too few images to train on. Either collect "
            f"more, merge them into a coarser class, or drop them with "
            f"--min-per-class so the model does not claim coverage it lacks: {under[:8]}"
        )

    imgs = p.get("images")
    if imgs:
        if imgs.get("uniform_resolution"):
            out.append(
                f"Every image shares {imgs['distinct_resolutions']} resolution(s) - this is a "
                "pre-processed studio corpus, not field photography. Train with "
                "data.aug_strength=1.0 and validate on real phone photos before trusting any number."
            )
        if imgs.get("share_below_224px", 0) > 0.3:
            out.append(
                f"{imgs['share_below_224px']:.0%} of images have a side under 224 px. "
                "Prefer image_size=160 or 192; upscaling to 224 invents detail that is not there."
            )
        if imgs.get("share_greyscale", 0) > 0.02:
            out.append(
                f"{imgs['share_greyscale']:.0%} of images are greyscale, and colour is "
                "diagnostic for most of these classes. Check whether they belong in the set."
            )

    sp = p.get("splits", {})
    if sp.get("leaked_groups"):
        out.append(
            f"{sp['leaked_groups']} group(s) span more than one split - validation is "
            "optimistic. Re-split before believing any metric."
        )
    if sp.get("classes_missing_val"):
        out.append(
            f"{len(sp['classes_missing_val'])} class(es) have no validation images, so "
            "their per-class metrics are undefined."
        )

    ls = p.get("life_stage", {})
    if ls.get("pest_images") and ls.get("labelled_fraction", 0) < 0.5:
        out.append(
            f"Only {ls['labelled_fraction']:.0%} of pest images carry a life stage. The "
            "life-stage head will train on that subset and be masked elsewhere."
        )

    sev = p.get("severity", {})
    if sev.get("labelled_fraction", 0) < 0.05:
        out.append(
            "Severity is essentially unlabelled. The severity head will be masked out; "
            "either collect severity labels or set optim.severity_weight=0."
        )

    cov = p.get("coverage", {})
    if cov.get("crops_without_healthy_class"):
        out.append(
            "These crops have no healthy class, so the model cannot say 'this plant is "
            f"fine': {cov['crops_without_healthy_class']}. Add healthy images."
        )
    if cov.get("classes_not_in_taxonomy"):
        out.append(
            f"{len(cov['classes_not_in_taxonomy'])} class id(s) in the manifest are not in "
            "the taxonomy - the advisory layer has nothing to say about them."
        )
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _bar(n: int, largest: int, width: int = 34) -> str:
    filled = int(round(width * n / largest)) if largest else 0
    return "#" * max(1 if n else 0, filled)


def format_report(p: dict) -> str:
    t = p["totals"]
    lines = [
        "# Dataset EDA",
        "",
        f"- images: **{t['images']:,}**",
        f"- classes: **{t['classes']}**",
        f"- sources: {t['sources']}",
        f"- splits: {t['splits']}",
        "",
    ]

    if p.get("recommendations"):
        lines += ["## What this means for training", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(p["recommendations"], 1)]
        lines.append("")

    b = p["balance"]
    lines += [
        "## Class balance",
        "",
        f"- largest: `{b['largest_class']['class_id']}` ({b['largest_class']['images']:,})",
        f"- smallest: `{b['smallest_class']['class_id']}` ({b['smallest_class']['images']:,})",
        f"- imbalance ratio: **{b['imbalance_ratio']:.1f}:1**" if b["imbalance_ratio"] else "- imbalance ratio: n/a",
        f"- Gini coefficient: **{b['gini']:.3f}** (0 = balanced, 1 = one class holds everything)",
        f"- median class size: {b['median_class_size']:.0f}",
        "",
        "### Largest classes",
        "",
        "```",
    ]
    largest = b["top_10"][0][1] if b["top_10"] else 1
    for cls, n in b["top_10"]:
        lines.append(f"{cls:<38} {n:>7,} {_bar(n, largest)}")
    lines += ["```", "", "### Smallest classes", "", "```"]
    for cls, n in b["bottom_10"]:
        lines.append(f"{cls:<38} {n:>7,} {_bar(n, largest)}")
    lines += ["```", ""]

    c = p["coverage"]
    lines += ["## Coverage", "", "| crop | images | classes |", "|---|---:|---:|"]
    for crop, n in c["images_per_crop"].items():
        lines.append(f"| {crop} | {n:,} | {c['classes_per_crop'].get(crop, 0)} |")
    lines += ["", "| category | images |", "|---|---:|"]
    for cat, n in c["images_per_category"].items():
        lines.append(f"| {cat} | {n:,} |")
    lines.append("")

    if p.get("images"):
        i = p["images"]
        lines += [
            "## Image properties",
            "",
            f"- decoded: {i['decoded']:,}  (failed: {i['failed']})",
            f"- width p05/p50/p95: {i['width']['p05']:.0f} / {i['width']['p50']:.0f} / {i['width']['p95']:.0f}",
            f"- shorter side p05/p50/p95: {i['min_side']['p05']:.0f} / {i['min_side']['p50']:.0f} / {i['min_side']['p95']:.0f}",
            f"- distinct resolutions: {i['distinct_resolutions']}",
            f"- share under 224 px: {i['share_below_224px']:.1%}",
            f"- share greyscale: {i['share_greyscale']:.1%}",
            f"- sharpness p05/p50/p95: {i['sharpness']['p05']:.0f} / {i['sharpness']['p50']:.0f} / {i['sharpness']['p95']:.0f}",
            "",
        ]

    ls, sev = p.get("life_stage", {}), p.get("severity", {})
    lines += [
        "## Label completeness",
        "",
        f"- pest images with a life stage: {ls.get('labelled_fraction', 0):.1%} "
        f"of {ls.get('pest_images', 0):,}  {ls.get('distribution', {})}",
        f"- images with a severity label: {sev.get('labelled_fraction', 0):.1%}  "
        f"{sev.get('distribution', {})}",
        "",
    ]

    s = p["splits"]
    lines += [
        "## Split integrity",
        "",
        f"- groups: {s['groups']:,}",
        f"- leaked groups (in more than one split): **{s['leaked_groups']}**",
        f"- classes with no validation images: {len(s['classes_missing_val'])}",
        f"- classes with no test images: {len(s['classes_missing_test'])}",
        "",
    ]
    return "\n".join(lines)


__all__ = ["analyse", "EDAResult", "format_report", "gini"]
