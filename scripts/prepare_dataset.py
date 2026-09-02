#!/usr/bin/env python3
"""Build a training manifest from one or more image folders.

    # public dataset in ImageFolder layout (PlantVillage, IP102, ...)
    python scripts/prepare_dataset.py --source data/PlantVillage --out artifacts/data/manifest.csv

    # several sources at once, including your own district scouting archive
    python scripts/prepare_dataset.py \
        --source data/PlantVillage --source data/rice_diseases --source data/kvk_photos \
        --out artifacts/data/manifest.csv --val-frac 0.15 --test-frac 0.15

    # no dataset yet - generate synthetic data and run the whole pipeline today
    python scripts/prepare_dataset.py --synthetic --per-class 60 --out artifacts/data/manifest.csv

Folder names are matched against the taxonomy's alias table, so upstream
naming (``Corn_(maize)___Common_rust_``) is remapped without renaming files.
Folders that match nothing are **reported, not silently dropped** - an
unmapped folder is usually a class worth adding to the taxonomy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropguard.data.manifest import (  # noqa: E402
    manifest_summary,
    scan_image_folder,
    stratified_group_split,
    write_manifest,
)
from cropguard.taxonomy import load_taxonomy, save_taxonomy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=[], help="image folder (repeatable)")
    p.add_argument("--out", default="artifacts/data/manifest.csv")
    p.add_argument("--taxonomy", help="custom taxonomy JSON")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--min-per-class", type=int, default=0,
                   help="drop classes with fewer than this many images")
    p.add_argument("--synthetic", action="store_true",
                   help="generate a synthetic dataset first (no download needed)")
    p.add_argument("--synthetic-dir", default="artifacts/data/synthetic")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--classes", help="comma separated class ids to restrict to")
    p.add_argument("--crops", help="comma separated crops to restrict to (e.g. cotton,rice)")
    p.add_argument("--save-taxonomy", help="write the taxonomy actually present to this path")
    args = p.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    if args.crops:
        taxonomy = taxonomy.filter(crops=[c.strip() for c in args.crops.split(",")])
    class_ids = [c.strip() for c in args.classes.split(",")] if args.classes else None
    if class_ids:
        taxonomy = taxonomy.subset(class_ids)

    sources = list(args.source)
    if args.synthetic:
        from cropguard.data.synthetic import generate_dataset

        print(f"generating synthetic data ({args.per_class}/class) -> {args.synthetic_dir}")
        root = generate_dataset(
            args.synthetic_dir, taxonomy=taxonomy, class_ids=class_ids,
            per_class=args.per_class, size=args.image_size, seed=args.seed,
        )
        sources.append(str(root))

    if not sources:
        p.error("give at least one --source, or use --synthetic")

    records = []
    all_unmapped: dict[str, int] = {}
    for src in sources:
        recs, unmapped = scan_image_folder(src, taxonomy=taxonomy, source=Path(src).name)
        print(f"  {src}: {len(recs)} images in {len({r.class_id for r in recs})} classes")
        records.extend(recs)
        for k, v in unmapped.items():
            all_unmapped[f"{Path(src).name}/{k}"] = v

    if all_unmapped:
        print("\nUNMAPPED FOLDERS (not in the taxonomy - add an alias or a class):")
        for name, count in sorted(all_unmapped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>7} images  {name}")

    if not records:
        print("\nno images matched the taxonomy - nothing written", file=sys.stderr)
        return 1

    if args.min_per_class > 0:
        from collections import Counter

        counts = Counter(r.class_id for r in records)
        dropped = {c for c, n in counts.items() if n < args.min_per_class}
        if dropped:
            print(f"\ndropping {len(dropped)} class(es) with < {args.min_per_class} images: "
                  f"{sorted(dropped)}")
            records = [r for r in records if r.class_id not in dropped]

    records = stratified_group_split(
        records, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    out = write_manifest(records, args.out)
    summary = manifest_summary(records)

    print(f"\nwrote {out}")
    print(f"  images  {summary['total']}")
    print(f"  classes {summary['classes']}")
    print(f"  splits  {summary['splits']}")
    if summary["leaked_groups"]:
        print(f"  WARNING {len(summary['leaked_groups'])} group(s) span splits (data leakage)")
    if summary["classes_missing_val"]:
        print(f"  WARNING no validation images for: {summary['classes_missing_val'][:10]}")
    if summary["degenerate_grouping"]:
        print(f"  WARNING too few split groups for: {summary['degenerate_grouping'][:10]}")
        print("          (the filename pattern may be collapsing distinct images into one group)")

    if args.save_taxonomy:
        present = [c for c in taxonomy.class_ids if c in {r.class_id for r in records}]
        save_taxonomy(taxonomy.subset(present), args.save_taxonomy)
        print(f"  taxonomy -> {args.save_taxonomy}")

    summary_path = Path(args.out).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  summary  -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
