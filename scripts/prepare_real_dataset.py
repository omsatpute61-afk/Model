#!/usr/bin/env python3
"""Ingest -> clean -> EDA -> manifest for the disease and pest corpora.

    python scripts/prepare_real_dataset.py \
        --disease /data/Plant-Diseases-100k-Labelled-Images \
        --pest    /data/Pestopia \
        --out artifacts/data/real \
        --min-per-class 40

Each source declares what it is allowed to contribute. A pest source may only
produce pest (and healthy/background) classes: a fungal or bacterial class
inside a pest dataset is rejected and listed, never trained on. Feeding it in
would put the same condition on both branches of the model with labels from two
different sources, and would corrupt the pest branch's life-stage and
economic-threshold logic - a fungus has neither.

The older corpora are still supported: --dlcpd (nested Crop/Class) and --ap162.

Runs the four steps in order and writes everything it learned to ``--out``:

    manifest.csv        the cleaned, split, taxonomy-mapped dataset
    eda.md / eda.json   the exploratory report and its raw numbers
    ingest_report.json  what mapped, what did not, what was excluded and why
    cleaning_report.json  every dropped image with its reason
    dropped.csv         the dropped images, so a decision can be reviewed

Use ``--dry-run`` to see the ingest and cleaning verdicts without writing a
manifest. Nothing is deleted from the source tree - cleaning only decides which
rows enter the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropguard.data import eda  # noqa: E402
from cropguard.data.clean import (  # noqa: E402
    CleaningConfig,
    cap_class_size,
    clean_records,
    drop_rare_classes,
    inspect_all,
)
from cropguard.data.ingest import (  # noqa: E402
    detect_layout,
    scan_ap162,
    scan_flat,
    scan_nested,
)
from cropguard.data.manifest import (  # noqa: E402
    manifest_summary,
    scan_image_folder,
    stratified_group_split,
    write_manifest,
)
from cropguard.taxonomy import load_taxonomy, save_taxonomy  # noqa: E402

LOGGER = logging.getLogger("prepare")

#: The ten land crops this model targets. Rice is deliberately absent: it is
#: grown flooded and was excluded from scope.
DEFAULT_CROPS = (
    "wheat", "cotton", "maize", "soybean", "potato",
    "tomato", "chilli", "mango", "citrus", "grape",
)

RULE = "=" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--disease", help="disease corpus root, e.g. Plant-Diseases-100k-Labelled-Images")
    p.add_argument("--pest", help="pest corpus root, e.g. Pestopia")
    p.add_argument("--dlcpd", help="unpacked DLCPD-25 root (Crop/Class/*.jpg)")
    p.add_argument("--ap162", help="unpacked AP162 root")
    p.add_argument("--ap162-classes", help="AP162 classes.txt (index<TAB>name)")
    p.add_argument("--extra-source", action="append", default=[],
                   help="additional flat ImageFolder source, treated as mixed (repeatable)")
    p.add_argument("--allow-disease-in-pest-source", action="store_true",
                   help="do NOT reject disease classes found in the pest corpus "
                        "(off by default; they belong to the disease branch)")
    p.add_argument("--out", default="artifacts/data/real")
    p.add_argument("--crops", default=",".join(DEFAULT_CROPS),
                   help="comma separated crops to keep, or 'all'")
    p.add_argument("--min-per-class", type=int, default=40,
                   help="drop classes with fewer images than this after cleaning")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="cap over-represented classes (0 = no cap)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--min-side", type=int, default=64)
    p.add_argument("--near-duplicate-distance", type=int, default=4)
    p.add_argument("--keep-duplicates", action="store_true",
                   help="report duplicates but do not drop them")
    p.add_argument("--workers", type=int, default=8, help="threads for image decoding")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(name)s | %(message)s")

    if not (args.disease or args.pest or args.dlcpd or args.ap162 or args.extra_source):
        p.error("give at least one of --disease, --pest, --dlcpd, --ap162 or --extra-source")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy()
    crops = None if args.crops.strip().lower() == "all" else [
        c.strip() for c in args.crops.split(",") if c.strip()
    ]

    # ---------------- 1. ingest ----------------
    banner("1. INGEST")
    records = []
    reports = {}

    def ingest_source(path: str, name: str, kind: str) -> None:
        """Read one corpus with its kind enforced, whatever its layout."""
        root = Path(path)
        layout = detect_layout(root)
        LOGGER.info("%s layout detected as %r, kind=%s", name, layout, kind)
        if layout == "nested":
            rep = scan_nested(root, taxonomy, source=name, crops=crops, source_kind=kind)
        else:
            rep = scan_flat(root, taxonomy, source=name, source_kind=kind, crops=crops)
        print(rep.format())
        reports[name] = rep.summary()
        records.extend(rep.records)

    pest_kind = "mixed" if args.allow_disease_in_pest_source else "pest"
    if args.disease:
        ingest_source(args.disease, "Plant-Diseases-100k", "disease")
    if args.pest:
        ingest_source(args.pest, "Pestopia", pest_kind)

    if args.dlcpd:
        root = Path(args.dlcpd)
        layout = detect_layout(root)
        LOGGER.info("DLCPD-25 layout detected as %r", layout)
        if layout == "nested":
            rep = scan_nested(root, taxonomy, source="DLCPD-25", crops=crops)
        else:
            recs, unmapped = scan_image_folder(root, taxonomy, source="DLCPD-25")
            from cropguard.data.ingest import IngestReport

            rep = IngestReport(source="DLCPD-25", records=recs, unmapped=unmapped)
        print(rep.format())
        reports["DLCPD-25"] = rep.summary()
        records.extend(rep.records)

    if args.ap162:
        rep = scan_ap162(Path(args.ap162), taxonomy, classes_file=args.ap162_classes)
        print(rep.format())
        reports["AP162"] = rep.summary()
        records.extend(rep.records)

    for src in args.extra_source:
        recs, unmapped = scan_image_folder(src, taxonomy, source=Path(src).name)
        print(f"{src}: {len(recs)} images, {len(unmapped)} unmapped folders")
        if unmapped:
            for name, n in sorted(unmapped.items(), key=lambda kv: -kv[1])[:15]:
                print(f"    {n:>7}  {name}")
        reports[str(src)] = {"images": len(recs), "unmapped_folders": len(unmapped)}
        records.extend(recs)

    (out / "ingest_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    rejected = sum(r.get("wrong_kind_images", 0) for r in reports.values())
    if rejected:
        print(
            f"\nREJECTED {rejected} image(s) whose class does not belong in its source "
            f"(see wrong_kind_detail in ingest_report.json)."
        )
    if not records:
        print("\nnothing ingested - check the paths and the unmapped folders above",
              file=sys.stderr)
        return 1
    print(f"\ningested {len(records):,} images across "
          f"{len({r.class_id for r in records})} classes")

    # ---------------- 2. clean ----------------
    banner("2. CLEAN")
    cfg = CleaningConfig(
        min_side=args.min_side,
        near_duplicate_distance=args.near_duplicate_distance,
        keep_one_per_duplicate_group=not args.keep_duplicates,
        max_workers=args.workers,
    )
    LOGGER.info("inspecting %d images (this reads every file once)", len(records))
    facts = inspect_all([r.path for r in records], cfg)
    report = clean_records(records, cfg, facts=facts)
    print(report.format())

    kept = report.kept
    if args.min_per_class > 0:
        kept, rare = drop_rare_classes(kept, args.min_per_class)
        if rare:
            print(f"\ndropped {len(rare)} class(es) with < {args.min_per_class} images:")
            for cls, n in sorted(rare.items(), key=lambda kv: -kv[1]):
                print(f"    {n:>6}  {cls}")
    if args.max_per_class > 0:
        kept, capped = cap_class_size(kept, args.max_per_class, seed=args.seed)
        if capped:
            print(f"\ncapped {len(capped)} class(es) at {args.max_per_class} images")

    with open(out / "dropped.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reason", "path", "class_id"])
        for reason, items in sorted(report.dropped.items()):
            for path, cls in items:
                w.writerow([reason, path, cls])
    (out / "cleaning_report.json").write_text(
        json.dumps(report.summary(), indent=2), encoding="utf-8"
    )

    # ---------------- 3. split ----------------
    banner("3. SPLIT")
    kept = stratified_group_split(
        kept, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    summary = manifest_summary(kept)
    print(f"images {summary['total']:,} | classes {summary['classes']} | "
          f"splits {summary['splits']}")
    if summary["leaked_groups"]:
        print(f"WARNING: {len(summary['leaked_groups'])} leaked group(s)")
    if summary["classes_missing_val"]:
        print(f"WARNING: {len(summary['classes_missing_val'])} class(es) have no val images")

    # ---------------- 4. EDA ----------------
    banner("4. EDA")
    result = eda.analyse(kept, facts=facts, taxonomy=taxonomy, min_train=args.min_per_class)
    paths = result.save(out)
    for i, rec in enumerate(result.payload["recommendations"], 1):
        print(f"{i}. {rec}")
    print(f"\nfull report: {paths['markdown']}")

    if args.dry_run:
        print("\n--dry-run: no manifest written")
        return 0

    manifest = write_manifest(kept, out / "manifest.csv")
    present = [c for c in taxonomy.class_ids if c in {r.class_id for r in kept}]
    save_taxonomy(taxonomy.subset(present), out / "taxonomy.json")
    print(f"\nmanifest  -> {manifest}")
    print(f"taxonomy  -> {out / 'taxonomy.json'}  ({len(present)} classes)")
    print(f"\nnext:\n  python -m cropguard.train --manifest {manifest} "
          f"--set data.taxonomy={out / 'taxonomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
