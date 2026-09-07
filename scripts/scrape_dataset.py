#!/usr/bin/env python3
"""Collect training images from iNaturalist, driven by the CropGuard taxonomy.

    # see the plan and the realistic yield without downloading anything
    python scripts/scrape_dataset.py --category pest --dry-run

    # then collect
    python scripts/scrape_dataset.py --category pest --per-class 200 \
        --out artifacts/data/scraped --place india

    # feed it into the normal pipeline
    python scripts/prepare_real_dataset.py --pest artifacts/data/scraped \
        --out artifacts/data/pest

Why iNaturalist rather than an image search: identifications are community
verified, every photo carries an explicit licence and photographer, and there
is a documented public API with published rate limits. An image search gives
none of those, and a dataset whose licences are unknown cannot be published.

**Start with --dry-run.** It reports how many observations actually exist per
class, which is the difference between a class worth collecting and one that
will come back with eleven pictures.

Politeness and licensing are enforced by the client, not left to the caller:
requests are rate limited to the published limit and identify themselves, only
reusable licences are downloaded, and every image's photographer and licence
are written to provenance.csv. Keep that file with the dataset - attribution is
a condition of the licences, not paperwork to add later.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropguard.data.scrape import (  # noqa: E402
    PLACE_INDIA,
    INaturalistClient,
    ScrapeConfig,
    classes_to_scrape,
    scrape_classes,
    write_attribution,
    write_provenance,
)
from cropguard.taxonomy import load_taxonomy  # noqa: E402

LOGGER = logging.getLogger("scrape")

PLACES = {"india": PLACE_INDIA, "world": None, "global": None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--category", action="append", default=[],
                   choices=["pest", "disease"],
                   help="which taxonomy categories to collect (repeatable, default: pest)")
    p.add_argument("--classes", help="comma separated class ids, overrides --category")
    p.add_argument("--crops", help="comma separated crops to restrict to")
    p.add_argument("--out", default="artifacts/data/scraped")
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--place", default="india", choices=sorted(PLACES),
                   help="restrict to a region; 'india' gives the morphotypes and "
                        "crop context a model deployed in India will meet")
    p.add_argument("--size", default="large", choices=["small", "medium", "large", "original"])
    p.add_argument("--rate", type=float, default=1.0,
                   help="seconds between requests; iNaturalist asks for >= 1.0, "
                        "and lowering it is not a speed-up, it is a 429")
    p.add_argument("--include-noncommercial", action="store_true",
                   help="also take CC-BY-NC images (research use only; excludes "
                        "the dataset from any commercial deployment)")
    p.add_argument("--quality", default="research", choices=["research", "needs_id", "any"],
                   help="'research' means the identification was community verified")
    p.add_argument("--dry-run", action="store_true",
                   help="report availability per class, download nothing")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(name)s | %(message)s")

    taxonomy = load_taxonomy()
    categories = args.category or ["pest"]
    class_ids = [c.strip() for c in args.classes.split(",")] if args.classes else None
    crops = [c.strip() for c in args.crops.split(",")] if args.crops else None

    classes = classes_to_scrape(taxonomy, categories, class_ids, crops)
    if not classes:
        print("no classes with a usable scientific name matched", file=sys.stderr)
        return 1

    cfg = ScrapeConfig(
        out_dir=Path(args.out),
        per_class=args.per_class,
        place_id=PLACES[args.place],
        quality_grade=args.quality,
        photo_size=args.size,
        rate_seconds=max(1.0, args.rate),
        include_noncommercial=args.include_noncommercial,
    )

    print(f"classes: {len(classes)}   licences: {', '.join(cfg.licences)}")
    print(f"place: {args.place}   quality: {cfg.quality_grade}   "
          f"target: {cfg.per_class}/class   rate: {cfg.rate_seconds}s")
    if args.include_noncommercial:
        print("\nNOTE: --include-noncommercial adds CC-BY-NC images. The resulting\n"
              "dataset is research-use only and cannot back a commercial deployment.")
    print()

    if not args.dry_run:
        estimate = len(classes) * cfg.per_class * cfg.rate_seconds / 60
        print(f"rough lower bound on runtime: {estimate:.0f} min "
              f"(one request per image, plus paging)\n")

    try:
        client = INaturalistClient(cfg)
    except ImportError:
        print("requests is required:  pip install requests", file=sys.stderr)
        return 1

    report = scrape_classes(classes, cfg, client, dry_run=args.dry_run)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scrape_report.json").write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")

    if args.dry_run:
        print("\navailable on iNaturalist (nothing downloaded):\n")
        rows = sorted(report.available.items(), key=lambda kv: -kv[1])
        for cls, n in rows:
            queries = ", ".join(report.queries.get(cls, []))
            flag = "   <-- too few to be worth a class" if n < 50 else ""
            print(f"  {n:>7}  {cls:<34} {queries}{flag}")
        thin = [c for c, n in rows if n < 50]
        print(f"\n{len(rows) - len(thin)} class(es) have >= 50 observations; "
              f"{len(thin)} are too thin to bother with.")
        print(f"report -> {out / 'scrape_report.json'}")
        return 0

    print()
    print(report.format())
    prov = write_provenance(report, out / "provenance.csv")
    attrib = write_attribution(report, out / "ATTRIBUTION.md")
    print(f"\nimages      -> {out}")
    print(f"provenance  -> {prov}   (keep this: attribution is a licence condition)")
    print(f"credits     -> {attrib}")
    print(f"\nnext:\n  python scripts/prepare_real_dataset.py --pest {out} "
          f"--out artifacts/data/pest --min-per-class 40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
