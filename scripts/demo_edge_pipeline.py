#!/usr/bin/env python3
"""End-to-end field simulation, from photo to the SMS a farmer receives.

    python scripts/demo_edge_pipeline.py --bundle artifacts/runs/demo/export

Walks the whole decision path so the behaviour can be inspected without a
device:

1. diagnose a set of leaf photos on the exported edge model,
2. show what the device does with an image it does not understand,
3. accumulate the accepted detections into a pest-pressure trend,
4. add environmental sensor readings and compute weather-driven infection risk,
5. merge the two and print the alerts, including the 160-character SMS.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropguard.early_warning import (  # noqa: E402
    PestPressureTracker,
    WeatherReading,
    combined_risk,
    infection_risk,
)
from cropguard.edge import EdgeClassifier  # noqa: E402

RULE = "-" * 78


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="exported bundle directory")
    p.add_argument("--images", help="glob of leaf photos (default: the bundle's training data)")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--field-id", default="plot-7")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    clf = EdgeClassifier(args.bundle)
    header("DEVICE")
    print(f"bundle       {args.bundle}")
    print(f"backend      {clf.backend} ({clf._model_path.name})")
    print(f"classes      {len(clf.card.class_ids)}")
    print(f"input        {clf.card.preprocess.image_size}px")
    print(f"threshold    {clf.card.policy.min_confidence:.2f} (temperature {clf.card.policy.temperature:.3f})")
    if clf.ood is not None and clf.ood.enabled:
        print(f"novelty      Mahalanobis, reject above {clf.ood.threshold:.0f}")
    else:
        print("novelty      DISABLED - unfamiliar images will be given a diagnosis")

    pattern = args.images or "artifacts/data/synth_v2/*/*.jpg"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\nno images matched {pattern!r}", file=sys.stderr)
        return 1
    random.seed(args.seed)
    sample = random.sample(files, min(args.limit, len(files)))

    header("1. LEAF DIAGNOSIS")
    tracker = PestPressureTracker()
    now = datetime.now(timezone.utc)
    for i, path in enumerate(sample):
        d = clf.diagnose(path)
        truth = Path(path).parent.name
        mark = "ok " if d.class_id == truth else "   "
        status = f"{d.class_id}" if d.accepted else "ABSTAINED"
        print(f"{mark}{Path(path).name[:38]:38s} -> {status:26s} p={d.confidence:.2f} {d.latency_ms:5.1f}ms")
        if d.accepted and d.advisory and d.advisory.urgency in ("warning", "critical"):
            print(f"     {d.advisory.headline}")
            print(f"     SMS: {d.advisory.to_sms()}")
        tracker.add_diagnosis(
            d, field_id=args.field_id, timestamp=now - timedelta(days=(args.limit - i) % 6)
        )

    header("2. AN IMAGE THE MODEL DOES NOT UNDERSTAND")
    rng = np.random.default_rng(args.seed)
    noise = Image.fromarray(rng.integers(0, 255, (320, 320, 3)).astype("uint8"))
    d = clf.diagnose(noise)
    print(f"result       {'ABSTAINED' if not d.accepted else d.class_id}")
    print(f"confidence   {d.confidence:.3f}" + (f"   novelty {d.novelty:.0f}" if d.novelty else ""))
    if d.reason:
        print(f"reason       {d.reason}")
    if d.advisory:
        print(f"advice       {d.advisory.message}")

    header("3. PEST PRESSURE OVER TIME")
    camera_alerts = tracker.evaluate(field_id=args.field_id, now=now)
    if not camera_alerts:
        print("no actionable problems accumulated in the window")
    for a in camera_alerts:
        print(f"[{a.level.upper():8s}] {a.display_name}  ({a.detections} detections, trend {a.trend})")
        for e in a.evidence:
            print(f"           - {e}")

    header("4. ENVIRONMENTAL SENSORS -> INFECTION RISK")
    readings = [
        WeatherReading(now - timedelta(hours=h), temp_c=16.5, humidity_pct=93.0,
                       leaf_wetness_hours=1.0, rainfall_mm=0.4)
        for h in range(14, 0, -1)
    ]
    print("simulated: 14 h at 16.5 C, RH 93%, leaf wet throughout (a classic blight window)")
    weather_alerts = infection_risk(readings, field_id=args.field_id)
    for a in weather_alerts[:4]:
        print(f"[{a.level.upper():8s}] {a.display_name}: {a.message}")

    header("5. COMBINED FIELD ADVISORY")
    for a in combined_risk(camera_alerts, weather_alerts)[:6]:
        print(f"[{a.level.upper():8s}] {a.title}")
        print(f"           {a.message}")
        print(f"           SMS: {a.to_sms()}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
