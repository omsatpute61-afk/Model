"""Standalone evaluation, calibration and operating-point selection.

    python -m cropguard.evaluate --run artifacts/runs/cropguard --split test

Produces the artefacts an agronomist or a reviewer actually needs:

* per-class precision / recall / F1 (the rare-pest recall is the number that
  matters, not overall accuracy),
* the confusion table, with **cross-category** confusions called out - a
  pest mistaken for a deficiency changes what the farmer buys,
* a reliability curve and ECE, before and after temperature scaling,
* a coverage/error table so the abstention threshold is a decision with
  evidence behind it rather than a magic 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .data.datasets import build_dataloaders
from .data.transforms import AugmentationConfig
from .metrics import (
    ClassificationReport,
    classification_report,
    confusion_pairs,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    select_threshold,
    selective_risk,
)
from .model_card import ModelCard
from .models.detector import CropGuardNet, ModelConfig
from .taxonomy import CATEGORIES, SEVERITY_LEVELS, Taxonomy, load_taxonomy
from .train import collect_predictions, pick_device, softmax

LOGGER = logging.getLogger("cropguard.evaluate")


def load_run(run_dir: str | Path, device: torch.device | None = None, checkpoint: str = "best.pt"):
    """Rebuild the exact model a run produced, with its card and taxonomy."""
    run_dir = Path(run_dir)
    device = device or pick_device()
    ckpt_path = run_dir / checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    taxonomy_path = run_dir / "taxonomy.json"
    taxonomy = (
        load_taxonomy(taxonomy_path)
        if taxonomy_path.exists()
        else load_taxonomy().subset(ckpt["class_ids"])
    )
    mc = ckpt.get("model_config") or {}
    model = CropGuardNet(
        ModelConfig(
            backbone=mc.get("backbone", "mobilenet_v3_small"),
            num_classes=len(ckpt["class_ids"]),
            num_categories=mc.get("num_categories", len(CATEGORIES)),
            num_severity=mc.get("num_severity", len(SEVERITY_LEVELS)),
            embedding_dim=mc.get("embedding_dim", 128),
            dropout=mc.get("dropout", 0.2),
            pretrained=False,           # weights come from the checkpoint
            image_size=mc.get("image_size", 224),
        )
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    card_path = run_dir / "model_card.json"
    card = ModelCard.load(card_path) if card_path.exists() else None
    return model, taxonomy, card, ckpt


def evaluate_run(
    run_dir: str | Path,
    manifest: str | Path | None = None,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
    recalibrate: bool = False,
    max_selective_error: float = 0.10,
    min_coverage: float = 0.50,
    min_threshold: float = 0.30,
    checkpoint: str = "best.pt",
    update_card: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    device = device or pick_device()
    model, taxonomy, card, ckpt = load_run(run_dir, device, checkpoint)

    train_cfg = ckpt.get("train_config", {})
    manifest = manifest or train_cfg.get("data", {}).get("manifest")
    if manifest is None:
        raise ValueError("no manifest given and none recorded in the checkpoint")
    image_size = (card.preprocess.image_size if card else train_cfg.get("data", {}).get("image_size", 224))

    loaders, loader_tax = build_dataloaders(
        manifest,
        taxonomy=taxonomy,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        aug=AugmentationConfig(image_size=image_size, strength=0.0),
        balanced_sampling=False,
    )
    if loader_tax.class_ids != taxonomy.class_ids:
        raise ValueError(
            "manifest classes do not match the trained label order - "
            f"model has {len(taxonomy)}, manifest has {len(loader_tax)}"
        )
    if split not in loaders:
        raise ValueError(f"split {split!r} not present; have {sorted(loaders)}")

    temperature = card.policy.temperature if card else 1.0
    if recalibrate and "val" in loaders:
        val_raw = collect_predictions(model, loaders["val"], device)
        temperature = fit_temperature(val_raw["logits"], val_raw["labels"])
        LOGGER.info("recalibrated temperature: %.3f", temperature)

    raw = collect_predictions(model, loaders[split], device)
    y = raw["labels"]
    probs_raw = softmax(raw["logits"], 1.0)
    probs = softmax(raw["logits"], temperature)

    report = classification_report(y, probs.argmax(1), taxonomy.class_ids, probs=probs)
    cat_probs = softmax(raw["category_logits"], temperature)
    cat_report = classification_report(
        raw["categories"], cat_probs.argmax(1), list(CATEGORIES), probs=cat_probs
    )

    threshold, operating = select_threshold(
        probs, y, max_selective_error=max_selective_error,
        min_coverage=min_coverage, min_threshold=min_threshold,
    )
    categories = [taxonomy[c].category for c in taxonomy.class_ids]

    result = {
        "run": str(run_dir),
        "split": split,
        "samples": int(len(y)),
        "temperature": temperature,
        "label": report.to_dict(),
        "category": {
            "accuracy": cat_report.accuracy,
            "macro_f1": cat_report.macro_f1,
            "per_class": cat_report.to_dict()["per_class"],
        },
        "calibration": {
            "ece_uncalibrated": expected_calibration_error(probs_raw, y),
            "ece_calibrated": expected_calibration_error(probs, y),
            "reliability": reliability_bins(probs, y),
        },
        "operating_point": operating,
        "suggested_threshold": threshold,
        "selective_risk": selective_risk(probs, y),
        "confusions": confusion_pairs(
            report.confusion, taxonomy.class_ids, top=15, categories=categories
        ),
        "worst_classes": [
            {"class_id": c.class_id, "f1": c.f1, "recall": c.recall, "support": c.support}
            for c in report.worst_classes(8)
        ],
        "cross_category_error_rate": _cross_category_rate(
            y, probs.argmax(1), categories
        ),
    }

    if update_card and card is not None:
        # Re-tuning the operating point is a routine, retraining-free decision:
        # a district that cannot afford wrong spray advice raises the error
        # budget's strictness, and the threshold moves. Persist it so the edge
        # bundle inherits the new policy on the next export.
        card.policy.temperature = temperature
        card.policy.min_confidence = threshold
        card.metrics = {**(card.metrics or {}), f"{split}_reeval": {
            "accuracy": report.accuracy, "macro_f1": report.macro_f1,
            "operating_point": operating,
        }}
        card.validate(num_outputs=len(taxonomy))
        card.save(run_dir / "model_card.json")
        export_card = run_dir / "export" / "model_card.json"
        if export_card.exists():
            card.save(export_card)
        LOGGER.info(
            "updated model card: threshold=%.2f temperature=%.3f", threshold, temperature
        )
        result["card_updated"] = True
    else:
        result["card_updated"] = False

    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_report.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _write_confusion_csv(out_dir / f"{split}_confusion.csv", report)
    _write_predictions_csv(out_dir / f"{split}_predictions.csv", raw, probs, taxonomy)
    LOGGER.info("wrote evaluation artefacts to %s", out_dir)
    return result


def _cross_category_rate(y_true: np.ndarray, y_pred: np.ndarray, categories: list[str]) -> float:
    """Share of predictions that land in the wrong *category*.

    This is the error rate that actually costs money: within-category
    confusions usually lead to the same intervention, cross-category ones do
    not.
    """
    if len(y_true) == 0:
        return 0.0
    cats = np.array(categories)
    return float((cats[y_true] != cats[y_pred]).mean())


def _write_confusion_csv(path: Path, report: ClassificationReport) -> None:
    cm = report.confusion
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\predicted", *report.class_ids])
        for i, cid in enumerate(report.class_ids):
            w.writerow([cid, *cm[i].tolist()])


def _write_predictions_csv(path: Path, raw: dict, probs: np.ndarray, taxonomy: Taxonomy) -> None:
    """Per-image predictions - the file you sort by confidence to find label errors."""
    ids = taxonomy.class_ids
    order = np.argsort(-probs, axis=1)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true", "predicted", "confidence", "correct", "top2", "top2_conf"])
        for i, p in enumerate(raw["paths"]):
            t, pred = ids[raw["labels"][i]], ids[order[i, 0]]
            w.writerow(
                [
                    p, t, pred, f"{probs[i, order[i, 0]]:.4f}", int(t == pred),
                    ids[order[i, 1]] if probs.shape[1] > 1 else "",
                    f"{probs[i, order[i, 1]]:.4f}" if probs.shape[1] > 1 else "",
                ]
            )


def format_summary(result: dict) -> str:
    lab = result["label"]
    lines = [
        f"split={result['split']}  n={result['samples']}  T={result['temperature']:.3f}",
        f"accuracy          {lab['accuracy']:.4f}",
        f"macro F1          {lab['macro_f1']:.4f}",
        f"balanced accuracy {lab['balanced_accuracy']:.4f}",
        f"top-3             {lab['top_k'].get('3', float('nan')):.4f}",
        f"category accuracy {result['category']['accuracy']:.4f}",
        f"cross-category error {result['cross_category_error_rate']:.4f}",
        f"ECE  {result['calibration']['ece_uncalibrated']:.4f} -> "
        f"{result['calibration']['ece_calibrated']:.4f} (after temperature scaling)",
        f"suggested confidence threshold {result['suggested_threshold']:.2f} "
        f"(coverage {result['operating_point']['coverage']:.2f}, "
        f"error-when-answering {result['operating_point']['selective_error']:.3f})",
        "",
        "weakest classes:",
    ]
    for c in result["worst_classes"]:
        lines.append(f"  {c['class_id']:<34} f1={c['f1']:.3f} recall={c['recall']:.3f} n={c['support']}")
    cross = [c for c in result["confusions"] if c.get("cross_category")]
    if cross:
        lines.append("")
        lines.append("cross-category confusions (these change the recommended action):")
        for c in cross[:6]:
            lines.append(
                f"  {c['true']} -> {c['predicted']}  x{c['count']} ({c['share_of_true']:.0%} of true)"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cropguard.evaluate")
    p.add_argument("--run", required=True, help="run directory produced by cropguard.train")
    p.add_argument("--manifest", help="override the manifest recorded in the checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--recalibrate", action="store_true", help="refit temperature on val")
    p.add_argument("--update-card", action="store_true",
                   help="write the chosen temperature and threshold back into the model card")
    p.add_argument("--max-selective-error", type=float, default=0.10)
    p.add_argument("--min-coverage", type=float, default=0.50)
    p.add_argument("--min-threshold", type=float, default=0.30,
                   help="floor on the abstention threshold, to preserve OOD rejection")
    p.add_argument("--device")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(message)s")
    result = evaluate_run(
        args.run, manifest=args.manifest, split=args.split, batch_size=args.batch_size,
        num_workers=args.num_workers, device=pick_device(args.device),
        recalibrate=args.recalibrate, max_selective_error=args.max_selective_error,
        min_coverage=args.min_coverage, min_threshold=args.min_threshold,
        checkpoint=args.checkpoint, update_card=args.update_card,
    )
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
