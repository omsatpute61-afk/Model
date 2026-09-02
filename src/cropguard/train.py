"""Training entry point for the pest / disease model.

Run:

    python -m cropguard.train --config configs/default.yaml
    python -m cropguard.train --manifest artifacts/data/manifest.csv --epochs 20

What the loop does beyond the obvious forward/backward:

* **Warm start.** The trunk is frozen for the first epoch(s) so the randomly
  initialised heads do not send garbage gradients through pretrained features.
* **Discriminative LR + cosine schedule with warmup.** Standard, but it is what
  makes transfer learning on a few thousand field images actually converge.
* **EMA weights.** Shipped in preference to the raw weights.
* **Selection on macro F1, not accuracy.** See ``cropguard.metrics``.
* **Calibration and threshold selection at the end**, written into the model
  card, so the edge runtime inherits an honest confidence scale and a
  defensible abstention point rather than a hardcoded 0.5.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data.datasets import (
    IGNORE_INDEX,
    build_dataloaders,
    build_eval_loader,
    mixup_cutmix,
)
from .data.manifest import class_weights, manifest_summary, read_manifest
from .data.transforms import IMAGENET_MEAN, IMAGENET_STD, AugmentationConfig
from .metrics import (
    classification_report,
    confusion_pairs,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
)
from .model_card import DecisionPolicy, ModelCard, PreprocessSpec
from .ood import OOD_FILENAME, MahalanobisOOD
from .models.backbones import freeze_backbone, unfreeze_last_blocks
from .models.detector import CropGuardNet, ModelConfig, ModelEMA, MultiHeadLoss
from .taxonomy import CATEGORIES, SEVERITY_LEVELS, Taxonomy, load_taxonomy, save_taxonomy

LOGGER = logging.getLogger("cropguard.train")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Slower, but two runs of the same config produce the same checkpoint -
        # which matters when a model is going to be defended to an agronomist.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - not a git checkout, or no git
        return ""


def cosine_lr(step: int, total: int, warmup: int, min_scale: float) -> float:
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_scale + (1 - min_scale) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# evaluation pass (shared by training and cropguard.evaluate)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Run a split and return raw logits/targets for downstream metrics."""
    model.eval()
    logits, cat_logits, sev_logits, embeddings = [], [], [], []
    labels, cats, sevs, paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        out = model(images)
        logits.append(out.label_logits.float().cpu().numpy())
        cat_logits.append(out.category_logits.float().cpu().numpy())
        sev_logits.append(out.severity_logits.float().cpu().numpy())
        embeddings.append(out.embedding.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        cats.append(batch["category"].numpy())
        sevs.append(batch["severity"].numpy())
        paths.extend(batch["path"])
    if not logits:
        raise ValueError("evaluation loader produced no batches")
    return {
        "logits": np.concatenate(logits),
        "category_logits": np.concatenate(cat_logits),
        "severity_logits": np.concatenate(sev_logits),
        "embeddings": np.concatenate(embeddings),
        "labels": np.concatenate(labels),
        "categories": np.concatenate(cats),
        "severities": np.concatenate(sevs),
        "paths": np.array(paths),
    }


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    taxonomy: Taxonomy,
    temperature: float = 1.0,
) -> tuple[dict, dict[str, np.ndarray]]:
    raw = collect_predictions(model, loader, device)
    probs = softmax(raw["logits"], temperature)
    report = classification_report(
        raw["labels"], probs.argmax(1), taxonomy.class_ids, probs=probs
    )
    cat_probs = softmax(raw["category_logits"], temperature)
    cat_report = classification_report(
        raw["categories"], cat_probs.argmax(1), list(CATEGORIES), probs=cat_probs
    )

    sev_mask = raw["severities"] != IGNORE_INDEX
    sev_acc = (
        float(
            (raw["severity_logits"][sev_mask].argmax(1) == raw["severities"][sev_mask]).mean()
        )
        if sev_mask.any()
        else None
    )

    metrics = {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "balanced_accuracy": report.balanced_accuracy,
        "weighted_f1": report.weighted_f1,
        "top_3": report.top_k.get(3),
        "category_accuracy": cat_report.accuracy,
        "category_macro_f1": cat_report.macro_f1,
        "severity_accuracy": sev_acc,
        "ece": expected_calibration_error(probs, raw["labels"]),
    }
    raw["probs"] = probs
    raw["report"] = report  # type: ignore[assignment]
    return metrics, raw


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig, device: torch.device | None = None) -> dict:
    set_seed(cfg.seed, cfg.deterministic)
    device = device or pick_device()
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    LOGGER.info("device=%s output=%s", device, out_dir)

    # -- data ------------------------------------------------------------
    base_tax = load_taxonomy(cfg.data.taxonomy) if cfg.data.taxonomy else load_taxonomy()
    aug = AugmentationConfig(image_size=cfg.data.image_size, strength=cfg.data.aug_strength)
    loaders, taxonomy = build_dataloaders(
        cfg.data.manifest,
        taxonomy=base_tax,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        aug=aug,
        balanced_sampling=cfg.data.balanced_sampling,
    )
    records = read_manifest(cfg.data.manifest)
    summary = manifest_summary(records)
    LOGGER.info(
        "data: %d images, %d classes, splits=%s",
        summary["total"], len(taxonomy), summary["splits"],
    )
    if summary["leaked_groups"]:
        LOGGER.warning(
            "%d group(s) appear in more than one split - validation is optimistic",
            len(summary["leaked_groups"]),
        )
    if summary["degenerate_grouping"]:
        LOGGER.warning(
            "classes with too few split groups: %s", summary["degenerate_grouping"][:10]
        )
    save_taxonomy(taxonomy, out_dir / "taxonomy.json")

    # -- model -----------------------------------------------------------
    model = CropGuardNet(
        ModelConfig(
            backbone=cfg.model.backbone,
            num_classes=len(taxonomy),
            num_categories=len(CATEGORIES),
            num_severity=len(SEVERITY_LEVELS),
            embedding_dim=cfg.model.embedding_dim,
            dropout=cfg.model.dropout,
            pretrained=cfg.model.pretrained,
            image_size=cfg.data.image_size,
        )
    ).to(device)
    LOGGER.info(
        "model: %s, %.2fM params, pretrained=%s",
        cfg.model.backbone, model.num_parameters() / 1e6, model.backbone_info.pretrained,
    )

    if not model.backbone_info.pretrained:
        # The warm-start schedule exists to protect *pretrained* features. On a
        # randomly initialised trunk, freezing it starves the heads and a 0.1x
        # trunk LR just slows convergence to a crawl - which looks exactly like
        # a broken training loop. Neither is ever right here, so override both.
        if cfg.optim.freeze_epochs or cfg.optim.backbone_lr_scale != 1.0:
            LOGGER.warning(
                "backbone is not pretrained: disabling the frozen warm-start "
                "(was %d epochs) and training the trunk at full LR (was %.2gx). "
                "Expect to need materially more epochs than a fine-tuning run.",
                cfg.optim.freeze_epochs, cfg.optim.backbone_lr_scale,
            )
        cfg.optim.freeze_epochs = 0
        cfg.optim.backbone_lr_scale = 1.0

    train_records = [r for r in records if r.split == "train"]
    weights = torch.tensor(
        class_weights(train_records, taxonomy.class_ids), dtype=torch.float32, device=device
    )
    criterion = MultiHeadLoss(
        category_weight=cfg.optim.category_weight,
        severity_weight=cfg.optim.severity_weight,
        class_weights=None if cfg.data.balanced_sampling else weights,
        label_smoothing=cfg.optim.label_smoothing,
        use_focal=cfg.optim.use_focal,
        focal_gamma=cfg.optim.focal_gamma,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.param_groups(cfg.optim.lr, cfg.optim.backbone_lr_scale),
        weight_decay=cfg.optim.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimiser.param_groups]

    use_amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = steps_per_epoch * cfg.optim.epochs
    warmup_steps = steps_per_epoch * cfg.optim.warmup_epochs

    # Scale the EMA ramp to the run: a fixed constant leaves short runs stuck
    # on the ramp, where the EMA copy is indistinguishable from the live model.
    ema = (
        ModelEMA(model, cfg.optim.ema_decay, tau=max(50.0, total_steps / 10.0))
        if cfg.optim.use_ema
        else None
    )

    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": -1}
    patience = 0
    global_step = 0
    started = time.time()

    for epoch in range(cfg.optim.epochs):
        # Warm start: heads first, then release the trunk.
        if epoch < cfg.optim.freeze_epochs:
            freeze_backbone(model.trunk, True)
        elif epoch == cfg.optim.freeze_epochs:
            if cfg.optim.unfreeze_blocks > 0:
                n = unfreeze_last_blocks(model.trunk, cfg.optim.unfreeze_blocks)
                LOGGER.info("epoch %d: unfroze last %d trunk blocks", epoch, n)
            else:
                freeze_backbone(model.trunk, False)
                LOGGER.info("epoch %d: unfroze full trunk", epoch)

        model.train()
        running = {"total": 0.0, "label": 0.0, "category": 0.0, "severity": 0.0}
        seen = 0
        for batch in loaders["train"]:
            lr_scale = cosine_lr(global_step, total_steps, warmup_steps, cfg.optim.min_lr_scale)
            for group, base in zip(optimiser.param_groups, base_lrs):
                group["lr"] = base * lr_scale

            images = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            category = batch["category"].to(device, non_blocking=True)
            severity = batch["severity"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            soft = None
            if cfg.optim.mixup_alpha > 0:
                images, soft = mixup_cutmix(
                    images, label, len(taxonomy),
                    alpha=cfg.optim.mixup_alpha,
                    label_smoothing=cfg.optim.label_smoothing,
                )

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss, parts = criterion(out, label, category, severity, soft, valid)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            if ema is not None:
                ema.update(model)

            bs = images.size(0)
            seen += bs
            for k in running:
                running[k] += parts[k] * bs
            global_step += 1

        train_loss = {k: v / max(1, seen) for k, v in running.items()}
        eval_model = ema.module if ema is not None else model
        val_metrics, _ = evaluate_split(eval_model, loaders.get("val", loaders["train"]), device, taxonomy)

        row = {
            "epoch": epoch,
            "lr": optimiser.param_groups[0]["lr"],
            "train_loss": train_loss["total"],
            "train_label_loss": train_loss["label"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        LOGGER.info(
            "epoch %2d/%d loss %.4f | val acc %.3f macroF1 %.3f cat %.3f ece %.3f",
            epoch + 1, cfg.optim.epochs, train_loss["total"],
            val_metrics["accuracy"], val_metrics["macro_f1"],
            val_metrics["category_accuracy"], val_metrics["ece"],
        )

        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val_metrics["macro_f1"], "epoch": epoch, **val_metrics}
            patience = 0
            _save_checkpoint(out_dir / "best.pt", eval_model, taxonomy, cfg, best)
        else:
            patience += 1
            if cfg.optim.early_stopping_patience and patience >= cfg.optim.early_stopping_patience:
                LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break

        _write_history(out_dir / "history.json", history)

    _save_checkpoint(out_dir / "last.pt", ema.module if ema else model, taxonomy, cfg, best)

    # -- calibrate + choose the abstention threshold ----------------------
    best_model = _load_into(model, out_dir / "best.pt", device)
    temperature, policy_row = 1.0, {}
    if "val" in loaders and cfg.policy.calibrate:
        _, val_raw = evaluate_split(best_model, loaders["val"], device, taxonomy)
        temperature = fit_temperature(val_raw["logits"], val_raw["labels"])
        cal_probs = softmax(val_raw["logits"], temperature)
        threshold, policy_row = select_threshold(
            cal_probs, val_raw["labels"],
            max_selective_error=cfg.policy.max_selective_error,
            min_coverage=cfg.policy.min_coverage,
            min_threshold=cfg.policy.min_threshold,
        )
        LOGGER.info(
            "calibration: T=%.3f threshold=%.2f coverage=%.2f selective_error=%.3f",
            temperature, threshold, policy_row["coverage"], policy_row["selective_error"],
        )
    else:
        threshold = 0.55

    # -- out-of-distribution detector -------------------------------------
    # Fitted last, on the *final* weights, using clean (un-augmented) training
    # embeddings. Without this the system cannot tell a novel input from a
    # familiar one and will answer a photo of the sky with a disease name.
    ood_stats = None
    if cfg.policy.fit_ood:
        train_eval = build_eval_loader(
            cfg.data.manifest, taxonomy, "train",
            image_size=cfg.data.image_size,
            batch_size=max(16, cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
        )
        if train_eval is not None:
            tr_raw = collect_predictions(best_model, train_eval, device)
            detector = MahalanobisOOD.fit(
                tr_raw["embeddings"], tr_raw["labels"], len(taxonomy), list(taxonomy.class_ids)
            )
            if "val" in loaders:
                _, val_for_ood = evaluate_split(best_model, loaders["val"], device, taxonomy)
                stats = detector.calibrate(val_for_ood["embeddings"], cfg.policy.ood_percentile)
            else:
                stats = detector.calibrate(tr_raw["embeddings"], cfg.policy.ood_percentile)
            detector.save(out_dir / OOD_FILENAME)
            ood_stats = stats.to_dict()
            LOGGER.info(
                "OOD detector: threshold=%.1f (p%.1f), rejects %.1f%% of in-distribution val",
                stats.threshold, stats.percentile, 100 * stats.val_reject_rate,
            )

    final_metrics = {"val": best}
    for split in ("val", "test"):
        if split in loaders:
            m, raw = evaluate_split(best_model, loaders[split], device, taxonomy, temperature)
            final_metrics[split] = m
            if split == "test":
                report = raw["report"]
                LOGGER.info("test macro F1 %.3f acc %.3f", m["macro_f1"], m["accuracy"])
                LOGGER.info("weakest classes:\n%s", report.format_table(max_rows=8))
                final_metrics["test_confusions"] = confusion_pairs(
                    report.confusion, taxonomy.class_ids, top=10,
                    categories=[taxonomy[c].category for c in taxonomy.class_ids],
                )

    card = ModelCard(
        backbone=cfg.model.backbone,
        class_ids=list(taxonomy.class_ids),
        categories=list(CATEGORIES),
        severity_levels=list(SEVERITY_LEVELS),
        preprocess=PreprocessSpec(
            image_size=cfg.data.image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD
        ),
        policy=DecisionPolicy(min_confidence=threshold, temperature=temperature),
        pretrained=model.backbone_info.pretrained,
        trained_on={
            "manifest": str(cfg.data.manifest),
            "images": summary["total"],
            "splits": summary["splits"],
            "classes": len(taxonomy),
        },
        metrics={**final_metrics, "operating_point": policy_row, "ood": ood_stats},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=git_commit(),
        notes=f"run={cfg.run_name}; best epoch={best['epoch']}",
    )
    card.validate(num_outputs=len(taxonomy))
    card.save(out_dir / "model_card.json")
    _write_history(out_dir / "history.json", history)

    LOGGER.info("done in %.1fs -> %s", time.time() - started, out_dir)
    return {
        "output_dir": str(out_dir),
        "best": best,
        "metrics": final_metrics,
        "temperature": temperature,
        "threshold": threshold,
        "history": history,
    }


def _save_checkpoint(path: Path, model, taxonomy: Taxonomy, cfg: TrainConfig, metrics: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model.cfg) if hasattr(model, "cfg") else {},
            "class_ids": list(taxonomy.class_ids),
            "categories": list(CATEGORIES),
            "severity_levels": list(SEVERITY_LEVELS),
            "train_config": cfg.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def _load_into(model: CropGuardNet, path: Path, device: torch.device) -> CropGuardNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def _write_history(path: Path, history: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cropguard.train", description="Train the CropGuard pest/disease model"
    )
    p.add_argument("--config", type=str, help="YAML/JSON config file")
    p.add_argument("--manifest", type=str, help="dataset manifest CSV")
    p.add_argument("--run-name", type=str)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--backbone", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="override any config value, e.g. --set optim.mixup_alpha=0.2",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.manifest:
        cfg.data.manifest = args.manifest
    if args.run_name:
        cfg.run_name = args.run_name
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.data.image_size = args.image_size
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.no_pretrained:
        cfg.model.pretrained = False
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    return cfg.apply_overrides(args.overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = config_from_args(args)
    result = train(cfg, device=pick_device(args.device))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
