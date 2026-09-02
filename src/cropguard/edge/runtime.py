"""On-device inference: image in, farmer advisory out.

Runs with **no torch installed**. The default backend is ONNX Runtime, which
is a ~15 MB wheel that installs on a Raspberry Pi; torch is used only if the
caller explicitly asks for it (handy on a workstation, hopeless on a Pi).

The class this module exists for is ``EdgeClassifier.diagnose``: it does not
return an argmax. It returns a decision - including the decision to abstain -
with the calibrated confidence, the coarse category fallback, and the advisory
text attached. Everything downstream (display, SMS, the early-warning tracker)
consumes that one object.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..advisory import Advisory, AdvisoryEngine
from ..model_card import CARD_FILENAME, ModelCard
from ..ood import OOD_FILENAME, MahalanobisOOD
from ..taxonomy import CATEGORIES, SEVERITY_LEVELS, Taxonomy, load_taxonomy
from .preprocess import preprocess_image, tile_image

LOGGER = logging.getLogger("cropguard.edge")


@dataclass
class Diagnosis:
    """One decision about one image."""

    class_id: str
    display_name: str
    category: str
    confidence: float
    accepted: bool                       # False -> abstained, class_id is "unknown"
    severity: str | None = None
    severity_confidence: float | None = None
    topk: list[tuple[str, float]] = field(default_factory=list)
    novelty: float | None = None          # Mahalanobis distance in feature space
    is_novel: bool = False                # True -> unlike anything in training
    category_from_head: str | None = None
    category_confidence: float | None = None
    latency_ms: float = 0.0
    reason: str = ""
    box: tuple[int, int, int, int] | None = None
    advisory: Advisory | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "class_id": self.class_id,
            "display_name": self.display_name,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "accepted": self.accepted,
            "severity": self.severity,
            "severity_confidence": (
                round(self.severity_confidence, 4) if self.severity_confidence is not None else None
            ),
            "topk": [(c, round(p, 4)) for c, p in self.topk],
            "novelty": round(self.novelty, 3) if self.novelty is not None else None,
            "is_novel": self.is_novel,
            "category_from_head": self.category_from_head,
            "category_confidence": (
                round(self.category_confidence, 4) if self.category_confidence is not None else None
            ),
            "latency_ms": round(self.latency_ms, 2),
            "reason": self.reason,
            "box": list(self.box) if self.box else None,
        }
        if self.advisory is not None:
            d["advisory"] = self.advisory.to_dict()
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


class EdgeClassifier:
    """Load an exported bundle and diagnose images.

    A "bundle" is the directory ``cropguard.export`` writes: a model file, its
    ``model_card.json``, and optionally ``taxonomy.json``. The card is not
    optional - without it there is no label order and no preprocessing spec,
    and a confident wrong answer is worse than no answer.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        model_file: str | None = None,
        backend: str = "auto",
        taxonomy: Taxonomy | None = None,
        advisory: AdvisoryEngine | None = None,
        min_confidence: float | None = None,
        num_threads: int | None = None,
        use_ood: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)
        self.card = ModelCard.load(self.bundle_dir / CARD_FILENAME)
        self.card.validate()

        tax_path = self.bundle_dir / "taxonomy.json"
        base = taxonomy or (load_taxonomy(tax_path) if tax_path.exists() else load_taxonomy())
        # The card's class order is authoritative; the taxonomy only supplies
        # the agronomic metadata for those ids.
        self.taxonomy = base.subset(self.card.class_ids) if not taxonomy else taxonomy
        self.advisory_engine = advisory or AdvisoryEngine(taxonomy=self.taxonomy)

        if min_confidence is not None:
            self.card.policy.min_confidence = min_confidence

        self.ood: MahalanobisOOD | None = None
        ood_path = self.bundle_dir / OOD_FILENAME
        if use_ood and ood_path.exists():
            try:
                self.ood = MahalanobisOOD.load(ood_path)
                if not self.ood.enabled:
                    LOGGER.warning(
                        "OOD detector is present but disabled: %s",
                        self.ood.stats.get("disabled_reason", "unknown reason"),
                    )
            except Exception as exc:  # noqa: BLE001 - never block inference on this
                LOGGER.warning("could not load %s (%s); OOD rejection disabled", OOD_FILENAME, exc)
        elif use_ood:
            LOGGER.warning(
                "no %s in %s - confidence thresholding alone does not reject "
                "unfamiliar images (a photo of the sky will be given a disease name)",
                OOD_FILENAME, self.bundle_dir,
            )

        self._model_path, self.backend = self._resolve_backend(model_file, backend)
        self._session = None
        self._torch_model = None
        self._num_threads = num_threads
        self._load()
        LOGGER.info(
            "loaded %s (%s backend, %d classes)",
            self._model_path.name, self.backend, len(self.card.class_ids),
        )

    # -- loading ---------------------------------------------------------
    def _resolve_backend(self, model_file: str | None, backend: str) -> tuple[Path, str]:
        if model_file:
            path = self.bundle_dir / model_file
            if not path.exists():
                raise FileNotFoundError(path)
            inferred = "onnx" if path.suffix == ".onnx" else "torch"
            return path, (backend if backend != "auto" else inferred)

        # Prefer the quantised model: on an ARM CPU it is the difference
        # between a usable and an unusable response time.
        for name, be in (
            ("cropguard.int8.onnx", "onnx"),
            ("cropguard.onnx", "onnx"),
            ("cropguard.ptl", "torch"),
            ("best.pt", "torch"),
        ):
            candidate = self.bundle_dir / name
            if candidate.exists() and (backend in ("auto", be)):
                return candidate, be
        raise FileNotFoundError(
            f"no model artefact in {self.bundle_dir}. Run cropguard.export first."
        )

    def _load(self) -> None:
        if self.backend == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "onnxruntime is required for the onnx backend "
                    "(pip install onnxruntime)"
                ) from exc
            opts = ort.SessionOptions()
            if self._num_threads:
                opts.intra_op_num_threads = self._num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self._model_path), opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            n_out = self._session.get_outputs()[0].shape[-1]
            if isinstance(n_out, int):
                self.card.validate(num_outputs=n_out)
        else:
            import torch

            if self._model_path.suffix == ".ptl":
                self._torch_model = torch.jit.load(str(self._model_path))
                self._torch_model.eval()
            else:
                from ..evaluate import load_run

                model, _, _, _ = load_run(self.bundle_dir, torch.device("cpu"))
                from ..models.detector import ExportWrapper

                self._torch_model = ExportWrapper(model, self.card.policy.temperature).eval()

    # -- inference -------------------------------------------------------
    def _run(
        self, batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns (label_probs, category_probs, severity_probs, embedding)."""
        if self.backend == "onnx":
            outs = self._session.run(None, {self._input_name: batch.astype(np.float32)})
        else:
            import torch

            with torch.no_grad():
                outs = [t.numpy() for t in self._torch_model(torch.from_numpy(batch))]
        embedding = outs[3] if len(outs) > 3 else None
        while len(outs) < 3:
            outs.append(np.zeros((batch.shape[0], 1), dtype=np.float32))
        return outs[0], outs[1], outs[2], embedding

    def _novelty(self, embedding: np.ndarray | None, index: int) -> tuple[float | None, bool]:
        if self.ood is None or embedding is None or not self.ood.enabled:
            return None, False
        try:
            score = float(self.ood.score(embedding[index : index + 1])[0])
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("novelty scoring failed: %s", exc)
            return None, False
        return score, bool(score > self.ood.threshold)

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = np.concatenate(
            [preprocess_image(im, self.card.preprocess) for im in images], axis=0
        )
        return self._run(batch)[0]

    def diagnose(
        self,
        image: Image.Image | str | Path,
        with_advisory: bool = True,
        affected_fraction: float | None = None,
    ) -> Diagnosis:
        """Diagnose a single close-up leaf image."""
        img = _as_image(image)
        started = time.perf_counter()
        batch = preprocess_image(img, self.card.preprocess)
        label_p, cat_p, sev_p, emb = self._run(batch)
        elapsed = (time.perf_counter() - started) * 1000.0
        novelty, is_novel = self._novelty(emb, 0)
        return self._decide(
            label_p[0], cat_p[0], sev_p[0], elapsed,
            with_advisory=with_advisory, affected_fraction=affected_fraction,
            novelty=novelty, is_novel=is_novel,
        )

    def diagnose_batch(
        self, images: Iterable[Image.Image | str | Path], with_advisory: bool = True
    ) -> list[Diagnosis]:
        imgs = [_as_image(i) for i in images]
        if not imgs:
            return []
        started = time.perf_counter()
        batch = np.concatenate(
            [preprocess_image(im, self.card.preprocess) for im in imgs], axis=0
        )
        label_p, cat_p, sev_p, emb = self._run(batch)
        per_image = (time.perf_counter() - started) * 1000.0 / len(imgs)
        out = []
        for i in range(len(imgs)):
            novelty, is_novel = self._novelty(emb, i)
            out.append(
                self._decide(
                    label_p[i], cat_p[i], sev_p[i], per_image,
                    with_advisory=with_advisory, novelty=novelty, is_novel=is_novel,
                )
            )
        return out

    def diagnose_canopy(
        self,
        image: Image.Image | str | Path,
        overlap: float = 0.2,
        max_tiles: int = 12,
        with_advisory: bool = True,
    ) -> dict[str, Any]:
        """Diagnose a wide canopy / whole-plant frame by tiling it.

        Returns the per-tile findings plus a field-level summary: which problem
        dominates and on what share of tiles. That share is the closest thing a
        single photo gives to a scouting count, and it feeds the "spot-treat vs
        treat the field" decision in the advisory.
        """
        img = _as_image(image)
        tiles = tile_image(img, self.card.preprocess.image_size, overlap, max_tiles)
        started = time.perf_counter()
        batch = np.concatenate(
            [preprocess_image(t, self.card.preprocess) for t, _ in tiles], axis=0
        )
        label_p, cat_p, sev_p, emb = self._run(batch)
        elapsed = (time.perf_counter() - started) * 1000.0

        findings = []
        for i, (_, box) in enumerate(tiles):
            novelty, is_novel = self._novelty(emb, i)
            d = self._decide(
                label_p[i], cat_p[i], sev_p[i], elapsed / len(tiles),
                with_advisory=False, novelty=novelty, is_novel=is_novel,
            )
            d.box = box
            findings.append(d)

        actionable = [
            f for f in findings
            if f.accepted and self.taxonomy.get(f.class_id) and self.taxonomy[f.class_id].is_actionable
        ]
        summary: dict[str, Any] = {
            "tiles": len(findings),
            "tiles_actionable": len(actionable),
            "latency_ms": round(elapsed, 2),
            "findings": [f.to_dict() for f in findings],
        }
        if actionable:
            counts: dict[str, list[float]] = {}
            for f in actionable:
                counts.setdefault(f.class_id, []).append(f.confidence)
            dominant = max(counts.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
            fraction = len(dominant[1]) / len(findings)
            summary["dominant_class"] = dominant[0]
            summary["affected_tile_fraction"] = round(fraction, 3)
            summary["mean_confidence"] = round(float(np.mean(dominant[1])), 4)
            if with_advisory:
                summary["advisory"] = self.advisory_engine.advise(
                    dominant[0],
                    confidence=float(np.mean(dominant[1])),
                    affected_fraction=fraction,
                ).to_dict()
        else:
            summary["dominant_class"] = None
            summary["affected_tile_fraction"] = 0.0
            if with_advisory:
                summary["advisory"] = self.advisory_engine.advise(
                    _healthy_id(self.taxonomy), confidence=1.0
                ).to_dict()
        return summary

    # -- decision --------------------------------------------------------
    def _decide(
        self,
        label_probs: np.ndarray,
        cat_probs: np.ndarray,
        sev_probs: np.ndarray,
        latency_ms: float,
        with_advisory: bool = True,
        affected_fraction: float | None = None,
        novelty: float | None = None,
        is_novel: bool = False,
    ) -> Diagnosis:
        policy = self.card.policy
        ids = self.card.class_ids
        order = np.argsort(-label_probs)
        top_idx = int(order[0])
        confidence = float(label_probs[top_idx])
        runner_up = float(label_probs[order[1]]) if len(order) > 1 else 0.0
        class_id = ids[top_idx]

        k = min(policy.topk, len(ids))
        topk = [(ids[int(i)], float(label_probs[int(i)])) for i in order[:k]]

        # Per-class thresholds let a high-cost class (late blight) be treated
        # more cautiously than a low-cost one, when the card provides them.
        threshold = self.card.per_class_thresholds.get(class_id, policy.min_confidence)

        accepted, reason = True, ""
        if is_novel:
            # Checked before confidence: an unfamiliar image can produce a very
            # high softmax score, so a confidence test would wave it straight
            # through. Feature-space distance is the signal that catches it.
            accepted, reason = False, (
                f"image is unlike the training data (novelty {novelty:.0f} vs "
                f"threshold {self.ood.threshold:.0f}) - the model has no basis "
                "for a diagnosis here, whatever its confidence says"
            )
        elif confidence < threshold:
            accepted, reason = False, (
                f"confidence {confidence:.2f} below the {threshold:.2f} threshold "
                "chosen on validation data"
            )
        elif policy.margin and (confidence - runner_up) < policy.margin:
            accepted, reason = False, (
                f"top-2 margin {confidence - runner_up:.2f} below {policy.margin:.2f} - "
                f"'{ids[int(order[1])]}' is nearly as likely"
            )

        cat_name = None
        cat_conf = None
        if cat_probs.size >= len(CATEGORIES):
            ci = int(np.argmax(cat_probs[: len(CATEGORIES)]))
            cat_name, cat_conf = CATEGORIES[ci], float(cat_probs[ci])

        severity = None
        sev_conf = None
        crop_class = self.taxonomy.get(class_id)
        if sev_probs.size >= len(SEVERITY_LEVELS) and crop_class is not None and crop_class.is_actionable:
            si = int(np.argmax(sev_probs[: len(SEVERITY_LEVELS)]))
            severity, sev_conf = SEVERITY_LEVELS[si], float(sev_probs[si])

        if accepted:
            advisory = (
                self.advisory_engine.advise(
                    class_id, confidence=confidence, severity=severity,
                    affected_fraction=affected_fraction,
                )
                if with_advisory
                else None
            )
            return Diagnosis(
                class_id=class_id,
                display_name=crop_class.display_name if crop_class else class_id,
                category=crop_class.category if crop_class else (cat_name or "background"),
                confidence=confidence,
                accepted=True,
                severity=severity,
                severity_confidence=sev_conf,
                topk=topk,
                novelty=novelty,
                is_novel=is_novel,
                category_from_head=cat_name,
                category_confidence=cat_conf,
                latency_ms=latency_ms,
                advisory=advisory,
            )

        # Abstained on the fine-grained class. The coarse head is trained on an
        # easier problem and is usually still right, so fall back to
        # category-level advice instead of saying nothing at all.
        advisory = None
        if with_advisory:
            advisory = self.advisory_engine.advise_uncertain(topk)
            if is_novel:
                advisory.message = (
                    "This photo does not look like the crop leaves this model was "
                    "trained on. Take a close-up of a single leaf, filling the frame, "
                    "in daylight. If the crop is not one this device supports, contact "
                    "your KVK / agriculture officer instead."
                )
                advisory.notes.append(
                    "Flagged as out-of-distribution, so no diagnosis is offered. This "
                    "is the system refusing to guess, not a failure."
                )
            elif cat_name and cat_conf and cat_conf >= policy.min_confidence and cat_name not in (
                "healthy", "background"
            ):
                advisory.notes.append(
                    f"The model is confident this is a {cat_name} problem "
                    f"({cat_conf:.0%}) even though it cannot name it. Treat it as a "
                    f"{cat_name} issue and confirm with your KVK before spending on inputs."
                )
        return Diagnosis(
            class_id=policy.abstain_label,
            display_name="Not identified",
            category=cat_name or "background",
            confidence=confidence,
            accepted=False,
            topk=topk,
            novelty=novelty,
            is_novel=is_novel,
            category_from_head=cat_name,
            category_confidence=cat_conf,
            latency_ms=latency_ms,
            reason=reason,
            advisory=advisory,
        )


def _as_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def _healthy_id(taxonomy: Taxonomy) -> str:
    for c in taxonomy:
        if c.category == "healthy":
            return c.id
    return taxonomy.class_ids[0]


__all__ = ["EdgeClassifier", "Diagnosis"]
