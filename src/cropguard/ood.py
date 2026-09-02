"""Out-of-distribution rejection in feature space.

Why this module exists: a softmax confidence threshold does **not** detect an
unfamiliar input. Photograph the sky, a hand, a bag of fertiliser, or a crop
the model was never trained on, and a CNN will report 0.99 for whatever class
happens to be nearest. Measured on this project's own model, a flat grey image
scored 0.996 for "healthy". A farmer acting on that is being misled by a system
that has no idea it is out of its depth.

The fix is to score in *feature* space rather than probability space. We fit a
class-conditional Gaussian to the training embeddings with a single shared
covariance and score a new image by its Mahalanobis distance to the nearest
class centre (Lee et al., NeurIPS 2018). Familiar leaves land close to a
centre; a photo of the sky does not, regardless of what the softmax says.

Cost on device: one 128-dim matrix-vector product per image, and a file of a
few tens of kilobytes. No extra forward pass - the embedding is already an
output of the exported model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OOD_FILENAME = "ood.npz"


@dataclass
class OODStats:
    """Summary of how the detector behaves, for the model card."""

    threshold: float
    percentile: float
    train_median: float
    val_median: float
    val_reject_rate: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "percentile": self.percentile,
            "train_median": self.train_median,
            "val_median": self.val_median,
            "val_reject_rate": self.val_reject_rate,
        }


class MahalanobisOOD:
    """Class-conditional Gaussian novelty detector over model embeddings."""

    def __init__(
        self,
        means: np.ndarray,
        precision: np.ndarray,
        threshold: float = float("inf"),
        class_ids: list[str] | None = None,
        stats: dict | None = None,
    ):
        self.means = np.asarray(means, dtype=np.float64)         # (C, D)
        self.precision = np.asarray(precision, dtype=np.float64)  # (D, D)
        self.threshold = float(threshold)
        self.class_ids = list(class_ids or [])
        self.stats = stats or {}

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        class_ids: list[str] | None = None,
        shrinkage: float | None = None,
    ) -> "MahalanobisOOD":
        """Fit class means and one shared covariance.

        A shared covariance (rather than per-class) is deliberate: a rare pest
        may have 30 training images, far too few to estimate a 128x128
        covariance. Pooling makes the estimate usable for every class,
        including the tail classes that matter most.

        Even pooled, the estimate is thin - a district set might give 400
        samples for a 128-dimensional space. An unregularised inverse of that
        is dominated by noise directions with near-zero eigenvalues, and the
        resulting "distances" are enormous and meaningless: measured here, a
        384-sample fit produced a threshold of 48 000 and rejected 0% of pure
        noise, i.e. the detector had stopped detecting anything. ``shrinkage``
        defaults to the Ledoit-Wolf optimal intensity, which fixes that without
        a hand-tuned constant.
        """
        emb = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
        d = emb.shape[1]

        means = np.zeros((num_classes, d), dtype=np.float64)
        global_mean = emb.mean(axis=0) if len(emb) else np.zeros(d)
        centred = np.empty_like(emb)
        for c in range(num_classes):
            m = labels == c
            if m.any():
                means[c] = emb[m].mean(axis=0)
                centred[m] = emb[m] - means[c]
            else:
                means[c] = global_mean  # unseen class: fall back, never NaN

        n = max(1, len(emb) - num_classes)
        cov = centred.T @ centred / n

        intensity = (
            _ledoit_wolf_intensity(centred, cov) if shrinkage is None else float(shrinkage)
        )
        intensity = float(min(1.0, max(0.0, intensity)))
        mu = float(np.trace(cov)) / max(1, d)
        cov = (1.0 - intensity) * cov + intensity * max(mu, 1e-12) * np.eye(d)

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)
        detector = cls(means, precision, float("inf"), class_ids)
        detector.stats = {"shrinkage": intensity, "fit_samples": int(len(emb)), "dim": int(d)}
        return detector

    #: Below this relative spread the embedding space has collapsed and
    #: distances carry no information.
    DEGENERATE_SPREAD = 1e-5

    def calibrate(
        self, embeddings: np.ndarray, percentile: float = 97.5
    ) -> OODStats:
        """Set the reject threshold from in-distribution validation scores.

        ``percentile=97.5`` means we accept that ~2.5% of genuine field photos
        will be sent back as "retake this" in exchange for rejecting novel
        inputs. That is the right trade for advice that costs money to follow:
        a second photo is cheap, a wrong spray is not.
        """
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        val_scores = self.score(emb)

        # A collapsed embedding space (undertrained model, dead activations)
        # produces distances of ~1e-8 for everything. A percentile of that is
        # also ~1e-8, and every real photo then scores above it: the device
        # would refuse 100% of images. Disabling the detector and falling back
        # to confidence alone is bad; silently rejecting every farmer's photo
        # is far worse.
        # Spread must be measured ACROSS SAMPLES, per dimension. A collapsed
        # model emits the same vector for every image; that vector still varies
        # across its own dimensions, so a global np.std() looks perfectly
        # healthy while the representation carries no information at all.
        per_dim = np.std(emb, axis=0)
        scale = max(1e-6, float(np.abs(emb).mean()))
        spread = float(np.mean(per_dim)) / scale
        if spread < self.DEGENERATE_SPREAD or not np.isfinite(val_scores).all():
            self.threshold = float("inf")
            self.stats = {
                **self.stats,
                "threshold": self.threshold,
                "percentile": percentile,
                "train_median": 0.0,
                "val_median": float(np.median(val_scores)) if val_scores.size else 0.0,
                "val_reject_rate": 0.0,
                "disabled_reason": (
                    f"across-sample embedding spread {spread:.2e} is degenerate - the model has "
                    "collapsed to a near-constant representation, so novelty "
                    "distances are meaningless. Train longer before relying on "
                    "out-of-distribution rejection."
                ),
            }
            return OODStats(
                threshold=self.threshold, percentile=percentile, train_median=0.0,
                val_median=self.stats["val_median"], val_reject_rate=0.0,
            )

        threshold = float(np.percentile(val_scores, percentile))
        self.threshold = threshold
        fit_stats = {k: v for k, v in self.stats.items() if k in ("shrinkage", "fit_samples", "dim")}
        self.stats = {
            **fit_stats,
            **OODStats(
                threshold=threshold,
                percentile=percentile,
                train_median=float(self.stats.get("train_median", np.median(val_scores))),
                val_median=float(np.median(val_scores)),
                val_reject_rate=float((val_scores > threshold).mean()),
            ).to_dict(),
        }
        return OODStats(
            threshold=self.stats["threshold"],
            percentile=self.stats["percentile"],
            train_median=self.stats["train_median"],
            val_median=self.stats["val_median"],
            val_reject_rate=self.stats["val_reject_rate"],
        )

    # -- scoring ---------------------------------------------------------
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to the nearest class centre.

        Lower is more familiar. Returned per row of ``embeddings``.
        """
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        # (N, C, D) would be wasteful; expand one class at a time.
        out = np.empty((emb.shape[0], self.means.shape[0]), dtype=np.float64)
        for c in range(self.means.shape[0]):
            delta = emb - self.means[c]
            out[:, c] = np.einsum("nd,dk,nk->n", delta, self.precision, delta)
        return out.min(axis=1)

    @property
    def enabled(self) -> bool:
        """False when calibration found the embedding space unusable."""
        return bool(np.isfinite(self.threshold))

    def is_ood(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros(np.atleast_2d(embeddings).shape[0], dtype=bool)
        return self.score(embeddings) > self.threshold

    def novelty_ratio(self, embeddings: np.ndarray) -> np.ndarray:
        """Score expressed relative to the threshold (1.0 = exactly at it)."""
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            return np.zeros(np.atleast_2d(embeddings).shape[0])
        return self.score(embeddings) / self.threshold

    # -- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / OOD_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            means=self.means.astype(np.float32),
            precision=self.precision.astype(np.float32),
            threshold=np.array([self.threshold], dtype=np.float64),
            class_ids=np.array(self.class_ids, dtype=object),
            stats=np.array([json.dumps(self.stats)], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisOOD":
        path = Path(path)
        if path.is_dir():
            path = path / OOD_FILENAME
        with np.load(path, allow_pickle=True) as z:
            stats = {}
            if "stats" in z:
                try:
                    stats = json.loads(str(z["stats"][0]))
                except Exception:  # noqa: BLE001
                    stats = {}
            return cls(
                means=z["means"],
                precision=z["precision"],
                threshold=float(z["threshold"][0]),
                class_ids=[str(c) for c in z["class_ids"]] if "class_ids" in z else [],
                stats=stats,
            )


def _ledoit_wolf_intensity(centred: np.ndarray, cov: np.ndarray) -> float:
    """Ledoit-Wolf optimal shrinkage towards a scaled identity.

    Closed form, no cross-validation, and it adapts automatically: near 0 when
    there are plenty of samples per dimension, near 1 when there are not.
    """
    n, d = centred.shape
    if n < 2:
        return 1.0
    mu = float(np.trace(cov)) / d
    delta = float(np.sum((cov - mu * np.eye(d)) ** 2)) / d
    if delta <= 0:
        return 1.0
    # Mean squared Frobenius distance between each sample's outer product and cov.
    sq = np.sum(centred**2, axis=1)
    beta = float(np.sum(sq**2) / n - np.sum(cov**2)) / (n * d)
    beta = max(0.0, min(beta, delta))
    return beta / delta


__all__ = ["MahalanobisOOD", "OODStats", "OOD_FILENAME"]
