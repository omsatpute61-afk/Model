"""Metrics chosen for an imbalanced, high-cost-of-error problem.

Plain accuracy is the wrong headline number here. A district archive that is
70% healthy leaves gives 70% accuracy to a model that says "healthy" every
time - and that model misses every outbreak. So the primary metrics are
**macro F1** (every class counts equally, including the rare pest) and
**per-class recall** (what fraction of real outbreaks did we catch).

Calibration matters as much as accuracy: the runtime decides whether to advise
a spray based on a probability, so that probability has to mean something.
``expected_calibration_error`` and temperature scaling are part of the pipeline,
not an afterthought.

numpy only - no sklearn dependency on the training box or in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    np.add.at(cm, (y_true[mask], y_pred[mask]), 1)
    return cm


@dataclass
class ClassMetrics:
    class_id: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class ClassificationReport:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    top_k: dict[int, float] = field(default_factory=dict)
    per_class: list[ClassMetrics] = field(default_factory=list)
    confusion: np.ndarray | None = None
    class_ids: list[str] = field(default_factory=list)

    def to_dict(self, include_confusion: bool = False) -> dict:
        d = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "top_k": {str(k): v for k, v in self.top_k.items()},
            "per_class": [
                {
                    "class_id": c.class_id, "support": c.support,
                    "precision": c.precision, "recall": c.recall, "f1": c.f1,
                }
                for c in self.per_class
            ],
        }
        if include_confusion and self.confusion is not None:
            d["confusion"] = self.confusion.tolist()
            d["class_ids"] = list(self.class_ids)
        return d

    def worst_classes(self, n: int = 5, min_support: int = 1) -> list[ClassMetrics]:
        """Where the model is failing - the list to take back to data collection."""
        candidates = [c for c in self.per_class if c.support >= min_support]
        return sorted(candidates, key=lambda c: (c.f1, c.recall))[:n]

    def format_table(self, max_rows: int | None = None) -> str:
        rows = sorted(self.per_class, key=lambda c: c.f1)
        if max_rows:
            rows = rows[:max_rows]
        w = max((len(c.class_id) for c in rows), default=10)
        lines = [f"{'class':<{w}}  {'prec':>6} {'rec':>6} {'f1':>6} {'n':>6}"]
        lines.append("-" * len(lines[0]))
        for c in rows:
            lines.append(
                f"{c.class_id:<{w}}  {c.precision:>6.3f} {c.recall:>6.3f} {c.f1:>6.3f} {c.support:>6d}"
            )
        return "\n".join(lines)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[str],
    probs: np.ndarray | None = None,
    top_k: Sequence[int] = (3, 5),
) -> ClassificationReport:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(class_ids)
    cm = confusion_matrix(y_true, y_pred, n)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    seen = support > 0  # classes absent from this split must not drag macro down
    total = float(support.sum())
    accuracy = float(tp.sum() / total) if total else 0.0
    balanced = float(recall[seen].mean()) if seen.any() else 0.0
    macro_f1 = float(f1[seen].mean()) if seen.any() else 0.0
    weighted_f1 = float((f1 * support).sum() / total) if total else 0.0

    topk_scores: dict[int, float] = {}
    if probs is not None and len(probs):
        order = np.argsort(-probs, axis=1)
        for k in top_k:
            k = min(k, probs.shape[1])
            hit = (order[:, :k] == y_true[:, None]).any(axis=1)
            topk_scores[k] = float(hit.mean())

    per_class = [
        ClassMetrics(
            class_id=class_ids[i], support=int(support[i]),
            precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
        )
        for i in range(n)
    ]
    return ClassificationReport(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=float(precision[seen].mean()) if seen.any() else 0.0,
        macro_recall=balanced,
        top_k=topk_scores,
        per_class=per_class,
        confusion=cm,
        class_ids=list(class_ids),
    )


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """Gap between stated confidence and observed accuracy.

    A model at "90% sure" should be right about 90% of the time. If it is right
    60% of the time, every confidence threshold downstream is meaningless.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin confidence vs accuracy - the data behind a reliability diagram."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append(
            {
                "lower": float(lo), "upper": float(hi), "count": int(m.sum()),
                "confidence": float(conf[m].mean()) if m.any() else 0.0,
                "accuracy": float(correct[m].mean()) if m.any() else 0.0,
            }
        )
    return out


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200, lr: float = 0.02
) -> float:
    """Single-parameter temperature scaling (Guo et al., 2017).

    Fitted on validation logits, applied at inference. Costs one division and
    buys confidence values the advisory layer can threshold on honestly.
    """
    import torch

    t = torch.ones(1, requires_grad=True)
    lg = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.long)
    opt = torch.optim.LBFGS([t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / t.clamp(min=1e-2), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:  # noqa: BLE001 - fall back to uncalibrated rather than fail a run
        return 1.0
    value = float(t.detach().clamp(min=0.05, max=20.0).item())
    return value if np.isfinite(value) else 1.0


def selective_risk(
    probs: np.ndarray, y_true: np.ndarray, thresholds: Sequence[float] | None = None
) -> list[dict]:
    """Coverage / error trade-off for the abstention threshold.

    Each row answers: "if we only act when confidence exceeds t, what share of
    photos do we answer (coverage) and how often are we wrong when we do
    (selective error)?" This is the table you pick an operating point from.
    """
    thresholds = list(thresholds or np.round(np.arange(0.05, 1.0, 0.05), 2))
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    n = len(y_true)
    rows = []
    for t in thresholds:
        act = conf >= t
        cov = float(act.mean()) if n else 0.0
        err = float(1.0 - correct[act].mean()) if act.any() else 0.0
        rows.append(
            {
                "threshold": float(t),
                "coverage": cov,
                "selective_error": err,
                "answered": int(act.sum()),
                "wrong_answers": int((~correct & act).sum()),
                "missed_correct": int((correct & ~act).sum()),
            }
        )
    return rows


def select_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    max_selective_error: float = 0.10,
    min_coverage: float = 0.50,
    min_threshold: float = 0.30,
) -> tuple[float, dict]:
    """Lowest threshold meeting the error budget while keeping coverage up.

    Default budget: at most 10% of the answers we *do* give may be wrong, and
    we must still answer at least half the photos - a model that abstains on
    everything is safe and useless.

    ``min_threshold`` is a floor, and it is not cosmetic. This search only sees
    *in-distribution* validation images, so on a well-separated model the
    lowest feasible threshold is often near zero - which satisfies the error
    budget and simultaneously destroys out-of-distribution rejection. A photo
    of a shoe, a crop the model was never trained on, or a blurred frame would
    be accepted at 0.06 confidence and turned into a spray recommendation. The
    validation set cannot warn about that, because it contains no such images.
    The floor keeps a meaningful "I don't know" region that the abstention
    logic in ``cropguard.edge`` depends on.
    """
    rows = selective_risk(probs, y_true)
    feasible = [
        r for r in rows
        if r["selective_error"] <= max_selective_error
        and r["coverage"] >= min_coverage
        and r["threshold"] >= min_threshold
    ]
    if feasible:
        best = min(feasible, key=lambda r: r["threshold"])
        return best["threshold"], best
    # Nothing meets the budget. Relax *coverage*, never the floor: abstaining
    # more often is the safe direction, and dropping the floor would make the
    # model accept near-uniform predictions as diagnoses. A model that cannot
    # meet its error budget should answer less, not answer worse.
    covered = [
        r for r in rows if r["coverage"] >= min_coverage and r["threshold"] >= min_threshold
    ] or [r for r in rows if r["threshold"] >= min_threshold] or rows
    best = min(covered, key=lambda r: (r["selective_error"], -r["coverage"]))
    return best["threshold"], best


def confusion_pairs(
    cm: np.ndarray,
    class_ids: Sequence[str],
    top: int = 10,
    categories: Sequence[str] | None = None,
) -> list[dict]:
    """Most frequent off-diagonal confusions, with an agronomic severity note.

    A tomato/potato early-blight mix-up is harmless (same fungicide). A
    pest/deficiency mix-up sends the farmer to buy the wrong input. The
    ``cross_category`` flag marks the second kind.
    """
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                cross = (
                    bool(categories[i] != categories[j]) if categories is not None else None
                )
                pairs.append(
                    {
                        "true": class_ids[i],
                        "predicted": class_ids[j],
                        "count": int(cm[i, j]),
                        "share_of_true": float(cm[i, j] / max(1, cm[i].sum())),
                        "cross_category": cross,
                    }
                )
    # Cross-category confusions first: those are the ones that change the advice.
    pairs.sort(key=lambda p: (not p["cross_category"], -p["count"]))
    return pairs[:top]


__all__ = [
    "confusion_matrix",
    "classification_report",
    "ClassificationReport",
    "ClassMetrics",
    "expected_calibration_error",
    "reliability_bins",
    "fit_temperature",
    "selective_risk",
    "select_threshold",
    "confusion_pairs",
]
