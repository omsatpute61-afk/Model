"""Model heads, loss masking, and the metrics the project is judged on."""

import numpy as np
import pytest
import torch

from cropguard.metrics import (
    classification_report,
    confusion_pairs,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
    selective_risk,
)
from cropguard.taxonomy import LIFE_STAGES
from cropguard.models.detector import (
    CropGuardNet,
    ExportWrapper,
    ModelConfig,
    ModelEMA,
    MultiHeadLoss,
    focal_loss,
)


@pytest.fixture(scope="module")
def model():
    return CropGuardNet(ModelConfig(num_classes=7, pretrained=False, embedding_dim=32))


def test_forward_returns_all_four_heads(model):
    out = model(torch.randn(2, 3, 64, 64))
    assert out.label_logits.shape == (2, 7)
    assert out.category_logits.shape == (2, 6)
    assert out.severity_logits.shape == (2, 4)
    assert out.embedding.shape == (2, 32)


def test_export_wrapper_emits_probabilities_and_the_embedding(model):
    out = ExportWrapper(model, temperature=1.7)(torch.randn(3, 3, 64, 64))
    label, category, severity, life_stage, embedding = out
    for p in (label, category, severity, life_stage):
        assert torch.allclose(p.sum(1), torch.ones(p.shape[0]), atol=1e-5)
    assert label.shape == (3, 7)
    assert life_stage.shape == (3, len(LIFE_STAGES))
    assert embedding.shape == (3, 32)


def test_forward_includes_the_life_stage_head(model):
    out = model(torch.randn(2, 3, 64, 64))
    assert out.life_stage_logits.shape == (2, len(LIFE_STAGES))


def test_life_stage_is_masked_when_unlabelled(model):
    """Most images have no life-stage label; those must not train the head."""
    out = model(torch.randn(4, 3, 64, 64))
    loss = MultiHeadLoss()
    label = torch.randint(0, 7, (4,))
    cat = torch.randint(0, 6, (4,))
    sev = torch.full((4,), -100)

    _, none_labelled = loss(out, label, cat, sev, life_stage=torch.full((4,), -100))
    assert none_labelled["life_stage"] == 0.0

    _, some_labelled = loss(out, label, cat, sev, life_stage=torch.tensor([2, -100, 4, -100]))
    assert some_labelled["life_stage"] > 0.0


def test_temperature_actually_changes_the_distribution(model):
    x = torch.randn(4, 3, 64, 64)
    sharp = ExportWrapper(model, temperature=0.5)(x)[0]
    flat = ExportWrapper(model, temperature=4.0)(x)[0]
    assert sharp.max(1).values.mean() > flat.max(1).values.mean()


def test_missing_severity_labels_are_masked_not_invented(model):
    out = model(torch.randn(4, 3, 64, 64))
    loss = MultiHeadLoss()
    label = torch.randint(0, 7, (4,))
    cat = torch.randint(0, 6, (4,))
    _, all_missing = loss(out, label, cat, torch.full((4,), -100))
    assert all_missing["severity"] == 0.0
    _, some_present = loss(out, label, cat, torch.tensor([1, -100, 2, -100]))
    assert some_present["severity"] > 0.0


def test_invalid_images_are_excluded_from_the_loss(model):
    out = model(torch.randn(4, 3, 64, 64))
    loss = MultiHeadLoss()
    label = torch.randint(0, 7, (4,))
    cat = torch.randint(0, 6, (4,))
    sev = torch.full((4,), -100)
    total, parts = loss(out, label, cat, sev, valid=torch.tensor([0, 0, 0, 0]))
    assert parts["total"] == 0.0
    assert float(total.detach()) == 0.0


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[8.0, 0.0], [0.2, 0.0]])
    targets = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    fl_easy = focal_loss(logits[:1], targets[:1])
    fl_hard = focal_loss(logits[1:], targets[1:])
    assert fl_easy / ce[0] < fl_hard / ce[1]


def test_param_groups_slow_the_pretrained_trunk(model):
    groups = model.param_groups(1e-3, backbone_lr_scale=0.1)
    by_name = {g["name"]: g["lr"] for g in groups}
    assert by_name["trunk"] < by_name["head"]


def test_ema_lags_the_live_weights_once_past_the_ramp(model):
    """Regression: with the stock tau=2000 ramp, a short run's EMA is a plain
    copy of the live model and contributes nothing."""
    ema = ModelEMA(model, decay=0.99, tau=5.0)
    for _ in range(60):  # well past the ramp
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        ema.update(model)
    assert ema.updates == 60
    assert any(
        not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(ema.module.state_dict().values(), model.state_dict().values())
        if a.dtype.is_floating_point
    )


def test_ema_ramp_is_scaled_by_tau(model):
    """A large tau keeps the EMA glued to the live weights; a small one frees it."""
    slow = ModelEMA(model, decay=0.99, tau=10000.0)
    fast = ModelEMA(model, decay=0.99, tau=5.0)
    for _ in range(40):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        slow.update(model)
        fast.update(model)

    def drift(ema):
        live = model.state_dict()
        return max(
            float((v - live[k]).abs().max())
            for k, v in ema.module.state_dict().items()
            if v.dtype.is_floating_point
        )

    assert drift(fast) > drift(slow)


def test_macro_f1_punishes_a_majority_class_predictor():
    """The reason macro F1 is the selection metric and accuracy is not."""
    y = np.array([0] * 90 + [1] * 10)
    always_zero = np.zeros(100, dtype=int)
    report = classification_report(y, always_zero, ["healthy", "rare_pest"])
    assert report.accuracy == pytest.approx(0.90)
    assert report.macro_f1 < 0.5


def test_absent_classes_do_not_drag_macro_f1_down():
    y = np.array([0, 0, 1, 1])
    report = classification_report(y, y, ["a", "b", "never_seen"])
    assert report.macro_f1 == pytest.approx(1.0)


def test_temperature_scaling_reduces_calibration_error():
    rng = np.random.default_rng(0)
    n, k = 800, 6
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.0
    logits *= 3.0  # deliberately overconfident
    before = expected_calibration_error(_softmax(logits), y)
    t = fit_temperature(logits, y)
    after = expected_calibration_error(_softmax(logits / t), y)
    assert after < before


def test_threshold_floor_preserves_an_abstention_region():
    rng = np.random.default_rng(1)
    n, k = 400, 8
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 8.0   # near-perfect, so any threshold is "feasible"
    probs = _softmax(logits)
    with_floor, _ = select_threshold(probs, y, min_threshold=0.30)
    without, _ = select_threshold(probs, y, min_threshold=0.0)
    assert with_floor >= 0.30
    assert without < with_floor


def test_selective_risk_error_falls_as_the_threshold_rises():
    rng = np.random.default_rng(2)
    n, k = 500, 5
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5
    rows = selective_risk(_softmax(logits), y)
    low = [r for r in rows if r["threshold"] <= 0.2][0]
    high = [r for r in rows if r["threshold"] >= 0.8][0]
    assert high["selective_error"] <= low["selective_error"]
    assert high["coverage"] <= low["coverage"]


def test_cross_category_confusions_are_surfaced_first():
    cm = np.array([[5, 1, 0], [0, 5, 1], [3, 0, 5]])
    pairs = confusion_pairs(cm, ["a", "b", "c"], top=3, categories=["disease", "disease", "pest"])
    assert pairs[0]["cross_category"] is True
    assert pairs[0]["true"] == "c"


def _softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
