"""Preprocessing parity, OOD rejection, and the export -> edge contract.

These are the tests that stop a silent field failure: a preprocessing drift or
a reordered label list produces confident, wrong, actionable advice.
"""

import numpy as np
import pytest
from PIL import Image

from cropguard.edge.preprocess import center_crop, preprocess_image, resize_shorter_side, tile_image
from cropguard.model_card import DecisionPolicy, ModelCard, PreprocessSpec
from cropguard.ood import MahalanobisOOD


@pytest.mark.parametrize("size,crop_pct", [(224, 0.90), (192, 0.875), (128, 1.0)])
@pytest.mark.parametrize("hw", [(300, 420), (421, 301), (224, 224), (97, 150), (640, 480)])
def test_edge_preprocessing_matches_torchvision(size, crop_pct, hw):
    """The device reimplements the eval transform in numpy. If the two drift,
    the model sees inputs it was never trained on and nobody finds out."""
    from cropguard.data.transforms import build_eval_transform

    h, w = hw
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (h, w, 3)).astype("uint8")
    )
    edge = preprocess_image(img, PreprocessSpec(image_size=size, crop_pct=crop_pct))[0]
    torchvision = build_eval_transform(size, crop_pct)(img).numpy()
    assert np.abs(edge - torchvision).max() < 1e-4


def test_resize_matches_torchvision_dimension_rule():
    img = Image.new("RGB", (420, 300))
    assert resize_shorter_side(img, 249).size == (348, 249)


def test_center_crop_pads_a_small_image_rather_than_failing():
    assert center_crop(Image.new("RGB", (40, 40)), 100).size == (100, 100)


def test_tiles_cover_the_frame_and_stay_bounded():
    tiles = tile_image(Image.new("RGB", (1200, 900)), 224, overlap=0.2, max_tiles=12)
    assert 1 < len(tiles) <= 12
    for img, box in tiles:
        assert box[2] <= 1200 and box[3] <= 900
        assert img.size[0] > 0 and img.size[1] > 0


def test_small_image_yields_one_tile():
    tiles = tile_image(Image.new("RGB", (100, 100)), 224)
    assert len(tiles) == 1


def test_model_card_refuses_a_label_count_mismatch():
    card = ModelCard(class_ids=["a", "b", "c"])
    card.validate(num_outputs=3)
    with pytest.raises(ValueError, match="refusing to guess"):
        card.validate(num_outputs=4)


def test_model_card_rejects_implausible_policy():
    with pytest.raises(ValueError):
        ModelCard(class_ids=["a"], policy=DecisionPolicy(temperature=0.0)).validate()
    with pytest.raises(ValueError):
        ModelCard(class_ids=["a"], policy=DecisionPolicy(min_confidence=1.5)).validate()


def test_model_card_round_trips(tmp_path):
    card = ModelCard(
        class_ids=["x", "y"],
        preprocess=PreprocessSpec(image_size=160, crop_pct=0.8),
        policy=DecisionPolicy(min_confidence=0.42, temperature=1.3),
    )
    loaded = ModelCard.load(card.save(tmp_path))
    assert loaded.class_ids == ["x", "y"]
    assert loaded.preprocess.image_size == 160
    assert loaded.policy.min_confidence == pytest.approx(0.42)
    assert loaded.preprocess.resize_to == 200


def test_mahalanobis_separates_novel_from_familiar():
    rng = np.random.default_rng(0)
    c, d = 5, 24
    centres = rng.normal(0, 4, (c, d))
    labels = rng.integers(0, c, 600)
    train = centres[labels] + rng.normal(0, 1, (600, d))
    val_labels = rng.integers(0, c, 200)
    val = centres[val_labels] + rng.normal(0, 1, (200, d))

    det = MahalanobisOOD.fit(train, labels, c)
    stats = det.calibrate(val, percentile=97.5)

    novel = rng.normal(0, 15, (200, d))
    assert det.is_ood(novel).mean() > 0.9
    assert det.is_ood(val).mean() < 0.10
    assert stats.threshold > 0


def test_mahalanobis_handles_a_class_with_no_samples():
    rng = np.random.default_rng(1)
    emb = rng.normal(0, 1, (50, 8))
    labels = np.zeros(50, dtype=int)
    det = MahalanobisOOD.fit(emb, labels, num_classes=3)  # classes 1 and 2 unseen
    assert np.isfinite(det.means).all()
    assert np.isfinite(det.score(emb)).all()


def test_mahalanobis_round_trips(tmp_path):
    rng = np.random.default_rng(2)
    emb = rng.normal(0, 1, (80, 12))
    labels = rng.integers(0, 3, 80)
    det = MahalanobisOOD.fit(emb, labels, 3, ["a", "b", "c"])
    det.calibrate(emb)
    reloaded = MahalanobisOOD.load(det.save(tmp_path))
    assert reloaded.class_ids == ["a", "b", "c"]
    assert reloaded.threshold == pytest.approx(det.threshold)
    assert np.abs(reloaded.score(emb) - det.score(emb)).max() < 1e-3


# --------------------------------------------------------------- life stage
def test_life_stage_changes_the_recommended_action():
    """The whole reason the model carries a life-stage head.

    Species says what the problem is; stage says whether today is the day to
    spend money on it. Spraying because moths appeared on a trap wastes the
    application and selects for resistance.
    """
    from cropguard.advisory import default_engine

    engine = default_engine()
    adult = engine.advise("pest__fall_armyworm", confidence=0.92, life_stage="adult")
    larva = engine.advise("pest__fall_armyworm", confidence=0.92, life_stage="larva")

    assert adult.action == "monitor_and_count"
    assert larva.action == "treat_affected_plants"
    assert adult.urgency_rank < larva.urgency_rank
    assert any("trap threshold" in n for n in adult.notes)
    assert any("feeding stage" in n for n in larva.notes)


def test_egg_stage_advises_timing_not_spraying_now():
    from cropguard.advisory import default_engine

    egg = default_engine().advise("pest__american_bollworm", confidence=0.9, life_stage="egg")
    assert egg.action == "scout_and_time_treatment"
    assert any("hatch" in n.lower() for n in egg.notes)


def test_life_stage_is_ignored_for_non_pest_classes():
    """A life stage on a leaf-spot image is meaningless and must not be recorded."""
    from cropguard.advisory import default_engine

    engine = default_engine()
    with_stage = engine.advise("tomato__late_blight", confidence=0.9, life_stage="adult")
    without = engine.advise("tomato__late_blight", confidence=0.9)
    assert with_stage.life_stage is None
    assert with_stage.urgency == without.urgency
    assert with_stage.action == without.action


def test_unknown_is_never_predicted_as_a_life_stage():
    """Index 0 is the absence of a label, not a fifth stage the model may pick."""
    from cropguard.taxonomy import LIFE_STAGES

    assert LIFE_STAGES[0] == "unknown"
    assert set(LIFE_STAGES[1:]) == {"egg", "larva", "nymph", "adult"}
