"""The whole pipeline in one test: data -> train -> export -> device -> advice.

Deliberately small (four classes, 64px) so it runs in CI in a couple of
minutes, but sized past the point where the model actually learns - from a
random initialisation, BatchNorm running statistics keep validation at chance
for the first several epochs, and a run that stops inside that window
exercises none of the accept/reject logic. It is sized to the point where the model actually learns
something: a run that stays at chance exercises none of the accept/reject
logic and would pass while the device refuses every photo. It does not check accuracy - synthetic data at this scale
proves nothing about field performance. It checks that the *contract* holds at
every hand-off, which is where this kind of system actually breaks:

* the exported label order matches the trained one,
* the device's numpy preprocessing matches training's torchvision one,
* ONNX outputs match torch outputs,
* an unfamiliar image is refused rather than diagnosed,
* a diagnosis carries advice a farmer could act on.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow

CLASSES = ["background", "tomato__healthy", "tomato__late_blight", "pest__aphid"]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    from cropguard.config import TrainConfig
    from cropguard.data.manifest import (
        scan_image_folder,
        stratified_group_split,
        write_manifest,
    )
    from cropguard.data.synthetic import generate_dataset
    from cropguard.train import train

    root = tmp_path_factory.mktemp("e2e")
    data = generate_dataset(root / "images", class_ids=CLASSES, per_class=80, size=64, seed=11)
    records, unmapped = scan_image_folder(data)
    assert not unmapped
    records = stratified_group_split(records, 0.2, 0.2, seed=3)
    manifest = write_manifest(records, root / "manifest.csv")

    cfg = TrainConfig()
    cfg.run_name = "e2e"
    cfg.output_dir = str(root / "runs")
    cfg.data.manifest = str(manifest)
    cfg.data.image_size = 64
    cfg.data.batch_size = 8
    cfg.data.num_workers = 0
    cfg.data.aug_strength = 0.15
    cfg.model.pretrained = False
    cfg.model.embedding_dim = 32
    cfg.optim.epochs = 45
    cfg.optim.lr = 3e-3
    cfg.optim.early_stopping_patience = 0
    cfg.optim.amp = False

    result = train(cfg)
    return Path(result["output_dir"]), manifest


def test_training_writes_a_complete_run(trained_run):
    run_dir, _ = trained_run
    for name in ("best.pt", "last.pt", "model_card.json", "taxonomy.json", "history.json", "ood.npz"):
        assert (run_dir / name).exists(), f"missing {name}"


def test_model_card_pins_the_contract(trained_run):
    from cropguard.model_card import ModelCard

    run_dir, _ = trained_run
    card = ModelCard.load(run_dir / "model_card.json")
    assert set(card.class_ids) == set(CLASSES)
    assert card.preprocess.image_size == 64
    assert card.policy.min_confidence >= 0.30      # the OOD-preserving floor
    assert 0 < card.policy.temperature < 20
    assert card.limitations
    card.validate(num_outputs=len(CLASSES))


@pytest.fixture(scope="module")
def bundle(trained_run):
    from cropguard.export import export_run

    run_dir, _ = trained_run
    summary = export_run(run_dir, formats=("onnx", "int8"))
    assert summary["all_verified"], summary
    return Path(summary["out_dir"])


def test_bundle_is_self_contained(bundle):
    for name in ("cropguard.onnx", "model_card.json", "taxonomy.json", "advisory.json", "ood.npz"):
        assert (bundle / name).exists(), f"{name} missing from the deployment bundle"
    # weights must be embedded, not left in a sidecar the deployer forgets
    assert not (bundle / "cropguard.onnx.data").exists()


def test_novelty_detector_is_refitted_per_artifact(bundle):
    """Regression for the worst bug found in this project.

    The detector fitted on torch embeddings does not transfer to the quantised
    model: INT8 embeddings have the same mean and standard deviation, but
    Mahalanobis distance amplifies the perturbation, and the device rejected
    **76% of real farm photos** while reporting 0.99 confidence on them.
    """
    from cropguard.ood import MahalanobisOOD

    for stem in ("cropguard", "cropguard.int8"):
        path = bundle / f"{stem}.ood.npz"
        assert path.exists(), f"{path.name} missing - the artefact would use a mismatched detector"
        assert MahalanobisOOD.load(path).enabled


def test_quantised_model_does_not_reject_its_own_training_data(bundle, trained_run):
    """The INT8 model is what the runtime picks by default. If its novelty
    threshold is mismatched, the device refuses real photos in the field."""
    from cropguard.data.manifest import read_manifest
    from cropguard.edge.runtime import EdgeClassifier

    _, manifest = trained_run
    records = [r for r in read_manifest(manifest) if r.split == "train"][:40]
    for model_file in ("cropguard.onnx", "cropguard.int8.onnx"):
        clf = EdgeClassifier(bundle, model_file=model_file, backend="onnx")
        rejected = sum(
            1 for r in records if not clf.diagnose(r.path, with_advisory=False).accepted
        )
        assert rejected < len(records) * 0.25, (
            f"{model_file} rejected {rejected}/{len(records)} of its own training "
            "images - the novelty threshold does not match this artefact"
        )


def test_int8_is_smaller_than_fp32(bundle):
    fp32 = (bundle / "cropguard.onnx").stat().st_size
    int8 = (bundle / "cropguard.int8.onnx").stat().st_size
    assert int8 < fp32


def test_onnx_matches_torch_on_the_same_input(trained_run, bundle):
    from cropguard.edge.runtime import EdgeClassifier

    run_dir, _ = trained_run
    onnx_clf = EdgeClassifier(bundle, model_file="cropguard.onnx", backend="onnx")
    torch_clf = EdgeClassifier(run_dir, model_file="best.pt", backend="torch", use_ood=False)

    img = Image.fromarray(
        np.random.default_rng(7).integers(0, 255, (120, 90, 3)).astype("uint8")
    )
    a = onnx_clf.predict_proba([img])
    b = torch_clf.predict_proba([img])
    assert np.abs(a - b).max() < 1e-3
    assert a.argmax() == b.argmax()


def test_device_diagnoses_training_classes(bundle, trained_run):
    from cropguard.data.manifest import read_manifest
    from cropguard.edge.runtime import EdgeClassifier

    _, manifest = trained_run
    clf = EdgeClassifier(bundle)
    records = [r for r in read_manifest(manifest) if r.split == "test"]
    assert records

    results = [(r.class_id, clf.diagnose(r.path, with_advisory=False)) for r in records]
    assert all(d.class_id in CLASSES + ["unknown"] for _, d in results)
    assert all(0.0 <= d.confidence <= 1.0 for _, d in results)
    # A short run on synthetic data need not be accurate, but it must not
    # abstain on everything: a device that refuses every photo is a field
    # outage, and that is exactly what a collapsed embedding space caused
    # before the degeneracy guard in cropguard.ood.
    answered = sum(1 for _, d in results if d.accepted)
    assert answered > len(results) * 0.3, (
        f"only {answered}/{len(results)} images answered - "
        "the abstention or OOD threshold is unusable"
    )


def test_unfamiliar_images_are_refused_not_diagnosed(bundle):
    """The failure this project cares most about: confident wrong advice."""
    from cropguard.edge.runtime import EdgeClassifier

    clf = EdgeClassifier(bundle)
    assert clf.ood is not None, "bundle shipped without an OOD detector"

    rng = np.random.default_rng(3)
    noise = Image.fromarray(rng.integers(0, 255, (200, 200, 3)).astype("uint8"))
    d = clf.diagnose(noise)
    assert d.novelty is not None
    if not d.accepted:
        assert d.class_id == "unknown"
        assert "photo" in d.advisory.message.lower()
    else:
        # Accepting is only defensible if it is called background/not-a-leaf.
        assert d.category == "background", (
            f"noise diagnosed as an actionable {d.class_id} at {d.confidence:.2f}"
        )


def test_accepted_diagnosis_carries_actionable_advice(bundle, trained_run):
    from cropguard.data.manifest import read_manifest
    from cropguard.edge.runtime import EdgeClassifier

    _, manifest = trained_run
    clf = EdgeClassifier(bundle)
    for record in read_manifest(manifest):
        if record.class_id != "tomato__late_blight":
            continue
        d = clf.diagnose(record.path)
        if d.accepted and d.class_id == "tomato__late_blight":
            assert d.advisory is not None
            assert d.advisory.steps
            assert len(d.advisory.to_sms()) <= 160
            assert d.advisory.urgency in ("warning", "critical")
            return
    pytest.skip("model never confidently identified late blight at this scale")


def test_diagnoses_feed_the_early_warning_tracker(bundle, trained_run):
    from datetime import datetime, timedelta, timezone

    from cropguard.data.manifest import read_manifest
    from cropguard.early_warning import PestPressureTracker
    from cropguard.edge.runtime import EdgeClassifier

    _, manifest = trained_run
    clf = EdgeClassifier(bundle)
    tracker = PestPressureTracker()
    now = datetime.now(timezone.utc)

    for i, record in enumerate(read_manifest(manifest)[:40]):
        tracker.add_diagnosis(
            clf.diagnose(record.path, with_advisory=False),
            field_id="f1",
            timestamp=now - timedelta(days=i % 5),
        )
    for alert in tracker.evaluate(field_id="f1", now=now):
        assert alert.message and alert.evidence
        assert len(alert.to_sms()) <= 160


def test_canopy_tiling_returns_a_field_level_summary(bundle, trained_run):
    from cropguard.data.manifest import read_manifest
    from cropguard.edge.runtime import EdgeClassifier

    _, manifest = trained_run
    clf = EdgeClassifier(bundle)
    path = read_manifest(manifest)[0].path
    wide = Image.open(path).resize((320, 240))
    summary = clf.diagnose_canopy(wide, max_tiles=6)
    assert 1 <= summary["tiles"] <= 6
    assert 0.0 <= summary["affected_tile_fraction"] <= 1.0
    assert "advisory" in summary


def test_evaluation_reproduces_and_writes_artefacts(trained_run):
    from cropguard.evaluate import evaluate_run

    run_dir, manifest = trained_run
    result = evaluate_run(run_dir, manifest=manifest, split="test", num_workers=0)
    assert result["samples"] > 0
    assert 0.0 <= result["label"]["macro_f1"] <= 1.0
    assert result["calibration"]["ece_calibrated"] <= result["calibration"]["ece_uncalibrated"] + 1e-6
    assert (run_dir / "eval" / "test_report.json").exists()
    assert (run_dir / "eval" / "test_confusion.csv").exists()
    assert (run_dir / "eval" / "test_predictions.csv").exists()


def test_benchmark_reports_a_latency_budget(bundle):
    from cropguard.benchmark import benchmark_bundle

    result = benchmark_bundle(bundle, runs=6, warmup=2)
    assert result.latency_ms["p50"] > 0
    assert result.throughput_ips > 0
    assert "p95" in result.format()
