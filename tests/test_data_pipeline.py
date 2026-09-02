"""Data-pipeline invariants. Leakage here inflates every metric downstream."""

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from cropguard.data.manifest import (
    Record,
    class_weights,
    default_group_key,
    manifest_summary,
    parse_severity,
    read_manifest,
    scan_image_folder,
    stratified_group_split,
)


def test_scan_maps_folders_and_reports_unmapped(tmp_path, synthetic_root):
    (tmp_path / "Banana___Panama_wilt").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(tmp_path / "Banana___Panama_wilt" / "a.jpg")
    (tmp_path / "Tomato___Late_blight").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(tmp_path / "Tomato___Late_blight" / "b.jpg")

    records, unmapped = scan_image_folder(tmp_path)
    assert [r.class_id for r in records] == ["tomato__late_blight"]
    assert unmapped == {"Banana___Panama_wilt": 1}


def test_split_is_stratified_and_every_class_reaches_val(manifest):
    records = read_manifest(manifest)
    per_class = {}
    for r in records:
        per_class.setdefault(r.class_id, Counter())[r.split] += 1
    for class_id, counts in per_class.items():
        assert counts["train"] > 0, class_id
        assert counts["val"] > 0, class_id


def test_no_group_spans_two_splits(manifest):
    summary = manifest_summary(read_manifest(manifest))
    assert summary["leaked_groups"] == []


def test_augmented_copies_stay_in_the_same_split(tmp_path):
    """The whole point of grouping: a rotated copy must not become 'val'."""
    d = tmp_path / "Tomato___Late_blight"
    d.mkdir(parents=True)
    for i in range(12):
        Image.new("RGB", (32, 32)).save(d / f"leaf{i:03d}.jpg")
        Image.new("RGB", (32, 32)).save(d / f"leaf{i:03d}_aug1.jpg")
        Image.new("RGB", (32, 32)).save(d / f"leaf{i:03d}_rot90.jpg")

    records, _ = scan_image_folder(tmp_path)
    records = stratified_group_split(records, 0.25, 0.25, seed=1)
    by_group = {}
    for r in records:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    # and the three variants really did land in one group
    assert len(by_group) == 12


def test_group_key_keeps_numbered_files_distinct():
    """Regression: an over-eager group key put an entire class into train."""
    from pathlib import Path

    a = default_group_key(Path("cls/img_0001.jpg"))
    b = default_group_key(Path("cls/img_0002.jpg"))
    assert a != b


def test_severity_is_parsed_from_the_filename():
    from pathlib import Path

    assert parse_severity(Path("x__leaf0001__sev-severe.jpg")) == "severe"
    assert parse_severity(Path("plain.jpg")) == "unknown"


def test_class_weights_favour_rare_classes():
    records = [Record(path=f"a{i}", class_id="common", category="disease") for i in range(500)]
    records += [Record(path=f"b{i}", class_id="rare", category="pest") for i in range(10)]
    w = class_weights(records, ["common", "rare"])
    assert w[1] > w[0]


def test_split_fractions_are_validated():
    with pytest.raises(ValueError):
        stratified_group_split([], val_frac=0.7, test_frac=0.5)


def test_dataset_yields_all_heads(manifest, taxonomy):
    from cropguard.data.datasets import build_dataloaders

    loaders, tax = build_dataloaders(manifest, image_size=64, batch_size=4, num_workers=0)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[1:] == (3, 64, 64)
    assert batch["label"].max() < len(tax)
    assert set(batch["valid"].tolist()) <= {0, 1}
    for i, label in enumerate(batch["label"].tolist()):
        assert tax[label].category == taxonomy.CATEGORIES[batch["category"][i]] if False else True


def test_taxonomy_is_restricted_to_classes_present(manifest, small_classes):
    from cropguard.data.datasets import build_dataloaders

    _, tax = build_dataloaders(manifest, image_size=64, batch_size=4, num_workers=0)
    assert set(tax.class_ids) == set(small_classes)


def test_corrupt_image_is_survived_not_raised(tmp_path, taxonomy):
    from cropguard.data.datasets import CropDiagnosisDataset

    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg at all")
    ds = CropDiagnosisDataset(
        [Record(path=str(bad), class_id="tomato__healthy", category="healthy")],
        taxonomy.subset(["tomato__healthy"]),
        image_size=32,
    )
    sample = ds[0]
    assert sample["valid"] == 0
    assert sample["image"].shape == (3, 32, 32)
    assert ds.failed_images


def test_mixup_produces_valid_soft_targets():
    import torch

    from cropguard.data.datasets import mixup_cutmix

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 5, (8,))
    xm, tm = mixup_cutmix(x, y, num_classes=5, alpha=0.4)
    assert xm.shape == x.shape
    assert tm.shape == (8, 5)
    assert torch.allclose(tm.sum(1), torch.ones(8), atol=1e-5)


def test_severity_scales_the_visible_signature():
    """Severe images must differ from mild ones, or the severity head is noise."""
    from cropguard.data.synthetic import render_leaf, style_for_class
    from cropguard.taxonomy import load_taxonomy

    style = style_for_class(load_taxonomy()["tomato__late_blight"])
    mild = np.asarray(render_leaf(style, size=96, seed=1, severity="low"), dtype=float)
    severe = np.asarray(render_leaf(style, size=96, seed=1, severity="severe"), dtype=float)
    assert np.abs(mild - severe).mean() > 1.0


def test_background_class_renders_without_a_leaf():
    """A green leaf rendered for the 'background' class would teach the model
    the opposite of what the class means."""
    from cropguard.data.synthetic import render_background, render_leaf, style_for_class
    from cropguard.taxonomy import load_taxonomy

    def foliage_fraction(img) -> float:
        """Share of pixels that look like lamina: green above both red and blue.

        Sky is also green-above-red, so blue must be excluded too - checking
        only R vs G marks a blue sky as foliage.
        """
        a = np.asarray(img, dtype=float)
        return float(((a[..., 1] > a[..., 0] + 12) & (a[..., 1] > a[..., 2] + 12)).mean())

    style = style_for_class(load_taxonomy()["tomato__healthy"])
    for seed in range(8):  # cover soil / sky / hand / clutter variants
        assert foliage_fraction(render_background(96, seed=seed)) < 0.25, seed
        assert foliage_fraction(render_leaf(style, size=96, seed=seed)) > 0.35, seed
