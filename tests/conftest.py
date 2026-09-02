import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def taxonomy():
    from cropguard.taxonomy import load_taxonomy

    return load_taxonomy()


@pytest.fixture(scope="session")
def small_classes():
    return [
        "background",
        "tomato__healthy",
        "tomato__late_blight",
        "pest__aphid",
        "deficiency__iron",
    ]


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory, small_classes):
    """A tiny ImageFolder dataset, generated once per test session."""
    from cropguard.data.synthetic import generate_dataset

    out = tmp_path_factory.mktemp("synth")
    return generate_dataset(out, class_ids=small_classes, per_class=8, size=64, seed=3)


@pytest.fixture(scope="session")
def manifest(tmp_path_factory, synthetic_root):
    from cropguard.data.manifest import (
        scan_image_folder,
        stratified_group_split,
        write_manifest,
    )

    records, unmapped = scan_image_folder(synthetic_root)
    assert not unmapped
    records = stratified_group_split(records, val_frac=0.25, test_frac=0.25, seed=5)
    path = tmp_path_factory.mktemp("manifest") / "manifest.csv"
    return write_manifest(records, path)
