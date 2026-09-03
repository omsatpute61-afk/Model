"""Ingest, cleaning and EDA for the real corpora (DLCPD-25, AP162).

The failures these guard against are all silent ones: a nested dataset read as
flat, a class folder resolved to the wrong crop's disease, near-duplicates
straddling a split. None of them raise; they just make the reported accuracy a
lie.
"""

import numpy as np
import pytest
from PIL import Image

from cropguard.data.clean import (
    CleaningConfig,
    cap_class_size,
    clean_records,
    drop_rare_classes,
    find_near_duplicates,
    inspect_image,
)
from cropguard.data.eda import analyse, gini
from cropguard.data.ingest import (
    detect_layout,
    scan_flat,
    load_dataset_map,
    parse_life_stage,
    scan_ap162,
    scan_nested,
)
from cropguard.data.manifest import Record, default_group_key, stratified_group_split

TEN_CROPS = ["wheat", "cotton", "maize", "soybean", "potato",
             "tomato", "chilli", "mango", "citrus", "grape"]


def _img(path, size=(64, 64), seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (*size, 3)).astype("uint8")).save(path)
    return path


@pytest.fixture
def dlcpd(tmp_path):
    root = tmp_path / "DLCPD-25"
    plan = {
        "Corn": ["Corn___Common_rust_", "Northern_Leaf_Blight", "healthy"],
        "Tomato": ["Tomato___Late_blight", "Tomato___healthy"],
        "Soybean": ["Rust", "Yellow_mosaic", "healthy"],
        "Wheat": ["Yellow_rust"],
        "Cotton": ["Bacterial_blight"],
        "Citrus": ["Greening"],
        "Vitis": ["Black_rot"],
        "Bell Pepper": ["Bacterial_spot"],
        "Rice": ["Leaf_blast"],       # out of scope
        "Alfalfa": ["Leaf_spot"],     # not in the taxonomy at all
    }
    for i, (crop, classes) in enumerate(plan.items()):
        for j, cls in enumerate(classes):
            for k in range(3):
                _img(root / crop / cls / f"{cls}_{k:03d}.jpg", seed=i * 100 + j * 10 + k)
    return root


# ---------------------------------------------------------------- ingest
def test_nested_layout_is_detected(dlcpd, tmp_path):
    assert detect_layout(dlcpd) == "nested"
    flat = tmp_path / "flat"
    _img(flat / "Tomato___Late_blight" / "a.jpg")
    assert detect_layout(flat) == "flat"


def test_nested_scan_maps_crop_and_class(dlcpd):
    report = scan_nested(dlcpd, source="DLCPD-25", crops=TEN_CROPS)
    resolved = {r.class_id for r in report.records}
    assert "maize__common_rust" in resolved      # Corn___Common_rust_ under Corn/
    assert "maize__northern_leaf_blight" in resolved
    assert "maize__healthy" in resolved          # bare "healthy" + crop context
    assert "chilli__bacterial_spot" in resolved  # "Bell Pepper" -> chilli
    assert "grape__black_rot" in resolved        # "Vitis" -> grape


def test_bare_class_name_cannot_resolve_to_another_crop(dlcpd):
    """Regression: Soybean/Rust resolved to sugarcane__rust.

    'Rust' is a legitimate alias for several crops' diseases. Resolving the
    folder without honouring the crop above it mislabels an entire class, and
    nothing in training or evaluation would ever flag it.
    """
    report = scan_nested(dlcpd, source="DLCPD-25", crops=TEN_CROPS)
    by_folder = {}
    from pathlib import Path

    for r in report.records:
        by_folder[f"{Path(r.path).parent.parent.name}/{Path(r.path).parent.name}"] = r.class_id
    assert by_folder["Soybean/Rust"] == "soybean__rust"
    assert by_folder["Cotton/Bacterial_blight"] == "cotton__bacterial_blight"
    assert by_folder["Wheat/Yellow_rust"] == "wheat__stripe_rust"


def test_out_of_scope_crops_are_excluded_not_dropped_silently(dlcpd):
    report = scan_nested(dlcpd, source="DLCPD-25", crops=TEN_CROPS)
    assert any("Rice" in k for k in report.excluded)
    assert not any(r.class_id.startswith("rice__") for r in report.records)


def test_unknown_class_is_reported_for_a_human(dlcpd):
    report = scan_nested(dlcpd, source="DLCPD-25", crops=None)
    assert any("Alfalfa" in k for k in report.unmapped)


# ---------------------------------------------------------------- AP162
@pytest.fixture
def ap162(tmp_path):
    root = tmp_path / "AP162"
    classes = {
        0: "Aphidoidea", 28: "Bemisia tabaci",
        10: "Spodoptera frugiperda larva", 124: "Spodoptera frugiperda",
        108: "Corythucha ciliata",   # deliberately excluded (ornamental)
    }
    (tmp_path / "classes.txt").write_text(
        "\n".join(f"{k}\t{v}" for k, v in classes.items())
    )
    for split in ("train", "test"):
        for k, v in classes.items():
            for i in range(2):
                _img(root / split / f"{k}_{v.replace(' ', '_')}" / f"{k}_{split}_{i}.jpg",
                     seed=k * 10 + i)
    return root, tmp_path / "classes.txt"


def test_ap162_merges_larva_and_adult_into_one_class(ap162):
    root, classes = ap162
    report = scan_ap162(root, classes_file=classes)
    faw = [r for r in report.records if r.class_id == "pest__fall_armyworm"]
    assert len(faw) == 8                                   # 2 classes x 2 splits x 2 images
    assert {r.life_stage for r in faw} == {"larva", "adult"}


def test_ap162_excludes_out_of_scope_species(ap162):
    root, classes = ap162
    report = scan_ap162(root, classes_file=classes)
    assert any("108" in k for k in report.excluded)
    assert report.unmapped == {}


def test_ap162_map_covers_every_class_exactly_once():
    """Every one of the 162 classes is either mapped or excluded with a reason."""
    spec = load_dataset_map("ap162")
    mapped = {int(k) for k in spec["map"]}
    excluded = {int(k) for k in spec["excluded"]}
    assert mapped & excluded == set()
    assert mapped | excluded == set(range(162))
    for reason in spec["excluded"].values():
        assert len(reason) > 20, "an exclusion needs a real reason, not a shrug"


def test_ap162_map_targets_exist_in_the_taxonomy(taxonomy):
    spec = load_dataset_map("ap162")
    for index, (class_id, stage) in spec["map"].items():
        assert class_id in taxonomy, f"AP162 {index} maps to unknown class {class_id}"
        assert stage in ("egg", "larva", "nymph", "adult", "any")


@pytest.mark.parametrize("name,expected", [
    ("Spodoptera frugiperda larva", "larva"),
    ("Helicoverpa armigera", "unknown"),
    ("Aphid nymph", "nymph"),
    ("egg mass", "egg"),
])
def test_life_stage_parsing(name, expected):
    assert parse_life_stage(name) == expected


# ---------------------------------------------------------------- grouping
def test_group_key_is_unique_across_identically_named_folders(tmp_path):
    """Regression: every crop has a 'healthy' folder with 'healthy_0000.jpg'.

    A group key built from the folder NAME merged maize and soybean images into
    one group, which then straddled two splits and showed up as leakage.
    """
    a = _img(tmp_path / "Corn" / "healthy" / "healthy_0000.jpg")
    b = _img(tmp_path / "Soybean" / "healthy" / "healthy_0000.jpg")
    assert default_group_key(a) != default_group_key(b)


def test_nested_ingest_produces_a_leak_free_split(dlcpd):
    report = scan_nested(dlcpd, source="DLCPD-25", crops=TEN_CROPS)
    records = stratified_group_split(report.records, 0.2, 0.2, seed=3)
    groups = {}
    for r in records:
        groups.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in groups.values())


# ---------------------------------------------------------------- cleaning
def test_unreadable_image_is_reported_not_raised(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"\xff\xd8 definitely not a jpeg")
    facts = inspect_image(bad)
    assert not facts.ok and facts.error


def test_cleaner_catches_every_planted_defect(tmp_path):
    rng = np.random.default_rng(0)
    recs = []

    def add(path, cls):
        recs.append(Record(path=str(path), class_id=cls, category="disease"))

    for i in range(12):
        add(_img(tmp_path / "ok" / f"{i}.jpg", (128, 128), seed=i), "tomato__late_blight")

    shared = rng.integers(0, 255, (128, 128, 3)).astype("uint8")
    for name, cls in (("orig.jpg", "tomato__late_blight"),
                      ("copy.jpg", "tomato__late_blight"),
                      ("mislabelled.jpg", "tomato__early_blight")):
        p = tmp_path / "dup" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(shared).save(p)
        add(p, cls)

    tiny = _img(tmp_path / "bad" / "tiny.jpg", (16, 16))
    add(tiny, "tomato__late_blight")
    blank = tmp_path / "bad" / "blank.jpg"
    Image.fromarray(np.full((128, 128, 3), 120, dtype="uint8")).save(blank)
    add(blank, "tomato__late_blight")
    broken = tmp_path / "bad" / "broken.jpg"
    broken.write_bytes(b"not an image")
    add(broken, "tomato__late_blight")

    report = clean_records(recs, CleaningConfig(min_side=64))
    reasons = {k for k, v in report.dropped.items() if v}
    assert {"too_small", "blank", "unreadable"} <= reasons
    assert report.cross_label_pairs, "identical images under two labels must be reported"
    assert len(report.kept) == 12


def test_near_duplicates_are_grouped(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (200, 200, 3)).astype("uint8")
    a = tmp_path / "a.jpg"
    Image.fromarray(base).save(a, quality=95)
    b = tmp_path / "b.jpg"          # same picture, heavily recompressed
    Image.fromarray(base).save(b, quality=30)
    c = _img(tmp_path / "c.jpg", (200, 200), seed=99)

    facts = [inspect_image(p) for p in (a, b, c)]
    groups = find_near_duplicates(facts, max_distance=4)
    assert len(groups) == 1
    assert {str(a), str(b)} == set(groups[0])


def test_rare_classes_are_dropped_with_a_report():
    recs = [Record(path=f"a{i}", class_id="common", category="disease") for i in range(50)]
    recs += [Record(path=f"b{i}", class_id="rare", category="pest") for i in range(4)]
    kept, dropped = drop_rare_classes(recs, min_images=25)
    assert dropped == {"rare": 4}
    assert len(kept) == 50


def test_class_cap_is_deterministic():
    recs = [Record(path=f"a{i}", class_id="big", category="disease") for i in range(100)]
    first, _ = cap_class_size(recs, 20, seed=7)
    second, _ = cap_class_size(recs, 20, seed=7)
    assert [r.path for r in first] == [r.path for r in second]
    assert len(first) == 20


# ---------------------------------------------------------------- EDA
def test_gini_reflects_imbalance():
    assert gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-6)
    assert gini([1, 1, 1, 1000]) > 0.6


def test_eda_flags_the_problems_that_change_training():
    recs = [Record(path=f"big{i}", class_id="tomato__late_blight", category="disease",
                   group=f"g{i}", split="train") for i in range(400)]
    recs += [Record(path=f"rare{i}", class_id="mango__anthracnose", category="disease",
                    group=f"h{i}", split="train") for i in range(6)]
    payload = analyse(recs, min_train=25).payload

    assert payload["balance"]["imbalance_ratio"] > 20
    assert "mango__anthracnose" in payload["balance"]["classes_under_min_train"]
    joined = " ".join(payload["recommendations"]).lower()
    assert "imbalance" in joined
    assert "too few images" in joined
    # no val or test rows at all -> must be called out
    assert payload["splits"]["classes_missing_val"]


def test_eda_report_renders(tmp_path):
    recs = [Record(path=f"p{i}", class_id="tomato__healthy", category="healthy",
                   group=f"g{i}", split="train" if i % 4 else "val") for i in range(40)]
    result = analyse(recs)
    paths = result.save(tmp_path)
    text = paths["markdown"].read_text()
    assert "# Dataset EDA" in text
    assert "Class balance" in text
    assert paths["json"].exists()


# ------------------------------------------------- source-kind enforcement
@pytest.fixture
def pestopia(tmp_path):
    """A pest corpus with fungal and bacterial classes mixed in, as scraped
    pest datasets routinely are."""
    root = tmp_path / "Pestopia"
    pests = ["aphids", "armyworm", "Pink Bollworm", "stem borer", "thrips",
             "whitefly", "mites", "grasshopper", "mealybug", "termites"]
    diseases = ["Anthracnose", "Leaf Blight", "Powdery Mildew", "Fusarium Wilt",
                "Bacterial Leaf Spot", "Root Rot", "Tomato___Late_blight",
                "Downy Mildew", "Fungal Leaf Spot"]
    for i, cls in enumerate(pests + diseases):
        for k in range(3):
            _img(root / cls / f"{k}.jpg", seed=i * 10 + k)
    return root, pests, diseases


def test_pest_source_rejects_every_disease_class(pestopia):
    """A fungal class in a pest dataset is not a quirk to tolerate.

    Training on it puts the same condition on both branches of the model with
    labels from two different sources, and corrupts the pest branch's
    life-stage and economic-threshold logic - a fungus has neither.
    """
    root, pests, diseases = pestopia
    report = scan_flat(root, source="Pestopia", source_kind="pest")

    assert {r.category for r in report.records} == {"pest"}
    assert len(report.wrong_kind) == len(diseases)
    for cls in diseases:
        assert cls in report.wrong_kind, f"{cls} was not rejected"
    for cls in pests:
        assert cls not in report.wrong_kind


def test_every_rejection_states_a_reason(pestopia):
    root, _, _ = pestopia
    report = scan_flat(root, source="Pestopia", source_kind="pest")
    for folder, reason in report.wrong_kind.items():
        assert len(reason) > 15, f"{folder} rejected without a usable reason"
    assert report.summary()["wrong_kind_images"] > 0


def test_known_disease_class_is_caught_by_the_category_gate(pestopia):
    """'Tomato___Late_blight' resolves in the taxonomy, so the gate catches it
    rather than the name heuristic."""
    root, _, _ = pestopia
    report = scan_flat(root, source="Pestopia", source_kind="pest")
    assert "resolved to tomato__late_blight" in report.wrong_kind["Tomato___Late_blight"]


def test_unknown_disease_class_is_caught_by_the_name_heuristic(pestopia):
    """'Fungal Leaf Spot' is in no taxonomy, which is exactly the case a
    hard-coded list of known diseases would miss."""
    root, _, _ = pestopia
    report = scan_flat(root, source="Pestopia", source_kind="pest")
    assert "not an insect" in report.wrong_kind["Fungal Leaf Spot"]


def test_mixed_kind_keeps_everything_it_can_resolve(pestopia):
    """The rejection is a property of the source, not of the class."""
    root, _, _ = pestopia
    strict = scan_flat(root, source="Pestopia", source_kind="pest")
    relaxed = scan_flat(root, source="Pestopia", source_kind="mixed")
    assert relaxed.image_count > strict.image_count
    assert "disease" in {r.category for r in relaxed.records}


def test_disease_source_rejects_insect_classes(tmp_path):
    """Enforcement runs both ways."""
    root = tmp_path / "Diseases"
    _img(root / "Tomato___Late_blight" / "a.jpg", seed=1)
    _img(root / "aphids" / "b.jpg", seed=2)
    report = scan_flat(root, source="D", source_kind="disease")
    assert [r.class_id for r in report.records] == ["tomato__late_blight"]
    assert "aphids" in report.wrong_kind


@pytest.mark.parametrize("name", [
    "Anthracnose", "Leaf Blight", "Powdery Mildew", "Fusarium Wilt",
    "Bacterial Spot", "Root Rot", "leafspot", "Downy Mildew", "Citrus canker",
])
def test_disease_names_are_recognised(name):
    from cropguard.data.ingest import looks_like_disease

    assert looks_like_disease(name)


@pytest.mark.parametrize("name", [
    "aphids", "Stem Borer", "leaf miner", "armyworm", "whitefly", "mealybug",
    "grasshopper", "thrips", "root grub", "ants", "Fruit Fly", "spider mites",
])
def test_pest_names_are_not_mistaken_for_diseases(name):
    """Regression: the pest hint 'ant' substring-matched inside 'anthracnose',
    letting a fungal class through as an insect."""
    from cropguard.data.ingest import looks_like_disease

    assert not looks_like_disease(name)


# ------------------------------------------------- bare-name resolution
def test_plain_pest_folder_names_resolve(taxonomy):
    """Pest datasets use plain folder names; class ids carry a 'pest__' prefix."""
    for name, expected in [
        ("grasshopper", "pest__grasshopper"),
        ("psyllid", "pest__psyllid"),
        ("stem borer", "pest__stem_borer"),
        ("mole cricket", "pest__mole_cricket"),
        ("bagworm", "pest__bagworm"),
    ]:
        resolved = taxonomy.resolve(name)
        assert resolved is not None and resolved.id == expected, name


def test_ambiguous_bare_names_stay_unresolved(taxonomy):
    """Four crops have a powdery mildew. Picking one silently would mislabel a
    whole class, so the folder is left for a human instead."""
    for name in ("powdery mildew", "anthracnose", "late blight", "rust", "healthy"):
        assert taxonomy.resolve(name) is None, name
        assert name in taxonomy.ambiguous_bare_names
        assert len(taxonomy.ambiguous_bare_names[name]) > 1
