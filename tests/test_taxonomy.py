"""The taxonomy is the model's contract; a break here mislabels every output."""

import pytest

from cropguard.taxonomy import CATEGORIES, Taxonomy, normalise_alias


def test_ids_and_aliases_are_unique(taxonomy):
    ids = taxonomy.class_ids
    assert len(ids) == len(set(ids))
    seen = {}
    for c in taxonomy:
        for alias in c.aliases:
            key = normalise_alias(alias)
            assert key not in seen, f"alias {alias!r} shared by {seen.get(key)} and {c.id}"
            seen[key] = c.id


def test_every_category_is_known(taxonomy):
    assert {c.category for c in taxonomy} <= set(CATEGORIES)


def test_plantvillage_folder_names_resolve(taxonomy):
    cases = {
        "Tomato___Late_blight": "tomato__late_blight",
        "Corn_(maize)___Common_rust_": "maize__common_rust",
        "Potato___healthy": "potato__healthy",
        "Grape___Esca_(Black_Measles)": "grape__esca",
        "Tomato___Spider_mites Two-spotted_spider_mite": "pest__spider_mite",
    }
    for folder, expected in cases.items():
        assert taxonomy.resolve(folder).id == expected, folder


def test_alias_matching_ignores_punctuation_and_case(taxonomy):
    assert taxonomy.resolve("tomato   LATE   blight").id == "tomato__late_blight"
    assert taxonomy.resolve("Corn (maize) Common rust").id == "maize__common_rust"


def test_unknown_folder_returns_none_rather_than_guessing(taxonomy):
    assert taxonomy.resolve("Banana___Panama_wilt") is None


def test_pest_classes_carry_an_economic_threshold(taxonomy):
    for c in taxonomy:
        if c.category == "pest":
            assert c.etl, f"{c.id} has no ETL - advisories cannot say when to spray"
            assert c.etl.get("threshold", 0) > 0


def test_index_order_is_stable_and_round_trips(taxonomy):
    for i, c in enumerate(taxonomy):
        assert taxonomy.index_of(c.id) == i
        assert taxonomy[i].id == c.id


def test_subset_preserves_requested_order(taxonomy):
    ids = ["pest__aphid", "tomato__healthy", "rice__blast"]
    sub = taxonomy.subset(ids)
    assert sub.class_ids == tuple(ids)
    assert len(sub) == 3


def test_filter_by_crop_keeps_generic_pests(taxonomy):
    cotton = taxonomy.filter(crops=["cotton"])
    assert "cotton__leaf_curl_virus" in cotton
    assert "pest__whitefly" in cotton      # crop == "any"
    assert "rice__blast" not in cotton


def test_duplicate_ids_are_rejected(taxonomy):
    dup = [taxonomy[0], taxonomy[0]]
    with pytest.raises(ValueError):
        Taxonomy(dup)


def test_category_vector_matches_classes(taxonomy):
    vec = taxonomy.category_vector()
    assert len(vec) == len(taxonomy)
    for i, c in enumerate(taxonomy):
        assert CATEGORIES[vec[i]] == c.category


def test_healthy_and_background_are_not_actionable(taxonomy):
    assert not taxonomy["tomato__healthy"].is_actionable
    assert not taxonomy["background"].is_actionable
    assert taxonomy["tomato__late_blight"].is_actionable
    assert taxonomy["abiotic__water_stress"].is_actionable


def test_virus_vectors_point_at_real_pest_classes(taxonomy):
    for c in taxonomy:
        if c.vector:
            assert c.vector in taxonomy, f"{c.id} names an unknown vector {c.vector}"
            assert taxonomy[c.vector].category == "pest"
