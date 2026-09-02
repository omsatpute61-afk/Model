"""Advisory text is what the farmer acts on; a wrong urgency costs money."""

import pytest

from cropguard.advisory import URGENCY_ORDER, default_engine


@pytest.fixture(scope="module")
def engine():
    return default_engine()


def test_every_taxonomy_class_gets_an_advisory(engine, taxonomy):
    for c in taxonomy:
        a = engine.advise(c.id, confidence=0.9)
        assert a.headline and a.message
        assert a.urgency in URGENCY_ORDER
        assert a.class_id == c.id


def test_actionable_classes_have_concrete_steps(engine, taxonomy):
    for c in taxonomy:
        if c.is_actionable:
            a = engine.advise(c.id, confidence=0.9)
            assert a.steps, f"{c.id} gives no next step"


def test_low_confidence_downgrades_and_asks_for_another_photo(engine):
    high = engine.advise("tomato__late_blight", confidence=0.95)
    low = engine.advise("tomato__late_blight", confidence=0.30)
    assert low.urgency_rank < high.urgency_rank
    assert low.needs_confirmation
    assert any("second" in n.lower() or "photo" in n.lower() for n in low.notes)


def test_fast_spreading_disease_escalates_when_confident(engine):
    a = engine.advise("tomato__late_blight", confidence=0.97)
    assert a.urgency == "critical"


def test_heavy_infestation_escalates_and_light_one_says_spot_treat(engine):
    heavy = engine.advise("pest__fall_armyworm", confidence=0.9, affected_fraction=0.25)
    light = engine.advise("pest__fall_armyworm", confidence=0.9, affected_fraction=0.01)
    assert heavy.urgency_rank > light.urgency_rank
    assert any("spot-treat" in n for n in light.notes)


def test_vector_borne_disease_mentions_its_vector(engine):
    a = engine.advise("tomato__yellow_leaf_curl_virus", confidence=0.9)
    assert any("whitefly" in n.lower() for n in a.notes)


def test_pest_advisory_quotes_the_economic_threshold(engine):
    a = engine.advise("pest__brown_planthopper", confidence=0.9)
    assert any("Economic threshold" in n for n in a.notes)


def test_healthy_is_not_an_alert(engine):
    a = engine.advise("tomato__healthy", confidence=0.99)
    assert a.urgency == "none"
    joined = " ".join(a.steps).lower()
    assert not a.steps or "scouting" in joined or "monitor" in joined


def test_sms_fits_one_segment_and_never_cuts_a_word(engine, taxonomy):
    for c in taxonomy:
        sms = engine.advise(c.id, confidence=0.9).to_sms()
        assert len(sms) <= 160, c.id
        if sms.endswith("..."):
            assert not sms[:-3].endswith(" ")


def test_unknown_class_falls_back_to_a_safe_advisory(engine):
    a = engine.advise("not__a__real__class", confidence=0.9)
    assert a.class_id == "unknown"
    assert a.needs_confirmation
    assert "kvk" in a.message.lower() or "officer" in a.message.lower()


def test_uncertain_advisory_lists_candidates(engine):
    a = engine.advise_uncertain([("tomato__early_blight", 0.3), ("tomato__target_spot", 0.2)])
    assert not a.class_id == "tomato__early_blight"
    assert any("Early blight" in n for n in a.notes)


def test_bacterial_disease_does_not_recommend_a_plain_fungicide(engine):
    a = engine.advise("rice__bacterial_leaf_blight", confidence=0.9)
    assert "fungicide" in a.chemical_guidance.lower()
    assert "ineffective" in a.chemical_guidance.lower() or "not work" in a.message.lower()


def test_every_advisory_carries_the_disclaimer(engine):
    a = engine.advise("tomato__late_blight", confidence=0.9)
    assert "label" in a.disclaimer.lower()


def test_generic_messages_have_hindi(engine):
    for key in ("irrigate_now", "pest_increasing", "flood_risk", "heat_stress"):
        assert engine.message(key, "hi") != engine.message(key, "en")


def test_advisory_serialises_to_json(engine):
    import json

    d = json.loads(engine.advise("pest__whitefly", confidence=0.8, severity="moderate").to_json())
    assert d["class_id"] == "pest__whitefly"
    assert d["confidence"] == 0.8
