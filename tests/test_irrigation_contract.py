"""The boundary between this model and the FasalSetu irrigation engine.

These are contract tests, not unit tests: each one pins down a promise the
irrigation subsystem is entitled to rely on. If one fails, an integration is
broken, not a helper.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from cropguard.advisory import URGENCY_ORDER, default_engine
from cropguard.early_warning import ALERT_LEVELS, Detection, PestPressureTracker
from cropguard.irrigation_contract import (
    CONSTRAINT_KINDS,
    MAX_TTL_HOURS,
    CONSTRAINT_PHRASES,
    NO_CONSTRAINT,
    IrrigationConstraint,
    merge,
    validate_flags,
)
from cropguard.taxonomy import load_taxonomy

#: Fixed so the window arithmetic in the tracker is deterministic.
NOW = datetime(2025, 7, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    return default_engine()


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


# -- the vocabulary itself ---------------------------------------------------
def test_every_flag_has_a_kind_and_a_phrase():
    assert set(CONSTRAINT_KINDS) == set(CONSTRAINT_PHRASES)
    assert set(CONSTRAINT_KINDS.values()) == {
        "veto",
        "drainage",
        "method",
        "timing",
        "hint",
    }


def test_unknown_flag_raises_rather_than_being_ignored():
    # The whole point of a vocabulary: a typo must not silently do nothing.
    with pytest.raises(ValueError, match="unknown irrigation constraint"):
        validate_flags(["no_overhaed"])
    with pytest.raises(ValueError):
        IrrigationConstraint(flags=("drench_the_field",))


def test_bad_water_status_raises():
    with pytest.raises(ValueError, match="water_status_evidence"):
        IrrigationConstraint(water_status_evidence="damp")


# -- precedence rule 1: no water quantity crosses the boundary ---------------
def test_no_constraint_field_carries_a_water_quantity(engine, taxonomy):
    """The engine owns volume and duration. ``ttl_hours`` is the only number
    that crosses, and it is a validity horizon, not an amount of water."""
    numeric = {
        k
        for c in taxonomy.classes
        for k, v in engine.advise(c.id, confidence=0.9)
        .irrigation_constraint.to_dict()
        .items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert numeric == {"ttl_hours", "priority"}


def test_constraint_phrases_carry_no_units():
    units = re.compile(r"\b(litre|liter|mm|millimet|hour|minute|cm|%)", re.I)
    for phrase in CONSTRAINT_PHRASES.values():
        assert not units.search(phrase), phrase


# -- precedence rule 2: exactly one veto, and only waterlogging raises it ----
def test_waterlogging_is_the_only_class_that_blocks_irrigation(engine, taxonomy):
    blocking = [
        c.id
        for c in taxonomy.classes
        if engine.advise(c.id, confidence=0.9).irrigation_constraint.blocks_irrigation
    ]
    assert blocking == ["abiotic__waterlogging"]


def test_waterlogging_suspends_and_drains(engine):
    c = engine.advise("abiotic__waterlogging", confidence=0.95).irrigation_constraint
    assert c.blocks_irrigation and c.needs_drainage
    assert "suspend" in c.flags
    assert c.water_status_evidence == "waterlogged"


def test_drain_alone_does_not_block_the_next_irrigation(engine):
    """Rice after bacterial leaf blight is drained and then watered shallowly
    again. Treating drainage as a veto would leave the crop dry."""
    c = engine.advise(
        "rice__bacterial_leaf_blight", confidence=0.9
    ).irrigation_constraint
    assert c.needs_drainage
    assert not c.blocks_irrigation


def test_a_leaf_disease_never_vetoes_irrigation(engine):
    """Late blight changes how you water, not whether. The engine's own
    RAIN_HOLD covers the wet spell the prose talks about."""
    for cid in ("tomato__late_blight", "potato__late_blight", "grape__downy_mildew"):
        c = engine.advise(cid, confidence=0.97).irrigation_constraint
        assert not c.blocks_irrigation, cid
        assert "no_overhead" in c.flags and "prefer_furrow" in c.flags, cid


# -- precedence rule 3: hints lose to standing water -------------------------
def test_veto_overrides_the_avoid_stress_hint():
    c = merge(
        [
            IrrigationConstraint(flags=("avoid_stress",), reasons=("Spider mite",)),
            IrrigationConstraint(
                flags=("suspend", "drain"),
                reasons=("Waterlogging",),
                water_status_evidence="waterlogged",
            ),
        ]
    )
    assert c.flags == ("suspend", "drain")
    assert c.overridden == ("avoid_stress",)  # recorded, not silently dropped
    assert set(c.reasons) == {"Spider mite", "Waterlogging"}


def test_drainage_alone_also_overrides_the_hint():
    c = IrrigationConstraint(flags=("drain", "avoid_stress"))
    assert c.flags == ("drain",)
    assert c.overridden == ("avoid_stress",)


def test_the_contradiction_cannot_be_built_by_any_route():
    # Not just via merge() - the invariant holds at construction.
    assert IrrigationConstraint(flags=("avoid_stress", "suspend")).flags == ("suspend",)


def test_avoid_stress_is_only_ever_a_hint(engine):
    c = engine.advise("pest__spider_mite", confidence=0.9).irrigation_constraint
    assert c.hints == ("avoid_stress",)
    assert not c.blocks_irrigation
    assert not c.method_flags and not c.timing_flags


# -- precedence rule 4: the probe wins on water status -----------------------
def test_only_the_two_water_classes_offer_water_status(engine, taxonomy):
    with_status = {
        c.id: engine.advise(c.id, confidence=0.9).irrigation_constraint
        for c in taxonomy.classes
        if engine.advise(c.id, confidence=0.9).irrigation_constraint.water_status_evidence
    }
    assert {
        cid: c.water_status_evidence for cid, c in with_status.items()
    } == {"abiotic__water_stress": "dry", "abiotic__waterlogging": "waterlogged"}


def test_water_stress_corroborates_without_commanding(engine):
    """The DEGRADED-state corroborator. It reports what the leaf shows and
    constrains the method; it does not order an irrigation."""
    c = engine.advise("abiotic__water_stress", confidence=0.9).irrigation_constraint
    assert c.water_status_evidence == "dry"
    assert c.flags == ("prefer_furrow",)
    assert not c.blocks_irrigation


def test_disagreeing_detections_report_no_water_status():
    c = merge(
        [
            IrrigationConstraint(water_status_evidence="dry"),
            IrrigationConstraint(water_status_evidence="waterlogged"),
        ]
    )
    assert c.water_status_evidence is None


# -- precedence rule 5: constraints expire -----------------------------------
def test_ttl_mirrors_the_recheck_interval(engine):
    a = engine.advise("abiotic__waterlogging", confidence=0.95)
    assert a.irrigation_constraint.ttl_hours == a.recheck_hours == 6


def test_merge_keeps_the_shortest_ttl():
    c = merge(
        [
            IrrigationConstraint(flags=("no_overhead",), ttl_hours=48),
            IrrigationConstraint(flags=("drain",), ttl_hours=6),
        ]
    )
    assert c.ttl_hours == 6


def test_every_constraint_expires(engine, taxonomy):
    for c in taxonomy.classes:
        ic = engine.advise(c.id, confidence=0.9).irrigation_constraint
        if not ic.is_empty:
            assert 0 < ic.ttl_hours <= MAX_TTL_HOURS, c.id


def test_a_slow_problem_does_not_get_a_slow_constraint(engine):
    """Nitrogen deficiency is rightly rechecked in a week; the irrigation
    constraint it raises must not survive that long."""
    a = engine.advise("deficiency__nitrogen", confidence=0.9)
    assert a.recheck_hours > MAX_TTL_HOURS
    assert a.irrigation_constraint.ttl_hours == MAX_TTL_HOURS


# -- wiring ------------------------------------------------------------------
def test_priority_tracks_urgency(engine):
    a = engine.advise("abiotic__waterlogging", confidence=0.95)
    assert a.irrigation_constraint.priority == URGENCY_ORDER.index(a.urgency)


def test_every_advisory_carries_a_constraint(engine, taxonomy):
    for c in taxonomy.classes:
        assert isinstance(
            engine.advise(c.id, confidence=0.9).irrigation_constraint,
            IrrigationConstraint,
        )
    # Including the reject path, which has nothing to say about water.
    assert engine.advise_uncertain().irrigation_constraint is NO_CONSTRAINT


def test_constraint_survives_json_round_trip(engine):
    d = json.loads(engine.advise("abiotic__waterlogging", confidence=0.95).to_json())
    ic = d["irrigation_constraint"]
    assert ic["blocks_irrigation"] is True
    assert ic["needs_drainage"] is True
    # The engine must be able to act on this without importing our code.
    assert set(ic) == {
        "flags",
        "blocks_irrigation",
        "needs_drainage",
        "reasons",
        "water_status_evidence",
        "ttl_hours",
        "priority",
        "overridden",
    }


def test_prose_and_flags_stay_in_step(engine, taxonomy):
    """Every class with irrigation prose beyond the category default must have
    had its flags reviewed - the two halves are edited together or not at all."""
    kb_classes = engine.kb["classes"]
    for cid, entry in kb_classes.items():
        if entry.get("irrigation_advice"):
            assert "irrigation_constraint" in entry, cid
            validate_flags(entry["irrigation_constraint"])
    for cat, entry in engine.kb["defaults"].items():
        assert "irrigation_constraint" in entry, cat
        validate_flags(entry["irrigation_constraint"])


# -- the voice rule (FasalSetu prompt I6) ------------------------------------
#: No millimetres, no percentages, no model terms in anything the farmer hears.
#: Scouting counts and economic thresholds are legitimate agronomy and live on
#: the detail path (``steps`` / ``notes``), which this deliberately excludes.
_BANNED = re.compile(
    r"\b(mm|millimet\w*|per cent|percent|kPa|ET0|ETc|Kc|RAW|depletion|"
    r"set ?point|confidence|probability|logit|softmax|classifier|embedding|"
    r"calibrat\w*)\b|%|\bp\s*=",
    re.I,
)


def test_voice_path_carries_no_units_or_model_terms(engine, taxonomy):
    offenders = []
    for c in list(taxonomy.classes) + [None]:
        a = engine.advise(c.id, confidence=0.9) if c else engine.advise_uncertain()
        surfaces = [a.headline, a.message, a.to_sms()]
        surfaces += a.voice_lines()
        surfaces += a.irrigation_constraint.phrases()
        offenders += [(a.class_id, s) for s in surfaces if _BANNED.search(s)]
    assert offenders == []


def test_voice_lines_are_at_most_three_facts(engine, taxonomy):
    for c in taxonomy.classes:
        lines = engine.advise(c.id, confidence=0.9).voice_lines()
        assert 0 < len(lines) <= 3, c.id
        assert len(lines) == len(set(lines)), c.id


def test_voice_leads_with_the_most_consequential_constraint(engine):
    """Only the first phrase survives into speech, so 'do not wet the leaves'
    must beat 'water in the morning'."""
    lines = engine.advise("tomato__late_blight", confidence=0.95).voice_lines()
    assert lines[1] == CONSTRAINT_PHRASES["no_overhead"]


# -- field level: what the irrigation engine actually calls -------------------
def _tracker_with(*specs):
    """A tracker carrying ``(class_id, n_days)`` of detections on field f1."""
    t = PestPressureTracker()
    for class_id, days in specs:
        for d in range(days):
            t.add(
                Detection(
                    timestamp=NOW - timedelta(days=d),
                    class_id=class_id,
                    confidence=0.93,
                    field_id="f1",
                )
            )
    return t


def test_field_constraint_merges_every_live_alert():
    t = _tracker_with(("tomato__late_blight", 4), ("pest__spider_mite", 2))
    c = t.irrigation_constraint("f1", now=NOW)
    assert set(c.flags) == {
        "no_overhead",
        "prefer_furrow",
        "morning_only",
        "avoid_stress",
    }
    assert not c.blocks_irrigation
    assert len(c.reasons) == 2


def test_waterlogging_on_the_field_overrides_the_mite_hint():
    """The integration case that motivated the whole contract: two problems on
    one field whose water advice contradicts."""
    t = _tracker_with(
        ("tomato__late_blight", 4),
        ("pest__spider_mite", 2),
        ("abiotic__waterlogging", 1),
    )
    c = t.irrigation_constraint("f1", now=NOW)
    assert c.blocks_irrigation and c.needs_drainage
    assert "avoid_stress" not in c.flags
    assert c.overridden == ("avoid_stress",)
    assert c.water_status_evidence == "waterlogged"
    # The most perishable observation sets the expiry for the whole thing.
    assert c.ttl_hours == 6
    # And speech leads with the one line that matters.
    assert c.phrases()[0] == CONSTRAINT_PHRASES["suspend"]


def test_a_quiet_field_constrains_nothing():
    assert PestPressureTracker().irrigation_constraint("f1", now=NOW) is NO_CONSTRAINT


def test_field_constraint_never_carries_a_volume_or_duration():
    """The load-bearing promise: the irrigation engine keeps sole ownership of
    how much water and for how long."""
    t = _tracker_with(("abiotic__water_stress", 3))
    d = t.irrigation_constraint("f1", now=NOW).to_dict()
    assert not {"litres", "volume", "duration", "runtime_s", "mm"} & set(d)
    # water_status_evidence is corroboration for the engine's DEGRADED state,
    # not an instruction to open the valve.
    assert d["water_status_evidence"] == "dry"
    assert d["blocks_irrigation"] is False


def test_alert_priority_is_repriced_against_the_alert_level():
    t = _tracker_with(("abiotic__waterlogging", 1))
    alert = t.evaluate("f1", now=NOW)[0]
    assert alert.irrigation_constraint.priority == ALERT_LEVELS.index(alert.level)


def test_alert_serialises_its_constraint():
    t = _tracker_with(("abiotic__waterlogging", 1))
    d = t.evaluate("f1", now=NOW)[0].to_dict()
    assert d["irrigation_constraint"]["blocks_irrigation"] is True
    assert json.dumps(d)  # must survive the wire to the app
