"""Early warning is where single detections become a field-level decision."""

from datetime import datetime, timedelta, timezone

from cropguard.early_warning import (
    ALERT_LEVELS,
    Detection,
    PestPressureTracker,
    WeatherReading,
    combined_risk,
    infection_risk,
)

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _tracker(**kw):
    return PestPressureTracker(**kw)


def _series(tracker, class_id, schedule, count=None, confidence=0.9, severity=None):
    for days_ago, n in schedule:
        for i in range(n):
            tracker.add(
                Detection(
                    class_id, confidence,
                    NOW - timedelta(days=days_ago, hours=i),
                    count=count, severity=severity,
                )
            )


def test_low_confidence_detections_never_create_a_trend():
    t = _tracker(min_confidence=0.6)
    _series(t, "pest__aphid", [(1, 5)], confidence=0.3)
    assert t.evaluate(now=NOW) == []


def test_rising_detections_escalate_the_alert():
    rising = _tracker()
    _series(rising, "pest__aphid", [(9, 1), (7, 1), (3, 4), (1, 6)])
    falling = _tracker()
    _series(falling, "pest__aphid", [(9, 6), (7, 4), (3, 1), (1, 1)])

    a = rising.evaluate(now=NOW)[0]
    b = falling.evaluate(now=NOW)[0]
    assert a.trend == "rising"
    assert b.trend == "falling"
    assert ALERT_LEVELS.index(a.level) > ALERT_LEVELS.index(b.level)


def test_a_single_detection_is_softened():
    t = _tracker()
    _series(t, "pest__aphid", [(1, 1)])
    alert = t.evaluate(now=NOW)[0]
    assert alert.trend == "new"
    assert any("single detection" in e for e in alert.evidence)


def test_crossing_the_economic_threshold_is_what_justifies_spraying():
    below = _tracker()
    _series(below, "pest__brown_planthopper", [(4, 3), (2, 4), (1, 5)], count=2.0)   # ETL 10
    above = _tracker()
    _series(above, "pest__brown_planthopper", [(4, 3), (2, 4), (1, 5)], count=25.0)

    lo = below.evaluate(now=NOW)[0]
    hi = above.evaluate(now=NOW)[0]
    assert lo.etl_ratio < 1.0 and hi.etl_ratio > 1.0
    assert ALERT_LEVELS.index(hi.level) > ALERT_LEVELS.index(lo.level)
    assert "threshold" in hi.message.lower()


def test_below_threshold_never_reaches_critical():
    """Calling a below-ETL finding an emergency trains farmers to ignore alerts."""
    t = _tracker()
    _series(t, "pest__fall_armyworm", [(9, 1), (6, 2), (3, 4), (1, 8)], count=0.5)  # ETL 5
    alert = t.evaluate(now=NOW)[0]
    assert alert.trend == "rising"
    assert alert.level != "critical"


def test_healthy_and_background_never_raise_an_alert():
    t = _tracker()
    _series(t, "tomato__healthy", [(2, 5), (1, 5)])
    _series(t, "background", [(2, 5)])
    assert t.evaluate(now=NOW) == []


def test_alerts_are_scoped_by_field_and_zone():
    t = _tracker()
    t.add(Detection("pest__aphid", 0.9, NOW - timedelta(days=1), field_id="f1", zone="north"))
    t.add(Detection("pest__aphid", 0.9, NOW - timedelta(days=1), field_id="f2", zone="south"))
    assert len(t.evaluate(field_id="f1", now=NOW)) == 1
    assert t.evaluate(field_id="f1", zone="south", now=NOW) == []


def test_events_outside_the_window_are_ignored():
    t = _tracker(window_days=7)
    _series(t, "pest__aphid", [(30, 10)])
    assert t.evaluate(now=NOW) == []


def test_prune_bounds_memory():
    t = _tracker(window_days=7)
    _series(t, "pest__aphid", [(40, 5), (1, 5)])
    removed = t.prune(now=NOW)
    assert removed == 5
    assert len(t.events) == 5


def test_alert_sms_fits_one_segment():
    t = _tracker()
    _series(t, "pest__brown_planthopper", [(3, 5), (1, 9)], count=40.0)
    sms = t.evaluate(now=NOW)[0].to_sms()
    assert len(sms) <= 160


def test_weather_alone_can_warn_before_symptoms_appear():
    readings = [
        WeatherReading(NOW - timedelta(hours=h), 16.0, 92.0, leaf_wetness_hours=1.0)
        for h in range(14, 0, -1)
    ]
    alerts = infection_risk(readings)
    classes = {a.class_id for a in alerts}
    assert "tomato__late_blight" in classes
    late = next(a for a in alerts if a.class_id == "tomato__late_blight")
    assert late.detections == 0
    assert late.level in ("warning", "critical")


def test_dry_warm_weather_raises_no_infection_alert():
    readings = [
        WeatherReading(NOW - timedelta(hours=h), 35.0, 30.0, leaf_wetness_hours=0.0)
        for h in range(14, 0, -1)
    ]
    assert infection_risk(readings) == []


def test_camera_plus_weather_outranks_either_alone():
    t = _tracker()
    _series(t, "tomato__late_blight", [(2, 2), (1, 3)])
    camera = t.evaluate(now=NOW)
    weather = infection_risk(
        [WeatherReading(NOW - timedelta(hours=h), 16.0, 92.0, leaf_wetness_hours=1.0)
         for h in range(14, 0, -1)]
    )
    camera_evidence_before = list(camera[0].evidence)
    camera_level_before = camera[0].level

    merged = combined_risk(camera, weather)
    blight = next(a for a in merged if a.class_id == "tomato__late_blight")
    assert blight.level == "critical"
    assert "symptoms detected AND" in blight.message
    assert len(blight.evidence) > len(camera_evidence_before)
    # merging must not rewrite the caller's own alert objects
    assert camera[0].evidence == camera_evidence_before
    assert camera[0].level == camera_level_before


def test_combined_risk_keeps_weather_only_alerts():
    weather = infection_risk(
        [WeatherReading(NOW - timedelta(hours=h), 16.0, 92.0, leaf_wetness_hours=1.0)
         for h in range(14, 0, -1)]
    )
    merged = combined_risk([], weather)
    assert len(merged) == len(weather)


def test_diagnosis_objects_can_be_ingested_directly():
    from cropguard.edge.runtime import Diagnosis

    t = _tracker()
    accepted = Diagnosis("pest__aphid", "Aphid", "pest", 0.9, accepted=True)
    rejected = Diagnosis("unknown", "Not identified", "background", 0.2, accepted=False)
    t.add_diagnosis(accepted, timestamp=NOW - timedelta(hours=1))
    t.add_diagnosis(rejected, timestamp=NOW - timedelta(hours=1))
    assert len(t.events) == 1


def test_alert_serialises():
    t = _tracker()
    _series(t, "pest__aphid", [(2, 3), (1, 4)])
    d = t.evaluate(now=NOW)[0].to_dict()
    assert d["class_id"] == "pest__aphid"
    assert d["level"] in ALERT_LEVELS
    assert isinstance(d["evidence"], list)
