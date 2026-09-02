"""Turn a stream of single-image detections into a field-level early warning.

One photo showing three aphids is not an outbreak. Ten photos across a week
showing a rising count is. The model answers per image; this module holds the
time dimension, which is where "early warning" actually lives.

Two independent signals, deliberately kept separate:

**Pest pressure** - the trend in confirmed detections per scouting round,
smoothed with an EWMA and compared against the economic threshold (ETL) from
the taxonomy. The ETL is the number Indian extension services actually
publish, so an alert phrased against it is one an extension officer can check.

**Weather-driven infection risk** - most fungal and oomycete diseases need a
specific temperature and leaf-wetness window before infection can occur. Late
blight needs cool, humid, wet-leaf conditions; rust needs dew. These rules fire
*before* symptoms are visible, which is the only useful time to apply a
protectant. They come from the environmental sensors, not the camera.

The two are combined in :func:`combined_risk`: sensors saying "infection
conditions present" plus camera saying "first lesions detected" is a far
stronger signal than either alone, and warrants a different message.

Pure standard library - this runs on the gateway alongside the classifier.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .advisory import AdvisoryEngine, default_engine
from .taxonomy import Taxonomy, load_taxonomy

ALERT_LEVELS = ("none", "info", "watch", "warning", "critical")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Detection:
    """One accepted model output, tagged with where and when."""

    class_id: str
    confidence: float
    timestamp: datetime = field(default_factory=_now)
    field_id: str = "default"
    zone: str = "default"
    severity: str | None = None
    count: float | None = None      # scouting count, when the app collected one
    source: str = "camera"

    @property
    def day(self) -> str:
        return self.timestamp.date().isoformat()


@dataclass
class Alert:
    """A field-level warning, ready for the display / SMS / dashboard."""

    field_id: str
    zone: str
    class_id: str
    display_name: str
    level: str
    title: str
    message: str
    detections: int
    window_days: float
    trend: str                       # rising | steady | falling | new
    trend_pct: float | None = None
    ewma: float = 0.0
    etl_ratio: float | None = None   # observed count / economic threshold
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    recommended_action: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def level_rank(self) -> int:
        return ALERT_LEVELS.index(self.level)

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "zone": self.zone,
            "class_id": self.class_id,
            "display_name": self.display_name,
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "detections": self.detections,
            "window_days": round(self.window_days, 2),
            "trend": self.trend,
            "trend_pct": round(self.trend_pct, 1) if self.trend_pct is not None else None,
            "ewma": round(self.ewma, 3),
            "etl_ratio": round(self.etl_ratio, 2) if self.etl_ratio is not None else None,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "recommended_action": self.recommended_action,
            "evidence": list(self.evidence),
        }

    def to_sms(self, limit: int = 160) -> str:
        prefix = {"critical": "URGENT: ", "warning": "ALERT: ", "watch": "WATCH: "}.get(self.level, "")
        body = f"{prefix}{self.message}"
        if len(body) <= limit:
            return body
        cut = body[: limit - 3]
        return (cut[: cut.rindex(" ")] if " " in cut else cut).rstrip(" ,.;:") + "..."


class PestPressureTracker:
    """Rolling detection history per (field, zone, class), with trend and ETL.

    Memory is bounded by ``max_events`` because this runs for a whole season on
    a device with no disk to spare.
    """

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        advisory: AdvisoryEngine | None = None,
        window_days: float = 14.0,
        ewma_alpha: float = 0.4,
        min_confidence: float = 0.5,
        max_events: int = 5000,
    ):
        self.taxonomy = taxonomy or load_taxonomy()
        self.advisory = advisory or default_engine()
        self.window = timedelta(days=window_days)
        self.window_days = window_days
        self.alpha = ewma_alpha
        self.min_confidence = min_confidence
        self._events: deque[Detection] = deque(maxlen=max_events)

    # -- ingestion -------------------------------------------------------
    def add(self, detection: Detection) -> None:
        if detection.confidence < self.min_confidence:
            return  # an uncertain frame must not create a trend
        self._events.append(detection)

    def add_diagnosis(
        self,
        diagnosis,
        field_id: str = "default",
        zone: str = "default",
        timestamp: datetime | None = None,
        count: float | None = None,
    ) -> None:
        """Ingest a :class:`cropguard.edge.Diagnosis` directly."""
        if not getattr(diagnosis, "accepted", False):
            return
        self.add(
            Detection(
                class_id=diagnosis.class_id,
                confidence=diagnosis.confidence,
                timestamp=timestamp or _now(),
                field_id=field_id,
                zone=zone,
                severity=getattr(diagnosis, "severity", None),
                count=count,
            )
        )

    def prune(self, now: datetime | None = None) -> int:
        now = now or _now()
        cutoff = now - self.window * 2  # keep one extra window for trend context
        before = len(self._events)
        self._events = deque(
            (e for e in self._events if e.timestamp >= cutoff), maxlen=self._events.maxlen
        )
        return before - len(self._events)

    @property
    def events(self) -> list[Detection]:
        return list(self._events)

    # -- analysis --------------------------------------------------------
    def daily_counts(
        self, class_id: str, field_id: str = "default", zone: str | None = None,
        now: datetime | None = None,
    ) -> list[tuple[str, int]]:
        now = now or _now()
        start = now - self.window
        buckets: dict[str, int] = defaultdict(int)
        for e in self._events:
            if e.class_id != class_id or e.field_id != field_id:
                continue
            if zone is not None and e.zone != zone:
                continue
            if e.timestamp < start:
                continue
            buckets[e.day] += 1
        return sorted(buckets.items())

    def _ewma(self, series: Sequence[float]) -> float:
        value = 0.0
        for i, x in enumerate(series):
            value = x if i == 0 else self.alpha * x + (1 - self.alpha) * value
        return value

    def _trend(self, series: Sequence[float]) -> tuple[str, float | None]:
        """Compare the recent half of the window with the earlier half."""
        if len(series) < 2:
            return "new", None
        mid = max(1, len(series) // 2)
        early, late = series[:mid], series[mid:]
        a = statistics.fmean(early) if early else 0.0
        b = statistics.fmean(late) if late else 0.0
        if a == 0 and b == 0:
            return "steady", 0.0
        if a == 0:
            return "rising", 100.0
        change = (b - a) / a * 100.0
        if change > 25:
            return "rising", change
        if change < -25:
            return "falling", change
        return "steady", change

    def evaluate(
        self, field_id: str = "default", zone: str | None = None, now: datetime | None = None
    ) -> list[Alert]:
        """Produce one alert per active problem, most urgent first."""
        now = now or _now()
        start = now - self.window
        grouped: dict[tuple[str, str], list[Detection]] = defaultdict(list)
        for e in self._events:
            if e.field_id != field_id or e.timestamp < start:
                continue
            if zone is not None and e.zone != zone:
                continue
            grouped[(e.class_id, e.zone)].append(e)

        alerts = []
        for (class_id, ezone), events in grouped.items():
            alert = self._build_alert(class_id, ezone, events, field_id, now)
            if alert is not None:
                alerts.append(alert)
        alerts.sort(key=lambda a: (-a.level_rank, -a.detections))
        return alerts

    def _build_alert(
        self, class_id: str, zone: str, events: list[Detection], field_id: str, now: datetime
    ) -> Alert | None:
        crop_class = self.taxonomy.get(class_id)
        if crop_class is None or not crop_class.is_actionable:
            return None

        events.sort(key=lambda e: e.timestamp)
        by_day: dict[str, int] = defaultdict(int)
        for e in events:
            by_day[e.day] += 1
        series = [c for _, c in sorted(by_day.items())]
        ewma = self._ewma([float(x) for x in series])
        trend, trend_pct = self._trend([float(x) for x in series])

        # Start from the advisory's own urgency for this class, then move it
        # on the evidence: a rising trend and an ETL crossing both escalate.
        base = self.advisory.advise(
            class_id, confidence=statistics.fmean(e.confidence for e in events)
        )
        level = base.urgency if base.urgency in ALERT_LEVELS else "watch"
        evidence = [f"{len(events)} detection(s) over {len(series)} day(s)"]

        if trend == "rising":
            level = _bump_level(level, 1)
            evidence.append(
                f"detections rising {trend_pct:.0f}% between the first and second "
                f"half of the {self.window_days:.0f}-day window"
            )
        elif trend == "falling":
            evidence.append(f"detections falling {abs(trend_pct or 0):.0f}% - control appears to be working")

        etl_ratio = None
        counts = [e.count for e in events if e.count is not None]
        if crop_class.etl and counts:
            threshold = float(crop_class.etl.get("threshold", 0)) or None
            if threshold:
                observed = statistics.fmean(counts)
                etl_ratio = observed / threshold
                unit = str(crop_class.etl.get("unit", "")).replace("_", " ")
                if etl_ratio >= 1.0:
                    level = _bump_level(level, 1)
                    evidence.append(
                        f"scouting count {observed:.1f} {unit} is at or above the "
                        f"economic threshold of {threshold:g}"
                    )
                else:
                    # The ETL is exactly the "do not spray below this" line, and
                    # the whole point of this system is targeted intervention
                    # instead of calendar spraying. A rising trend below the
                    # threshold is worth watching closely; it is not an
                    # emergency, and calling it one trains farmers to ignore
                    # the level.
                    level = min(level, "warning", key=lambda l: ALERT_LEVELS.index(l))
                    evidence.append(
                        f"scouting count {observed:.1f} {unit} is {etl_ratio:.0%} of the "
                        f"economic threshold of {threshold:g} - below the spray trigger, "
                        f"so the alert is capped at 'warning'"
                    )

        if any(e.severity == "severe" for e in events):
            level = _bump_level(level, 1)
            evidence.append("at least one frame graded severe")

        if len(events) == 1 and trend == "new":
            level = _bump_level(level, -1)
            evidence.append("single detection only - confirm before acting")

        message = self._message(crop_class, level, trend, len(events), etl_ratio)
        return Alert(
            field_id=field_id,
            zone=zone,
            class_id=class_id,
            display_name=crop_class.display_name,
            level=level,
            title=f"{crop_class.display_name} - {level}",
            message=message,
            detections=len(events),
            window_days=self.window_days,
            trend=trend,
            trend_pct=trend_pct,
            ewma=ewma,
            etl_ratio=etl_ratio,
            first_seen=events[0].timestamp,
            last_seen=events[-1].timestamp,
            recommended_action=base.action,
            evidence=evidence,
        )

    def _message(
        self, crop_class, level: str, trend: str, n: int, etl_ratio: float | None
    ) -> str:
        name = crop_class.display_name
        if crop_class.category == "pest":
            if etl_ratio is not None and etl_ratio >= 1.0:
                return (
                    f"{name} has crossed the economic threshold ({etl_ratio:.0%} of it). "
                    "Targeted control is now justified - treat the affected patch first."
                )
            if trend == "rising":
                return (
                    f"{name} activity is increasing ({n} detections). Scout and count "
                    "against the threshold before spraying."
                )
            return f"{name} detected ({n} detection(s)). Keep scouting; below the spray trigger so far."
        if trend == "rising" and level in ("warning", "critical"):
            return (
                f"{name} is spreading ({n} detections and rising). Act now - "
                "remove affected material and protect the healthy crop."
            )
        return f"{name} detected ({n} detection(s)). Confirm and follow the advisory."


def _bump_level(level: str, steps: int) -> str:
    i = ALERT_LEVELS.index(level) + steps if level in ALERT_LEVELS else 2
    return ALERT_LEVELS[max(0, min(len(ALERT_LEVELS) - 1, i))]


# ---------------------------------------------------------------------------
# weather-driven infection risk
# ---------------------------------------------------------------------------
@dataclass
class WeatherReading:
    """One environmental sample from the field sensor node."""

    timestamp: datetime
    temp_c: float
    humidity_pct: float
    leaf_wetness_hours: float = 0.0
    rainfall_mm: float = 0.0


@dataclass
class InfectionRule:
    """Conditions under which a pathogen can infect, from published models."""

    class_id: str
    name: str
    temp_range: tuple[float, float]
    min_humidity: float
    min_wet_hours: float
    consecutive_hours: int = 4
    note: str = ""


#: Simplified infection windows. These are decision-support approximations of
#: published models (Smith periods for late blight, standard rust/mildew dew
#: requirements) - useful for "spray a protectant before this window", not a
#: substitute for a validated local disease model.
INFECTION_RULES: tuple[InfectionRule, ...] = (
    InfectionRule("tomato__late_blight", "Late blight (Smith-period style)", (10.0, 24.0), 88.0, 10.0, 4,
                  "Two consecutive days with min temp >=10 C and >=11 h at RH >=90% is the classic Smith period."),
    InfectionRule("potato__late_blight", "Late blight (Smith-period style)", (10.0, 24.0), 88.0, 10.0, 4,
                  "Same window as tomato; blight moves between the two crops."),
    InfectionRule("grape__downy_mildew", "Downy mildew", (11.0, 25.0), 85.0, 4.0, 3,
                  "Needs free water on the leaf; the '10-10-24' rule of thumb is 10 C, 10 mm rain, 24 h."),
    InfectionRule("rice__blast", "Rice blast", (20.0, 28.0), 90.0, 8.0, 4,
                  "Long dew periods with cool nights after nitrogen top-dressing."),
    InfectionRule("wheat__stripe_rust", "Stripe rust", (8.0, 15.0), 85.0, 3.0, 3,
                  "Cool humid nights with dew; explosive in north Indian winters."),
    InfectionRule("wheat__leaf_rust", "Leaf rust", (15.0, 22.0), 85.0, 3.0, 3, "Dew period required for germination."),
    InfectionRule("maize__northern_leaf_blight", "Northern leaf blight", (18.0, 27.0), 90.0, 6.0, 4, ""),
    InfectionRule("chilli__anthracnose", "Anthracnose", (20.0, 30.0), 90.0, 8.0, 4, "Warm and wet, especially at fruiting."),
    InfectionRule("tomato__early_blight", "Early blight", (24.0, 29.0), 85.0, 5.0, 3, "Alternating wet and dry favours it."),
)


def infection_risk(
    readings: Sequence[WeatherReading],
    rules: Sequence[InfectionRule] = INFECTION_RULES,
    taxonomy: Taxonomy | None = None,
    field_id: str = "default",
    zone: str = "default",
) -> list[Alert]:
    """Flag pathogens whose infection window the weather has just satisfied.

    This is the part of early warning that fires *before* the camera sees
    anything - which is the only time a protectant spray is worth its cost.
    """
    if not readings:
        return []
    tax = taxonomy or load_taxonomy()
    ordered = sorted(readings, key=lambda r: r.timestamp)
    alerts: list[Alert] = []

    for rule in rules:
        crop_class = tax.get(rule.class_id)
        if crop_class is None:
            continue
        lo, hi = rule.temp_range
        favourable = [
            lo <= r.temp_c <= hi and r.humidity_pct >= rule.min_humidity for r in ordered
        ]
        run = best_run = 0
        for ok in favourable:
            run = run + 1 if ok else 0
            best_run = max(best_run, run)
        wet_hours = sum(r.leaf_wetness_hours for r in ordered)

        if best_run < rule.consecutive_hours:
            continue
        met_wetness = wet_hours >= rule.min_wet_hours
        level = "warning" if met_wetness else "watch"
        if crop_class.spread_rank >= 4 and met_wetness:
            level = "critical"

        evidence = [
            f"{best_run} consecutive readings in {lo:g}-{hi:g} C at RH >= {rule.min_humidity:g}%",
            f"leaf wetness {wet_hours:.1f} h against a {rule.min_wet_hours:g} h requirement",
        ]
        if rule.note:
            evidence.append(rule.note)

        message = (
            f"Weather favours {crop_class.name.lower()} on {crop_class.crop}. "
            + (
                "Infection conditions have been met - apply a protectant before the next wet spell."
                if met_wetness
                else "Conditions are borderline - scout closely and be ready to spray."
            )
        )
        alerts.append(
            Alert(
                field_id=field_id,
                zone=zone,
                class_id=rule.class_id,
                display_name=crop_class.display_name,
                level=level,
                title=f"{crop_class.display_name} - infection risk",
                message=message,
                detections=0,
                window_days=(ordered[-1].timestamp - ordered[0].timestamp).total_seconds() / 86400.0,
                trend="new",
                first_seen=ordered[0].timestamp,
                last_seen=ordered[-1].timestamp,
                recommended_action="preventive_protectant",
                evidence=evidence,
            )
        )
    alerts.sort(key=lambda a: -a.level_rank)
    return alerts


def combined_risk(
    camera_alerts: Sequence[Alert], weather_alerts: Sequence[Alert]
) -> list[Alert]:
    """Merge the two signals, escalating where they agree.

    Sensors saying "infection conditions present" and the camera saying "first
    lesions found" is the single strongest signal this system can produce, and
    it should not be delivered as two separate half-warnings.
    """
    import copy

    by_class: dict[str, Alert] = {}
    # Copy on the way in: callers routinely keep the camera alerts for the
    # dashboard while passing the same list here, and mutating them in place
    # would silently rewrite what the dashboard shows.
    for alert in list(camera_alerts) + list(weather_alerts):
        existing = by_class.get(alert.class_id)
        if existing is None:
            by_class[alert.class_id] = copy.deepcopy(alert)
            continue

        camera, weather = (
            (existing, alert) if existing.detections > 0 else (alert, existing)
        )
        merged = copy.deepcopy(camera)
        merged.level = _bump_level(
            max(camera.level, weather.level, key=lambda l: ALERT_LEVELS.index(l)), 1
        )
        merged.evidence = [*camera.evidence, *weather.evidence]
        merged.message = (
            f"{camera.display_name}: symptoms detected AND the weather is in the "
            "infection window. This is the highest-confidence warning the system "
            "produces - act today."
        )
        merged.title = f"{camera.display_name} - confirmed by weather and camera"
        by_class[alert.class_id] = merged

    out = list(by_class.values())
    out.sort(key=lambda a: (-a.level_rank, -a.detections))
    return out


__all__ = [
    "Detection",
    "Alert",
    "ALERT_LEVELS",
    "PestPressureTracker",
    "WeatherReading",
    "InfectionRule",
    "INFECTION_RULES",
    "infection_risk",
    "combined_risk",
]
