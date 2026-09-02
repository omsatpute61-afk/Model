"""Turn a model prediction into something a farmer can act on.

The classifier's job ends at "``tomato__late_blight``, p=0.91". That string is
useless on a field display. This module carries the last mile: urgency, the
concrete next step, what NOT to do, and a 160-character version for SMS on a
feature phone.

Design notes
------------
* Advice is data (``resources/advisory.json``), not code, so an agronomist can
  review and correct it without a release.
* Every class falls back to a per-category default, so adding a class to the
  taxonomy can never produce an empty advisory.
* Confidence is part of the advice. A 0.55 prediction of late blight must not
  read the same as a 0.97 one - the first asks for a second photo, the second
  asks for a spray.
* No dependencies beyond the standard library: this runs on the edge box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .taxonomy import RESOURCE_DIR, Taxonomy, load_taxonomy

ADVISORY_PATH = RESOURCE_DIR / "advisory.json"

URGENCY_ORDER: tuple[str, ...] = ("none", "info", "watch", "warning", "critical")

#: Below this the model is telling us it does not really know. We never issue a
#: spray recommendation on such a prediction - we ask for a better photo.
DEFAULT_MIN_CONFIDENCE = 0.55
#: Above this we are willing to let a fast-spreading disease escalate to
#: ``critical`` on a single frame.
DEFAULT_HIGH_CONFIDENCE = 0.85


def _bump(urgency: str, steps: int) -> str:
    i = URGENCY_ORDER.index(urgency) + steps
    return URGENCY_ORDER[max(0, min(len(URGENCY_ORDER) - 1, i))]


@dataclass
class Advisory:
    """A farmer-facing recommendation derived from one or more detections."""

    class_id: str
    display_name: str
    category: str
    urgency: str
    action: str
    headline: str
    message: str
    steps: list[str] = field(default_factory=list)
    ipm: list[str] = field(default_factory=list)
    chemical_guidance: str = ""
    irrigation_advice: str = ""
    escalate_if: str = ""
    recheck_hours: int = 48
    confidence: float | None = None
    severity: str | None = None
    needs_confirmation: bool = False
    notes: list[str] = field(default_factory=list)
    disclaimer: str = ""

    @property
    def urgency_rank(self) -> int:
        return URGENCY_ORDER.index(self.urgency)

    def to_sms(self, limit: int = 160) -> str:
        """Single-segment SMS for farmers without a smartphone.

        Truncates on a word boundary so the message never ends mid-word.
        """
        prefix = {"critical": "URGENT: ", "warning": "ALERT: "}.get(self.urgency, "")
        body = f"{prefix}{self.message}".strip()
        if len(body) <= limit:
            return body
        cut = body[: limit - 3]
        if " " in cut:
            cut = cut[: cut.rindex(" ")]
        return cut.rstrip(" ,.;:") + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "display_name": self.display_name,
            "category": self.category,
            "urgency": self.urgency,
            "action": self.action,
            "headline": self.headline,
            "message": self.message,
            "steps": list(self.steps),
            "ipm": list(self.ipm),
            "chemical_guidance": self.chemical_guidance,
            "irrigation_advice": self.irrigation_advice,
            "escalate_if": self.escalate_if,
            "recheck_hours": self.recheck_hours,
            "confidence": self.confidence,
            "severity": self.severity,
            "needs_confirmation": self.needs_confirmation,
            "notes": list(self.notes),
            "disclaimer": self.disclaimer,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


class AdvisoryEngine:
    """Look up and adapt advice for a predicted class."""

    def __init__(
        self,
        knowledge: dict | None = None,
        taxonomy: Taxonomy | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
    ):
        self.kb = knowledge if knowledge is not None else _load_knowledge()
        self.taxonomy = taxonomy or load_taxonomy()
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self._defaults = self.kb.get("defaults", {})
        self._classes = self.kb.get("classes", {})
        self.disclaimer = self.kb.get("disclaimer", "")

    # -- public API ------------------------------------------------------
    def message(self, key: str, lang: str = "en") -> str:
        """One of the short canned strings (``irrigate_now``, ``flood_risk``...)."""
        entry = self.kb.get("generic_messages", {}).get(key, {})
        return entry.get(lang) or entry.get("en") or key.replace("_", " ")

    def advise(
        self,
        class_id: str,
        confidence: float | None = None,
        severity: str | None = None,
        *,
        affected_fraction: float | None = None,
    ) -> Advisory:
        """Build the advisory for one prediction.

        Parameters
        ----------
        class_id:
            Predicted taxonomy id.
        confidence:
            Calibrated probability from the model. Drives whether we recommend
            acting or re-photographing.
        severity:
            Optional severity head output (``none``/``low``/``moderate``/``severe``).
        affected_fraction:
            Share of scouted plants showing the symptom, when the app has
            collected it. Turns a single-leaf finding into a field-level one.
        """
        crop_class = self.taxonomy.get(class_id)
        if crop_class is None:
            return self._unknown_advisory(confidence)

        base = dict(self._defaults.get(crop_class.category, {}))
        base.update(self._classes.get(class_id, {}))

        notes: list[str] = []
        urgency = base.get("urgency", "info")

        # A fast-spreading problem seen clearly deserves more urgency than the
        # category default; a marginal one deserves less.
        if confidence is not None:
            if confidence < self.min_confidence:
                urgency = _bump(urgency, -1)
                notes.append(
                    "Low confidence - take a second, closer photo in daylight "
                    "before spending money on any input."
                )
            elif confidence >= self.high_confidence and crop_class.spread_rank >= 3:
                urgency = _bump(urgency, 1)

        if severity in {"moderate", "severe"} and crop_class.is_actionable:
            urgency = _bump(urgency, 1 if severity == "severe" else 0)
            if severity == "severe":
                notes.append("Severe symptoms - treat this as a field-level problem, not a single plant.")

        if affected_fraction is not None and crop_class.is_actionable:
            if affected_fraction >= 0.10:
                urgency = _bump(urgency, 1)
                notes.append(
                    f"{affected_fraction:.0%} of scouted plants affected - above the "
                    "usual 10% action level for field-wide intervention."
                )
            elif affected_fraction <= 0.02:
                # A scouting round that found the problem on one plant in a
                # hundred is a watch item, not an emergency. Leaving it at the
                # same urgency as a field-wide infestation is how farmers learn
                # to ignore the alerts.
                urgency = _bump(urgency, -1)
                notes.append(
                    f"Only {affected_fraction:.0%} of scouted plants affected - "
                    "spot-treat that patch rather than the whole field."
                )

        if crop_class.vector:
            vector = self.taxonomy.get(crop_class.vector)
            if vector is not None:
                notes.append(
                    f"This is spread by {vector.name.lower()}. Controlling the "
                    "vector matters more than treating the symptom."
                )

        if crop_class.etl:
            etl = crop_class.etl
            notes.append(
                "Economic threshold: {threshold:g} {unit} ({note}).".format(
                    threshold=etl.get("threshold", 0),
                    unit=str(etl.get("unit", "")).replace("_", " "),
                    note=etl.get("note", "confirm locally"),
                )
            )

        needs_confirmation = bool(
            confidence is not None and confidence < self.min_confidence
        ) or crop_class.spread_rank >= 4

        message = base.get("farmer_message") or self._compose_message(crop_class, base)

        return Advisory(
            class_id=class_id,
            display_name=crop_class.display_name,
            category=crop_class.category,
            urgency=urgency,
            action=base.get("action", "scout_and_confirm"),
            headline=base.get("headline", crop_class.display_name),
            message=message,
            steps=list(base.get("steps", [])),
            ipm=list(base.get("ipm", [])),
            chemical_guidance=base.get("chemical_guidance", ""),
            irrigation_advice=base.get("irrigation_advice", ""),
            escalate_if=base.get("escalate_if", ""),
            recheck_hours=int(base.get("recheck_hours", 48)),
            confidence=confidence,
            severity=severity,
            needs_confirmation=needs_confirmation,
            notes=notes,
            disclaimer=self.disclaimer,
        )

    def advise_uncertain(self, top_candidates: list[tuple[str, float]] | None = None) -> Advisory:
        """Advisory for a rejected / below-threshold prediction."""
        return self._unknown_advisory(None, top_candidates)

    # -- internals -------------------------------------------------------
    def _compose_message(self, crop_class, base: dict) -> str:
        """Fallback farmer message built from the taxonomy entry itself."""
        head = base.get("headline", crop_class.display_name)
        symptom = crop_class.symptoms.split(";")[0].strip().rstrip(".")
        if crop_class.category == "healthy":
            return f"{crop_class.crop.title()} looks healthy. Continue normal monitoring."
        if crop_class.category == "background":
            return "No crop leaf detected in the photo. Fill the frame with one leaf and retake."
        agent = f" ({crop_class.agent})" if crop_class.agent else ""
        return f"{head} on {crop_class.crop}{agent}. Look for: {symptom}."

    def _unknown_advisory(
        self, confidence: float | None, top: list[tuple[str, float]] | None = None
    ) -> Advisory:
        notes = []
        if top:
            names = ", ".join(
                f"{(self.taxonomy.get(c).display_name if self.taxonomy.get(c) else c)}"
                f" ({p:.0%})"
                for c, p in top[:3]
            )
            notes.append(f"Closest matches: {names}.")
        return Advisory(
            class_id="unknown",
            display_name="Not identified",
            category="background",
            urgency="info",
            action="retake_or_escalate",
            headline="Could not identify the problem",
            message=(
                "Could not identify this reliably. Take another photo of a single "
                "affected leaf in daylight, filling the frame. If symptoms are "
                "spreading, contact your KVK / agriculture officer."
            ),
            steps=[
                "Hold the camera 20-30 cm away and fill the frame with one leaf.",
                "Shoot in shade or diffuse light - avoid harsh shadow and glare.",
                "Photograph the underside too; many pests and mildews hide there.",
                "Send the photo to your KVK / agriculture officer if it spreads.",
            ],
            confidence=confidence,
            needs_confirmation=True,
            notes=notes,
            disclaimer=self.disclaimer,
        )


@lru_cache(maxsize=2)
def _load_knowledge(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else ADVISORY_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=2)
def default_engine() -> AdvisoryEngine:
    return AdvisoryEngine()


__all__ = [
    "Advisory",
    "AdvisoryEngine",
    "URGENCY_ORDER",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_HIGH_CONFIDENCE",
    "default_engine",
]
