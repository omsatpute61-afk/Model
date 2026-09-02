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

#: What seeing a given life stage means for the decision to act. This is the
#: whole reason the model carries a life-stage head: the species tells you what
#: the problem is, the stage tells you whether today is the day to spend money
#: on it. Spraying a field because moths appeared on a trap is the classic way
#: to waste an application and select for resistance at the same time.
LIFE_STAGE_GUIDANCE: dict[str, dict[str, object]] = {
    "egg": {
        "urgency_shift": 0,
        "note": (
            "Egg masses present - the damage has not started yet. This is the "
            "cheapest moment to act: destroy the egg masses by hand, or time a "
            "treatment for just after hatching rather than now."
        ),
        "action": "scout_and_time_treatment",
    },
    "larva": {
        "urgency_shift": 1,
        "note": (
            "Larvae are the feeding stage - this is the damage happening now. "
            "Treat while they are small and still exposed; older larvae bore in "
            "and no spray reaches them."
        ),
        "action": "treat_affected_plants",
    },
    "nymph": {
        "urgency_shift": 1,
        "note": (
            "Nymphs are feeding and cannot fly away, which makes them the easiest "
            "stage to control. Target the leaf undersides where they sit."
        ),
        "action": "treat_affected_plants",
    },
    "adult": {
        "urgency_shift": -1,
        "note": (
            "Adults indicate a flight, not established damage. Count them against "
            "the trap threshold before spending on a spray - treating on moth "
            "catch alone wastes the application and builds resistance. Expect "
            "larvae in about a week and scout for them then."
        ),
        "action": "monitor_and_count",
    },
}

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
    life_stage: str | None = None
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
            "life_stage": self.life_stage,
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
        life_stage: str | None = None,
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
        life_stage:
            Optional pest life stage. Changes whether the advice is "monitor"
            or "treat now" - see :data:`LIFE_STAGE_GUIDANCE`.
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

        action_override = None
        if crop_class.category != "pest":
            # A life stage on a leaf-spot image is meaningless. Drop it rather
            # than record a value that had no bearing on the advice.
            life_stage = None
        elif life_stage:
            guidance = LIFE_STAGE_GUIDANCE.get(life_stage)
            if guidance:
                urgency = _bump(urgency, int(guidance["urgency_shift"]))
                notes.append(str(guidance["note"]))
                action_override = str(guidance["action"])

        if crop_class.vector:
            vector = self.taxonomy.get(crop_class.vector)
            if vector is not None:
                notes.append(
                    f"This is spread by {vector.name.lower()}. Controlling the "
                    "vector matters more than treating the symptom."
                )

        if not crop_class.etl and crop_class.etl_note:
            notes.append(crop_class.etl_note)
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
            action=action_override or base.get("action", "scout_and_confirm"),
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
            life_stage=life_stage,
            needs_confirmation=needs_confirmation,
            notes=notes,
            disclaimer=self.disclaimer,
        )

    def advise_uncertain(self, top_candidates: list[tuple[str, float]] | None = None) -> Advisory:
        """Advisory for a rejected / below-threshold prediction."""
        return self._unknown_advisory(None, top_candidates)

    # -- internals -------------------------------------------------------
    def _compose_message(self, crop_class, base: dict) -> str:
        """Fallback farmer message built from the taxonomy entry itself.

        Used for classes with no hand-written advisory. It leans on
        ``display_name`` rather than the raw ``crop`` field, because many pests
        are recorded against crop ``"any"`` and "detected on any" is not a
        sentence anybody should receive by SMS.
        """
        symptom = crop_class.symptoms.split(";")[0].strip().rstrip(".")
        if crop_class.category == "healthy":
            crop = "The crop" if crop_class.crop == "any" else crop_class.crop.title()
            return f"{crop} looks healthy. Continue normal monitoring."
        if crop_class.category == "background":
            return "No crop leaf detected in the photo. Fill the frame with one leaf and retake."

        # The causal agent helps for a disease or pest ("Alternaria solani");
        # for a deficiency it just restates the class name.
        agent = (
            f" ({crop_class.agent})"
            if crop_class.agent and crop_class.category in ("disease", "pest")
            else ""
        )
        return f"{crop_class.display_name} detected{agent}. Signs to confirm: {symptom}."

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
