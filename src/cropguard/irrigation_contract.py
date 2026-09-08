"""The boundary between this model and the irrigation subsystem.

FasalSetu has two things that can talk about water: a FAO-56 water-balance
engine driven by a soil probe, and this advisory layer driven by a photo of a
leaf. On a day when the water balance says *water today, 2 h 40 min* and the
model says *suspend overhead irrigation*, both are right and the farmer gets an
incoherent answer. Somebody has to own the split.

The split
---------
**The irrigation engine owns whether and how much. This model may only
constrain how and when.**

So this module does not emit litres, millimetres, hours or set points. It emits
a small vocabulary of flags (:data:`CONSTRAINT_KINDS`) that the irrigation state
machine consumes as an *input* alongside soil, weather and power. The engine
still computes the number; the constraint decides whether that number is
delivered overhead or in a furrow, at dawn or at dusk, in one run or three.

One flag is the exception, and it is the reason this is structured data rather
than a sentence: ``suspend`` stops irrigation outright. Watering a waterlogged
field is wrong no matter what the balance says, and a paragraph of English
prose cannot be safely acted on by a state machine.

Precedence
----------
1. Volume and duration belong to the irrigation engine. Nothing here carries a
   water quantity; :attr:`IrrigationConstraint.ttl_hours` is the only number
   that crosses, and it describes how long the observation stays valid.
2. ``suspend`` is the only flag that may stop a run the engine decided on, and
   only waterlogging raises it. ``drain`` drives the drainage side and says
   nothing on its own about the next irrigation. Every other flag may change
   only the method or the timing of a run.
3. ``avoid_stress`` is a hint and never causes irrigation on its own. Where it
   meets ``suspend`` or ``drain`` those win - a field with water standing on it
   is not drying out - and the dropped hint is recorded in
   :attr:`IrrigationConstraint.overridden`.
4. On water status the soil probe wins. :attr:`water_status_evidence` is
   corroboration for the engine's ``DEGRADED`` / ``UNCALIBRATED`` states, where
   there is no live probe to trust - never a substitute for one that is
   answering.
5. Constraints expire. A photo taken last week must not hold the valve shut
   today, so every constraint carries ``ttl_hours`` from the advisory's own
   recheck interval, clamped to :data:`MAX_TTL_HOURS`.

Nothing in this module imports the irrigation subsystem, and the irrigation
subsystem does not import this model. The contract is the dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: What each flag means, and what the irrigation engine is allowed to do with
#: it. ``veto`` may block a run; ``drainage`` asks for water to be taken *off*
#: the field, which is a different actuator from the valve and on its own says
#: nothing about the next irrigation; ``method`` and ``timing`` may only reshape
#: a run the engine already decided on; ``hint`` may not change the engine's
#: arithmetic at all.
#:
#: ``suspend`` is deliberately the only veto, and only one class carries it
#: (:data:`waterlogging <cropguard.advisory>`). A veto that fires on every
#: leaf-spot photo is one the irrigation team would rightly ignore.
CONSTRAINT_KINDS: dict[str, str] = {
    "suspend": "veto",
    "drain": "drainage",
    "no_overhead": "method",
    "prefer_furrow": "method",
    "morning_only": "timing",
    "no_evening": "timing",
    "short_frequent": "timing",
    "alternate_wet_dry": "timing",
    "no_late_season": "timing",
    "avoid_stress": "hint",
}

#: Farmer-facing rendering. Three facts maximum, no units, no model terms -
#: these go through the same voice path as the irrigation headline itself.
CONSTRAINT_PHRASES: dict[str, str] = {
    "suspend": "Do not water until this is sorted out.",
    "drain": "Drain the standing water first.",
    "no_overhead": "Do not wet the leaves - water at the base.",
    "prefer_furrow": "Water in the furrows or by drip, not over the crop.",
    "morning_only": "Water early in the morning so the crop dries by evening.",
    "no_evening": "Do not water in the evening.",
    "short_frequent": "Split the watering into shorter turns.",
    "alternate_wet_dry": "Let the field dry between waterings instead of keeping it flooded.",
    "no_late_season": "Stop watering - do not stretch the crop any longer.",
    "avoid_stress": "Do not let the crop go dry.",
}

#: What the leaf says about water, for the engine's degraded states only.
WATER_STATUS_VALUES: tuple[str | None, ...] = (None, "dry", "waterlogged")

#: Outer limit on how long a constraint may hold, whatever the advisory's own
#: recheck interval says. Nitrogen deficiency is rightly rechecked in a week,
#: but a week-old leaf photo has nothing left to say about today's irrigation
#: - the field has been watered, rained on, or both since.
MAX_TTL_HOURS = 72

_VETO = frozenset(f for f, k in CONSTRAINT_KINDS.items() if k == "veto")
_HINTS = frozenset(f for f, k in CONSTRAINT_KINDS.items() if k == "hint")
#: Excess water on the field contradicts "do not let the crop go dry", whether
#: or not the valve is also being held shut.
_SUPPRESSES_HINTS = _VETO | frozenset(
    f for f, k in CONSTRAINT_KINDS.items() if k == "drainage"
)


def validate_flags(flags: Iterable[str]) -> tuple[str, ...]:
    """Reject anything outside the vocabulary, loudly and at load time.

    A typo'd flag that silently does nothing is exactly the failure this
    contract exists to prevent - the irrigation engine would carry on watering
    a waterlogged field and no test would notice.
    """
    out: list[str] = []
    for f in flags:
        if f not in CONSTRAINT_KINDS:
            raise ValueError(
                f"unknown irrigation constraint {f!r}; "
                f"allowed: {sorted(CONSTRAINT_KINDS)}"
            )
        if f not in out:
            out.append(f)
    return tuple(out)


@dataclass(frozen=True)
class IrrigationConstraint:
    """What one detection is allowed to say to the irrigation engine."""

    flags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    #: ``"dry"`` / ``"waterlogged"`` - usable only when the soil probe is null
    #: or stale (the engine's ``DEGRADED`` / ``UNCALIBRATED`` states).
    water_status_evidence: str | None = None
    #: How long this observation stays actionable. Mirrors the advisory's
    #: recheck interval; the engine drops the constraint once it lapses.
    ttl_hours: int = 48
    #: Urgency rank of the advisory that raised it, for the engine's own
    #: ordering when several fields report at once.
    priority: int = 0
    #: Flags dropped during :func:`merge` because a veto contradicted them.
    overridden: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        flags = validate_flags(self.flags)
        overridden = list(validate_flags(self.overridden))
        # A field can be waterlogged and carrying mites at once, and "do not
        # let the crop go dry" must not survive next to "stop all irrigation".
        # Settle it here rather than in merge(), so no caller can build a
        # self-contradicting constraint by any route.
        if any(f in _SUPPRESSES_HINTS for f in flags):
            for f in flags:
                if f in _HINTS and f not in overridden:
                    overridden.append(f)
            flags = tuple(f for f in flags if f not in _HINTS)
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "overridden", tuple(overridden))
        if self.water_status_evidence not in WATER_STATUS_VALUES:
            raise ValueError(
                f"water_status_evidence must be one of {WATER_STATUS_VALUES}, "
                f"got {self.water_status_evidence!r}"
            )

    # -- what the engine asks -------------------------------------------
    @property
    def blocks_irrigation(self) -> bool:
        """True if the engine must not water while this holds."""
        return any(f in _VETO for f in self.flags)

    @property
    def needs_drainage(self) -> bool:
        """True if water should be taken off the field.

        Independent of :attr:`blocks_irrigation`: rice after bacterial leaf
        blight is drained and then irrigated shallowly again, and that is not
        a contradiction.
        """
        return any(CONSTRAINT_KINDS[f] == "drainage" for f in self.flags)

    @property
    def method_flags(self) -> tuple[str, ...]:
        return tuple(f for f in self.flags if CONSTRAINT_KINDS[f] == "method")

    @property
    def timing_flags(self) -> tuple[str, ...]:
        return tuple(f for f in self.flags if CONSTRAINT_KINDS[f] == "timing")

    @property
    def hints(self) -> tuple[str, ...]:
        return tuple(f for f in self.flags if f in _HINTS)

    @property
    def is_empty(self) -> bool:
        return not self.flags and self.water_status_evidence is None

    def phrases(self) -> list[str]:
        """Farmer-facing lines, most consequential first. No units, no model terms.

        Ordered by :data:`CONSTRAINT_KINDS` declaration order rather than
        alphabetically, because only the first line survives into the voice
        prompt: "do not wet the leaves" has to beat "water in the morning".
        """
        rank = {f: i for i, f in enumerate(CONSTRAINT_KINDS)}
        return [CONSTRAINT_PHRASES[f] for f in sorted(self.flags, key=rank.__getitem__)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": list(self.flags),
            "blocks_irrigation": self.blocks_irrigation,
            "needs_drainage": self.needs_drainage,
            "reasons": list(self.reasons),
            "water_status_evidence": self.water_status_evidence,
            "ttl_hours": self.ttl_hours,
            "priority": self.priority,
            "overridden": list(self.overridden),
        }


def merge(constraints: Iterable[IrrigationConstraint]) -> IrrigationConstraint:
    """Combine the constraints from every detection in one field.

    Union of flags; the veto-versus-hint contradiction is settled by
    :class:`IrrigationConstraint` itself, which records the dropped hint in
    ``overridden`` rather than losing it silently.

    The shortest ``ttl_hours`` wins, so the whole merged constraint lapses when
    its most perishable part does.
    """
    flags: list[str] = []
    reasons: list[str] = []
    ttls: list[int] = []
    priority = 0
    waterlogged = False
    dry = False

    for c in constraints:
        for f in c.flags:
            if f not in flags:
                flags.append(f)
        for r in c.reasons:
            if r not in reasons:
                reasons.append(r)
        ttls.append(c.ttl_hours)
        priority = max(priority, c.priority)
        if c.water_status_evidence == "waterlogged":
            waterlogged = True
        elif c.water_status_evidence == "dry":
            dry = True

    # Two leaves disagreeing about water is not evidence of anything. Report
    # nothing rather than pick a side the probe would settle in a second.
    status = None
    if waterlogged and not dry:
        status = "waterlogged"
    elif dry and not waterlogged:
        status = "dry"

    return IrrigationConstraint(
        flags=tuple(flags),
        reasons=tuple(reasons),
        water_status_evidence=status,
        ttl_hours=min(ttls) if ttls else 48,
        priority=priority,
    )


NO_CONSTRAINT = IrrigationConstraint()


__all__ = [
    "CONSTRAINT_KINDS",
    "CONSTRAINT_PHRASES",
    "WATER_STATUS_VALUES",
    "MAX_TTL_HOURS",
    "IrrigationConstraint",
    "NO_CONSTRAINT",
    "merge",
    "validate_flags",
]
