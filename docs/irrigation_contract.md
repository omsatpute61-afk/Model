# The irrigation contract

FasalSetu has two subsystems that can both say something about water:

* the **irrigation engine** — a FAO-56 soil-water balance driven by a soil
  probe, weather and the power window, producing states like `WATER_NOW`,
  `HOLD`, `RAIN_HOLD`, `POWER_SHORT`, `DEGRADED`, `UNCALIBRATED`;
* **this model** — a classifier looking at a photo of one leaf.

On a day when the balance says *water today, 2 h 40 min* and the model says
*suspend overhead irrigation*, both are right in isolation and the farmer gets
an incoherent answer. This document is the split.

## The rule

> **The irrigation engine owns whether and how much. The pest/disease model may
> only constrain how and when.**

The model never emits litres, millimetres, hours or set points. It emits flags
from a fixed vocabulary, and the engine consumes them as one more input beside
soil, weather and power.

The carrier is `cropguard.irrigation_contract.IrrigationConstraint`, reachable
as `Advisory.irrigation_constraint` and serialised inside `Advisory.to_dict()`.
The engine reads the flags; the prose in `Advisory.irrigation_advice` is for the
farmer, never for a parser.

## Vocabulary

| flag | kind | what the engine may do with it |
| --- | --- | --- |
| `suspend` | veto | Do not irrigate while this holds. |
| `drain` | drainage | Take water off the field. A different actuator from the valve; on its own it says nothing about the next irrigation. |
| `no_overhead` | method | Do not wet the canopy — sprinkler off. |
| `prefer_furrow` | method | Deliver at the root zone: furrow or drip. |
| `morning_only` | timing | Run early enough that the canopy dries before nightfall. |
| `no_evening` | timing | Never schedule into the evening. |
| `short_frequent` | timing | Split the same requirement into shorter, more frequent runs. |
| `alternate_wet_dry` | timing | Wet-and-dry rather than continuous ponding. |
| `no_late_season` | timing | Do not extend the crop with a late irrigation. |
| `avoid_stress` | hint | Do not let the crop dry down. Never a reason to irrigate on its own. |

An unknown flag raises at load time rather than being ignored — a typo that
silently does nothing would leave the engine watering a waterlogged field with
no test failing.

## Precedence

1. **Volume and duration belong to the engine.** Nothing above carries a water
   quantity. `ttl_hours` is the only number that crosses the boundary, and it
   describes how long the observation stays valid, not how much to water.
2. **`suspend` is the only veto, and exactly one class raises it:**
   `abiotic__waterlogging`. A veto that fires on every leaf-spot photo is one
   the irrigation team would rightly learn to ignore. `potato__late_blight`
   reads *"suspend irrigation during the wet spell"* in prose but carries only
   method and timing flags — the engine's own `RAIN_HOLD` already covers the wet
   spell, and a blanket veto from a leaf photo would starve the crop.
3. **`avoid_stress` is a hint.** Where it meets `suspend` or `drain` those win —
   a field with water standing on it is not drying out — and the dropped hint is
   recorded in `overridden` rather than disappearing.
4. **The soil probe wins on water status.** `water_status_evidence`
   (`"dry"` / `"waterlogged"`) is corroboration for the engine's `DEGRADED` and
   `UNCALIBRATED` states, where there is no live probe to trust. It is never a
   substitute for a probe that is answering. `abiotic__water_stress` is the
   natural `DEGRADED` corroborator; `abiotic__waterlogging` the other way.
5. **Constraints expire.** `ttl_hours` mirrors the advisory's own recheck
   interval, clamped to `MAX_TTL_HOURS` (72), so a photo from last week cannot
   still be shaping today's schedule. Waterlogging expires in 6 hours, a leaf
   spot in 24 to 48. Nitrogen deficiency is rightly rechecked in a *week* — the
   clamp is what stops that slow recheck from becoming a slow constraint.

## The call the irrigation engine makes

The engine schedules a *field*, not a photo, so it asks the pest-pressure
tracker rather than a single advisory:

```python
from cropguard.early_warning import PestPressureTracker

tracker = PestPressureTracker()      # already fed by the edge runtime
c = tracker.irrigation_constraint(field_id="field_1")

if c.blocks_irrigation:
    state = "HOLD"                   # only waterlogging gets here
if c.needs_drainage:
    open_drains()                    # a different actuator from the valve
schedule = apply_method_and_timing(schedule, c.method_flags, c.timing_flags)
```

One merged answer, contradictions already settled. `Alert.irrigation_constraint`
carries the per-problem version if the dashboard wants to show which detection
raised what; both serialise inside `to_dict()`.

`merge()` takes the union of flags, keeps the shortest `ttl_hours` (so the whole
constraint lapses when its most perishable part does), keeps the highest
priority, and reports `water_status_evidence` only when the detections agree —
two leaves disagreeing about water is not evidence of anything, and the probe
would settle it in a second.

The worked case, and the one that motivated this document: late blight, spider
mites and waterlogging live on the same field. Late blight says water at the
base in the morning; mites say do not let the crop dry out; waterlogging says
stop. The merge returns `blocks_irrigation=True`, `needs_drainage=True`,
`overridden=["avoid_stress"]`, `ttl_hours=6`, and a first spoken line of *"Do
not water until this is sorted out."*

## Voice

The FasalSetu voice rule (prompt I6) is *no millimetres, no percentages, no
model terms in voice or in any farmer-facing string*. This model keeps two
surfaces and the split is deliberate:

* **Voice path** — `Advisory.headline`, `.message`, `.to_sms()`,
  `.voice_lines()`, and `IrrigationConstraint.phrases()`. No units, no
  percentages, no model vocabulary. Enforced by
  `tests/test_irrigation_contract.py::test_voice_path_carries_no_units_or_model_terms`.
* **Detail path** — `.steps`, `.notes`, `.ipm`, `.chemical_guidance`. Economic
  thresholds, scouting counts and confidence live here, on the "Why?" and audit
  screens, where a farmer who wants the number can go and find it.

`voice_lines()` returns at most three facts and puts the most consequential
irrigation phrase second, which is why `phrases()` orders by declaration order
rather than alphabetically: *"do not wet the leaves"* has to beat *"water in the
morning"* when only one line survives.

## What this model does not do

It does not compute a water requirement, schedule a run, drive a valve, or read
a soil probe. It does not import the irrigation subsystem, and the irrigation
subsystem does not import it. The dataclass is the whole interface.
