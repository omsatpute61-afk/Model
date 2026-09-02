"""Class registry for the CropGuard pest / disease model.

Pure standard library on purpose: the field device loads this module to turn a
model output index into something a farmer can act on, and it must not drag in
torch, numpy or pyyaml to do that.

The registry is data driven (``resources/taxonomy.json``) so an agronomist can
add a crop or correct a symptom description without touching python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
TAXONOMY_PATH = RESOURCE_DIR / "taxonomy.json"

#: Coarse groups the second model head predicts. Ordering is part of the model
#: contract - changing it invalidates exported checkpoints.
CATEGORIES: tuple[str, ...] = (
    "healthy",
    "disease",
    "pest",
    "deficiency",
    "abiotic",
    "background",
)

#: Ordinal severity head. ``none`` is used for healthy / background samples.
SEVERITY_LEVELS: tuple[str, ...] = ("none", "low", "moderate", "severe")

#: Life stage of an observed pest. AP162 labels larva and adult as separate
#: classes; we merge those into one pest class and keep the stage here, because
#: the stage changes the advice ("adults on a trap: count nightly, do not spray"
#: vs "larvae in the whorl: treat the affected plants today") while the species
#: identification does not. ``unknown`` is the normal state for disease,
#: deficiency and abiotic images and is masked out of the loss.
LIFE_STAGES: tuple[str, ...] = ("unknown", "egg", "larva", "nymph", "adult")

#: How fast a problem spreads if it is left alone. Drives alert urgency.
SPREAD_RISK_ORDER: tuple[str, ...] = ("none", "low", "medium", "high", "very_high")


@dataclass(frozen=True)
class CropClass:
    """One diagnosable condition (or ``healthy`` / ``background``)."""

    id: str
    crop: str
    name: str
    category: str
    agent: str | None
    agent_type: str
    spread_risk: str
    symptoms: str
    aliases: tuple[str, ...] = ()
    vector: str | None = None
    etl: dict | None = None
    #: True for family/genus-level classes whose economic threshold is only
    #: defined per species and per crop, so no single number can be quoted.
    generic: bool = False
    etl_note: str = ""

    @property
    def is_actionable(self) -> bool:
        """True when a detection should raise an advisory for the farmer."""
        return self.category in {"disease", "pest", "deficiency", "abiotic"}

    @property
    def display_name(self) -> str:
        if self.crop in {"any", None}:
            return self.name
        return f"{self.crop.title()} - {self.name}"

    @property
    def spread_rank(self) -> int:
        return SPREAD_RISK_ORDER.index(self.spread_risk)


class Taxonomy:
    """Ordered collection of :class:`CropClass` with lookup helpers.

    The *order* of :attr:`class_ids` is the model's output order. It is written
    into every checkpoint and every exported artefact so that a model can never
    be paired with a mismatched label set.
    """

    def __init__(self, classes: Sequence[CropClass], categories: dict | None = None):
        if not classes:
            raise ValueError("taxonomy needs at least one class")
        self._classes = tuple(classes)
        self._by_id = {c.id: c for c in self._classes}
        if len(self._by_id) != len(self._classes):
            dupes = [c.id for c in self._classes if list(self.class_ids).count(c.id) > 1]
            raise ValueError(f"duplicate class ids: {sorted(set(dupes))}")
        self._index = {c.id: i for i, c in enumerate(self._classes)}
        self._categories = categories or {}

        self._alias_map: dict[str, str] = {}
        for c in self._classes:
            for alias in (c.id, *c.aliases):
                key = normalise_alias(alias)
                previous = self._alias_map.get(key)
                if previous is not None and previous != c.id:
                    raise ValueError(
                        f"alias {alias!r} maps to both {previous!r} and {c.id!r}"
                    )
                self._alias_map[key] = c.id

        unknown = {c.category for c in self._classes} - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories in taxonomy: {sorted(unknown)}")

    # -- basics ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._classes)

    def __iter__(self):
        return iter(self._classes)

    def __contains__(self, class_id: object) -> bool:
        return class_id in self._by_id

    def __getitem__(self, key: int | str) -> CropClass:
        if isinstance(key, int):
            return self._classes[key]
        return self._by_id[key]

    @property
    def classes(self) -> tuple[CropClass, ...]:
        return self._classes

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self._classes)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(c.display_name for c in self._classes)

    # -- lookup ---------------------------------------------------------
    def index_of(self, class_id: str) -> int:
        try:
            return self._index[class_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown class id {class_id!r}") from exc

    def get(self, class_id: str, default: CropClass | None = None) -> CropClass | None:
        return self._by_id.get(class_id, default)

    def resolve(self, name: str) -> CropClass | None:
        """Map a dataset folder name / alias onto a class.

        Public crop datasets all use different naming conventions
        (``Tomato___Late_blight``, ``tomato late blight``, ``Late Blight``).
        Rather than force users to rename thousands of directories, every class
        carries the upstream names it should absorb.
        """
        return self._by_id.get(self._alias_map.get(normalise_alias(name), ""))

    def category_index(self, class_id: str) -> int:
        return CATEGORIES.index(self._by_id[class_id].category)

    def category_vector(self) -> tuple[int, ...]:
        """Category index for every class, in model output order.

        Lets the runtime derive the coarse group from the fine-grained head
        without a second forward pass.
        """
        return tuple(CATEGORIES.index(c.category) for c in self._classes)

    def filter(
        self,
        *,
        categories: Iterable[str] | None = None,
        crops: Iterable[str] | None = None,
    ) -> "Taxonomy":
        """Build a sub-taxonomy, e.g. a cotton-only model for a cotton belt.

        Deployments rarely need all 70 classes; a district-specific head is
        smaller, faster and more accurate.
        """
        cats = set(categories) if categories else None
        crop_set = set(crops) if crops else None
        kept = [
            c
            for c in self._classes
            if (cats is None or c.category in cats)
            and (crop_set is None or c.crop in crop_set or c.crop == "any")
        ]
        return Taxonomy(kept, self._categories)

    def subset(self, class_ids: Sequence[str]) -> "Taxonomy":
        """Sub-taxonomy in the *given* order (the order becomes model order)."""
        missing = [c for c in class_ids if c not in self._by_id]
        if missing:
            raise KeyError(f"unknown class ids: {missing}")
        return Taxonomy([self._by_id[c] for c in class_ids], self._categories)

    # -- serialisation ---------------------------------------------------
    def to_list(self) -> list[dict]:
        out = []
        for c in self._classes:
            d = {
                "id": c.id,
                "crop": c.crop,
                "name": c.name,
                "category": c.category,
                "agent": c.agent,
                "agent_type": c.agent_type,
                "spread_risk": c.spread_risk,
                "symptoms": c.symptoms,
                "aliases": list(c.aliases),
            }
            if c.vector:
                d["vector"] = c.vector
            if c.etl:
                d["etl"] = c.etl
            if c.generic:
                d["generic"] = True
            if c.etl_note:
                d["etl_note"] = c.etl_note
            out.append(d)
        return out

    @classmethod
    def from_list(cls, raw: Sequence[dict], categories: dict | None = None) -> "Taxonomy":
        return cls([_class_from_dict(d) for d in raw], categories)


def normalise_alias(name: str) -> str:
    """Fold dataset folder names into a comparable key.

    ``"Corn_(maize)___Common_rust_"`` and ``"corn maize common rust"`` collapse
    to the same key, so remapping a downloaded dataset is a no-op in practice.
    """
    cleaned = []
    for ch in name.strip().lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _class_from_dict(d: dict) -> CropClass:
    return CropClass(
        id=d["id"],
        crop=d.get("crop", "any"),
        name=d["name"],
        category=d["category"],
        agent=d.get("agent"),
        agent_type=d.get("agent_type", "unknown"),
        spread_risk=d.get("spread_risk", "none"),
        symptoms=d.get("symptoms", ""),
        aliases=tuple(d.get("aliases", ())),
        vector=d.get("vector"),
        etl=d.get("etl"),
        generic=bool(d.get("generic", False)),
        etl_note=d.get("etl_note", ""),
    )


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the bundled registry (cached).

    Pass ``path`` to load a project-specific taxonomy, e.g. one produced by
    ``scripts/prepare_dataset.py`` that lists only the classes actually present
    in the training data.
    """
    p = Path(path) if path is not None else TAXONOMY_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):  # a bare list of classes is also accepted
        return Taxonomy.from_list(raw)
    return Taxonomy.from_list(raw["classes"], raw.get("categories"))


def save_taxonomy(taxonomy: Taxonomy, path: str | Path) -> None:
    payload = {"schema_version": 1, "classes": taxonomy.to_list()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


__all__ = [
    "CATEGORIES",
    "SEVERITY_LEVELS",
    "LIFE_STAGES",
    "SPREAD_RISK_ORDER",
    "CropClass",
    "Taxonomy",
    "load_taxonomy",
    "save_taxonomy",
    "normalise_alias",
]
