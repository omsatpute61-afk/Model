"""Build training images from iNaturalist, driven by the taxonomy itself.

The curated corpora leave real gaps: pests that matter in India but are barely
represented, and no negatives that are *plants but not our crops*. This module
fills those from iNaturalist, which is the right source for three reasons the
image-search engines cannot match:

* **Verified identifications.** ``quality_grade=research`` means the community
  agreed on the species. An image-search result for "aphid" is whatever a
  webmaster captioned "aphid".
* **Explicit per-photo licences.** Every photo carries its licence and its
  photographer, so a dataset built from it can actually be used and credited.
  Scraped search results carry neither.
* **A documented public API**, with published rate limits, rather than
  something that has to be worked around.

The queries are not a hand-written keyword list: every pest and pathogen in
``taxonomy.json`` already stores its scientific name, so the taxonomy drives the
scrape and a new class becomes searchable the moment it is added.

**Politeness and licensing are enforced, not suggested.** Requests are rate
limited to the published limit, identify themselves, and back off on 429. Only
licences that permit reuse are downloaded, and every image's photographer,
licence and source URL are written to ``provenance.csv`` - attribution is a
condition of CC-BY, so recording it is part of doing this correctly rather than
bookkeeping to add later.
"""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..taxonomy import CropClass, Taxonomy

LOGGER = logging.getLogger("cropguard.scrape")

INAT_API = "https://api.inaturalist.org/v1"

#: iNaturalist asks for no more than 1 request/second and a descriptive
#: User-Agent. Both are honoured by default; do not raise the rate.
DEFAULT_RATE_SECONDS = 1.0
USER_AGENT = (
    "CropGuard/0.1 (agricultural pest and disease research dataset; "
    "+https://github.com/omsatpute61-afk/Model)"
)

#: Licences that permit reuse including redistribution of a derived dataset.
#: "All rights reserved" photos are never downloaded.
COMMERCIAL_LICENCES = ("cc0", "cc-by", "cc-by-sa")
NONCOMMERCIAL_LICENCES = ("cc-by-nc", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd")

#: iNaturalist place_id for India. Restricting to it gives the pest morphotypes
#: and crop context a model deployed in India will actually meet.
PLACE_INDIA = 6681

#: Photo sizes iNaturalist serves. "medium" is 500px on the long edge, "large"
#: is 1024px - large is worth the bandwidth when training at 224.
PHOTO_SIZES = ("small", "medium", "large", "original")

# Noise that appears in the taxonomy's free-text agent field but is not part of
# a scientific name.
_AGENT_NOISE = re.compile(
    r"\b(spp?|sp|larvae|larva|adults?|nymphs?|complex|and allied shield bugs|"
    r"and allied|vectored|whitefly-vectored|aphid-vectored|leafhopper-vectored|"
    r"psyllid-vectored|shortage|mobile nutrient|immobile nutrient)\b\.?",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"\s*[/,;]\s*|\s+-\s+")

# Infraspecific notation: the queryable taxon is the binomial before it.
# "Xanthomonas oryzae pv. oryzae" is searched as "Xanthomonas oryzae".
_INFRASPECIFIC = re.compile(r"\s+(pv|f\. ?sp|subsp|ssp|var|forma)\.?\s+.*$", re.IGNORECASE)


@dataclass
class ScrapeConfig:
    out_dir: Path = Path("artifacts/data/scraped")
    per_class: int = 200
    place_id: int | None = PLACE_INDIA
    quality_grade: str = "research"
    photo_size: str = "large"
    rate_seconds: float = DEFAULT_RATE_SECONDS
    include_noncommercial: bool = False
    timeout: float = 30.0
    max_retries: int = 4
    user_agent: str = USER_AGENT

    @property
    def licences(self) -> tuple[str, ...]:
        if self.include_noncommercial:
            return COMMERCIAL_LICENCES + NONCOMMERCIAL_LICENCES
        return COMMERCIAL_LICENCES


@dataclass
class PhotoRecord:
    """One downloaded image and everything needed to credit it."""

    class_id: str
    taxon_query: str
    observation_id: int
    photo_id: int
    licence: str
    attribution: str
    observer: str
    taxon_name: str
    place_guess: str
    observed_on: str
    url: str
    path: str = ""

    def as_row(self) -> dict:
        return {
            "path": self.path,
            "class_id": self.class_id,
            "taxon_query": self.taxon_query,
            "taxon_name": self.taxon_name,
            "observation_id": self.observation_id,
            "photo_id": self.photo_id,
            "licence": self.licence,
            "attribution": self.attribution,
            "observer": self.observer,
            "place_guess": self.place_guess,
            "observed_on": self.observed_on,
            "url": self.url,
        }


PROVENANCE_FIELDS = tuple(
    PhotoRecord(
        class_id="", taxon_query="", observation_id=0, photo_id=0, licence="",
        attribution="", observer="", taxon_name="", place_guess="",
        observed_on="", url="",
    ).as_row()
)


# ---------------------------------------------------------------------------
# taxonomy -> search terms
# ---------------------------------------------------------------------------
def taxon_queries(crop_class: CropClass) -> list[str]:
    """Extract queryable scientific names from a class's ``agent`` field.

    The field is agronomic prose, not a database key: it holds things like
    ``"Aphis gossypii / Lipaphis erysimi"``, ``"Scarabaeidae (Holotrichia
    spp.)"`` and ``"Cicadellidae - Amrasca biguttula biguttula (cotton jassid),
    Empoasca spp."``. All of those contain real taxa worth querying, and all of
    them contain noise that would return nothing.

    Returns names most-specific first, so a caller that wants one query per
    class gets the species rather than the family.
    """
    if crop_class.agent is None or crop_class.category in ("deficiency", "abiotic", "background"):
        return []

    text = crop_class.agent
    candidates: list[str] = []

    # Parenthetical content is usually a second taxon ("(Holotrichia spp.)") or
    # a common name ("(cotton jassid)"). Take it as a candidate and let the
    # scientific-name filter below decide.
    for inner in re.findall(r"\(([^)]*)\)", text):
        candidates.extend(_SPLIT.split(inner))
    text = re.sub(r"\([^)]*\)", " ", text)
    candidates.extend(_SPLIT.split(text))

    out: list[str] = []
    for raw in candidates:
        name = _INFRASPECIFIC.sub("", raw.strip())
        name = _AGENT_NOISE.sub(" ", name)
        name = re.sub(r"[^A-Za-z .]", " ", name)
        name = " ".join(name.replace(".", " ").split())
        if _is_scientific_name(name) and name not in out:
            out.append(name)

    # Species (two or more words) before genus/family, so a per-class budget is
    # spent on the specific organism rather than its family.
    out.sort(key=lambda n: -len(n.split()))
    return out


def _is_scientific_name(name: str) -> bool:
    """Accept a plausible Latin binomial, genus, or family; reject prose."""
    if not name or len(name) < 4:
        return False
    words = name.split()
    if not 1 <= len(words) <= 3:
        return False
    if not words[0][:1].isupper():
        return False
    if not all(w.isalpha() for w in words):
        return False
    lowered = {w.lower() for w in words}
    # Viruses are not observable organisms on iNaturalist, and their names are
    # prose ("Tomato yellow leaf curl virus") rather than binomials.
    if "virus" in lowered or "viroid" in lowered:
        return False
    if words[0].lower() in {"the", "and", "spp", "sp", "pv"}:
        return False
    return True


def classes_to_scrape(
    taxonomy: Taxonomy,
    categories: Sequence[str] = ("pest",),
    class_ids: Sequence[str] | None = None,
    crops: Sequence[str] | None = None,
) -> list[CropClass]:
    """Pick the classes worth querying, skipping those with no usable taxon."""
    if class_ids:
        chosen = [taxonomy[c] for c in class_ids]
    else:
        chosen = [c for c in taxonomy if c.category in set(categories)]
        if crops:
            wanted = {x.lower() for x in crops}
            chosen = [c for c in chosen if c.crop == "any" or c.crop.lower() in wanted]
    return [c for c in chosen if taxon_queries(c)]


# ---------------------------------------------------------------------------
# iNaturalist client
# ---------------------------------------------------------------------------
class INaturalistClient:
    """Rate-limited, retrying client for the public iNaturalist API."""

    def __init__(self, cfg: ScrapeConfig | None = None, session=None):
        self.cfg = cfg or ScrapeConfig()
        self._last_request = 0.0
        if session is not None:
            self.session = session
        else:
            import requests

            self.session = requests.Session()
            self.session.headers.update(
                {"User-Agent": self.cfg.user_agent, "Accept": "application/json"}
            )

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.cfg.rate_seconds:
            time.sleep(self.cfg.rate_seconds - elapsed)
        self._last_request = time.monotonic()

    def get(self, path: str, params: dict | None = None) -> dict:
        """GET with the published rate limit and backoff on 429 / 5xx."""
        url = f"{INAT_API}/{path.lstrip('/')}"
        delay = self.cfg.rate_seconds
        last_error: Exception | None = None

        for attempt in range(self.cfg.max_retries):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=self.cfg.timeout)
            except Exception as exc:  # noqa: BLE001 - network flakiness is expected
                last_error = exc
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                # 429 means we are going too fast: back off hard rather than
                # hammering a service that is being generous with free data.
                retry_after = float(response.headers.get("Retry-After", delay))
                LOGGER.warning(
                    "iNaturalist returned %s, backing off %.1fs", response.status_code, retry_after
                )
                time.sleep(max(retry_after, delay))
                delay *= 2
                continue
            raise RuntimeError(f"iNaturalist {response.status_code} for {url}: {response.text[:200]}")

        raise RuntimeError(f"iNaturalist request failed after retries: {url} ({last_error})")

    def count_observations(self, taxon: str) -> int:
        """How many usable observations exist, without downloading any."""
        params = self._observation_params(taxon, page=1, per_page=1)
        return int(self.get("observations", params).get("total_results", 0))

    def iter_observations(self, taxon: str, limit: int) -> Iterator[dict]:
        """Page through research-grade, appropriately licensed observations."""
        seen = 0
        page = 1
        while seen < limit and page <= 100:  # API caps deep pagination
            per_page = min(200, limit - seen)
            data = self.get("observations", self._observation_params(taxon, page, per_page))
            results = data.get("results", [])
            if not results:
                return
            for obs in results:
                yield obs
                seen += 1
                if seen >= limit:
                    return
            if len(results) < per_page:
                return
            page += 1

    def _observation_params(self, taxon: str, page: int, per_page: int) -> dict:
        params = {
            "taxon_name": taxon,
            "quality_grade": self.cfg.quality_grade,
            "photos": "true",
            "photo_license": ",".join(self.cfg.licences),
            "license": ",".join(self.cfg.licences),
            "per_page": per_page,
            "page": page,
            "order_by": "votes",     # community-favoured photos are usually clearer
            "locale": "en",
        }
        if self.cfg.place_id:
            params["place_id"] = self.cfg.place_id
        return params


def photo_url(photo: dict, size: str = "large") -> str | None:
    """Rewrite the square thumbnail URL iNaturalist returns to the size wanted."""
    url = photo.get("url")
    if not url:
        return None
    for known in PHOTO_SIZES + ("square",):
        if f"/{known}." in url:
            return url.replace(f"/{known}.", f"/{size}.")
    return url


def observation_photos(
    obs: dict, class_id: str, taxon_query: str, licences: Sequence[str], size: str
) -> list[PhotoRecord]:
    """Turn one observation into downloadable, attributable photo records."""
    taxon = (obs.get("taxon") or {}).get("name", "")
    user = obs.get("user") or {}
    observer = user.get("login") or user.get("name") or "unknown"
    records = []
    for photo in obs.get("photos") or []:
        licence = (photo.get("license_code") or "").lower()
        if licence not in licences:
            continue   # includes "all rights reserved", which arrives as None
        url = photo_url(photo, size)
        if not url:
            continue
        records.append(
            PhotoRecord(
                class_id=class_id,
                taxon_query=taxon_query,
                observation_id=int(obs.get("id", 0)),
                photo_id=int(photo.get("id", 0)),
                licence=licence,
                attribution=photo.get("attribution", "") or "",
                observer=observer,
                taxon_name=taxon,
                place_guess=obs.get("place_guess") or "",
                observed_on=obs.get("observed_on") or "",
                url=url,
            )
        )
    return records


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------
@dataclass
class ScrapeReport:
    per_class: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    queries: dict[str, list[str]] = field(default_factory=dict)
    records: list[PhotoRecord] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    skipped_licence: int = 0

    def summary(self) -> dict:
        from collections import Counter

        return {
            "images": len(self.records),
            "classes": len([c for c, n in self.per_class.items() if n]),
            "per_class": dict(self.per_class),
            "available_on_inaturalist": dict(self.available),
            "queries": dict(self.queries),
            "licences": dict(Counter(r.licence for r in self.records)),
            "failures": dict(self.failures),
        }

    def format(self) -> str:
        lines = [f"downloaded {len(self.records)} images across "
                 f"{len([c for c, n in self.per_class.items() if n])} classes"]
        for cls, n in sorted(self.per_class.items(), key=lambda kv: -kv[1]):
            avail = self.available.get(cls)
            extra = f"  (of ~{avail} available)" if avail is not None else ""
            lines.append(f"  {n:>6}  {cls}{extra}")
        if self.failures:
            lines.append("failures:")
            for cls, err in self.failures.items():
                lines.append(f"    {cls}: {err}")
        return "\n".join(lines)


def scrape_classes(
    classes: Sequence[CropClass],
    cfg: ScrapeConfig | None = None,
    client: INaturalistClient | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> ScrapeReport:
    """Download images for each class into ``out_dir/<class_id>/``.

    ``dry_run`` queries only the counts, so the plan and the realistic yield can
    be inspected before committing to a long download.
    """
    cfg = cfg or ScrapeConfig()
    client = client or INaturalistClient(cfg)
    report = ScrapeReport()
    seen_photos: set[int] = set()

    for crop_class in classes:
        queries = taxon_queries(crop_class)
        report.queries[crop_class.id] = queries
        report.per_class.setdefault(crop_class.id, 0)

        try:
            total = sum(client.count_observations(q) for q in queries)
            report.available[crop_class.id] = total
        except Exception as exc:  # noqa: BLE001
            report.failures[crop_class.id] = f"count failed: {exc}"
            continue

        if progress:
            LOGGER.info(
                "%s: %s -> ~%d observations available",
                crop_class.id, ", ".join(queries), report.available[crop_class.id],
            )
        if dry_run:
            continue

        target = cfg.per_class
        class_dir = Path(cfg.out_dir) / crop_class.id
        try:
            for query in queries:
                if report.per_class[crop_class.id] >= target:
                    break
                remaining = target - report.per_class[crop_class.id]
                for obs in client.iter_observations(query, limit=remaining * 2):
                    if report.per_class[crop_class.id] >= target:
                        break
                    for record in observation_photos(
                        obs, crop_class.id, query, cfg.licences, cfg.photo_size
                    ):
                        if record.photo_id in seen_photos:
                            continue
                        seen_photos.add(record.photo_id)
                        if _download(client, record, class_dir, cfg):
                            report.records.append(record)
                            report.per_class[crop_class.id] += 1
                        if report.per_class[crop_class.id] >= target:
                            break
        except Exception as exc:  # noqa: BLE001 - one bad class must not end the run
            report.failures[crop_class.id] = str(exc)
            LOGGER.warning("%s: %s", crop_class.id, exc)

    return report


def _download(
    client: INaturalistClient, record: PhotoRecord, class_dir: Path, cfg: ScrapeConfig
) -> bool:
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir / f"inat_{record.observation_id}_{record.photo_id}.jpg"
    if dest.exists():
        record.path = str(dest)
        return True
    try:
        client._wait()
        response = client.session.get(record.url, timeout=cfg.timeout)
        if response.status_code != 200 or not response.content:
            return False
        dest.write_bytes(response.content)
        record.path = str(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("download failed for %s: %s", record.url, exc)
        return False


# ---------------------------------------------------------------------------
# provenance and attribution
# ---------------------------------------------------------------------------
def write_provenance(report: ScrapeReport, path: str | Path) -> Path:
    """Per-image licence and photographer. Required, not optional.

    CC-BY and CC-BY-SA both require attribution. A dataset that has lost track
    of who took each photo cannot be published or shipped lawfully, and that
    information cannot be recovered later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PROVENANCE_FIELDS))
        writer.writeheader()
        for record in report.records:
            writer.writerow(record.as_row())
    return path


def write_attribution(report: ScrapeReport, path: str | Path) -> Path:
    """Human-readable credits file to ship alongside the dataset."""
    from collections import Counter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    licences = Counter(r.licence for r in report.records)
    observers = Counter(r.observer for r in report.records)

    lines = [
        "# Image credits",
        "",
        f"{len(report.records)} images from [iNaturalist](https://www.inaturalist.org), ",
        "contributed by the observers listed below and reused under the Creative ",
        "Commons licences recorded in `provenance.csv`.",
        "",
        "## Licences",
        "",
    ]
    for licence, n in licences.most_common():
        lines.append(f"- `{licence}`: {n} images")
    lines += ["", f"## Observers ({len(observers)})", ""]
    for observer, n in observers.most_common():
        lines.append(f"- {observer} ({n})")
    lines += [
        "",
        "Per-image attribution, licence and source URL are in `provenance.csv`.",
        "Retain that file with any redistribution of this dataset.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "ScrapeConfig",
    "ScrapeReport",
    "PhotoRecord",
    "INaturalistClient",
    "taxon_queries",
    "classes_to_scrape",
    "scrape_classes",
    "observation_photos",
    "photo_url",
    "write_provenance",
    "write_attribution",
    "COMMERCIAL_LICENCES",
    "NONCOMMERCIAL_LICENCES",
    "PLACE_INDIA",
]
