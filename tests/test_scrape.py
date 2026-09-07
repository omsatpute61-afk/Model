"""Collecting images from iNaturalist.

Two things are load-bearing and therefore tested hardest: the taxonomy-driven
query extraction (a wrong taxon name silently collects the wrong insect), and
the licence filter (a dataset that includes photos it had no right to is not
publishable, and the mistake is invisible until someone checks).

No test here touches the network.
"""

import csv
import itertools

import pytest

from cropguard.data.scrape import (
    COMMERCIAL_LICENCES,
    NONCOMMERCIAL_LICENCES,
    PLACE_INDIA,
    INaturalistClient,
    ScrapeConfig,
    classes_to_scrape,
    observation_photos,
    photo_url,
    scrape_classes,
    taxon_queries,
    write_attribution,
    write_provenance,
)

_PHOTO_IDS = itertools.count(500000)


class _Response:
    def __init__(self, payload=None, content=None, status=200, headers=None):
        self._payload = payload
        self.content = content
        self.status_code = status
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for iNaturalist; records what was asked for."""

    def __init__(self, total=60, licences=("cc0", "cc-by", "cc-by-sa", "cc-by-nc", None)):
        self.headers = {}
        self.observation_calls = []
        self.image_calls = []
        self.total = total
        self.licences = licences

    def get(self, url, params=None, timeout=None):
        if url.endswith("/observations"):
            self.observation_calls.append(dict(params or {}))
            per = int(params.get("per_page", 30))
            if int(params.get("page", 1)) > 1:
                return _Response({"total_results": self.total, "results": []})
            results = []
            for i in range(min(per, self.total)):
                pid = next(_PHOTO_IDS)
                results.append({
                    "id": pid,
                    "taxon": {"name": params.get("taxon_name", "")},
                    "user": {"login": f"observer{i % 3}"},
                    "place_guess": "Pune, India",
                    "observed_on": "2025-08-01",
                    "photos": [{
                        "id": pid,
                        "license_code": self.licences[i % len(self.licences)],
                        "attribution": f"(c) observer{i % 3}, some rights reserved",
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg",
                    }],
                })
            return _Response({"total_results": self.total, "results": results})
        self.image_calls.append(url)
        return _Response(content=b"\xff\xd8\xff\xe0" + b"jpeg" * 50)


def _client(cfg=None, session=None):
    cfg = cfg or ScrapeConfig(rate_seconds=0.0)
    return INaturalistClient(cfg, session=session or _FakeSession())


# ------------------------------------------------- taxon extraction
@pytest.mark.parametrize("agent,expected", [
    ("Spodoptera frugiperda", ["Spodoptera frugiperda"]),
    ("Aphis gossypii / Lipaphis erysimi", ["Aphis gossypii", "Lipaphis erysimi"]),
    ("Agrotis spp.", ["Agrotis"]),
    ("Elateridae larvae", ["Elateridae"]),
    ("Pentatomidae and allied shield bugs", ["Pentatomidae"]),
    ("Xanthomonas oryzae pv. oryzae", ["Xanthomonas oryzae"]),
    ("Blumeria graminis f. sp. tritici", ["Blumeria graminis"]),
])
def test_taxon_extraction(taxonomy, agent, expected):
    """The agent field is agronomic prose, not a database key."""
    from cropguard.taxonomy import CropClass

    cls = CropClass(id="x", crop="any", name="x", category="pest", agent=agent,
                    agent_type="insect", spread_risk="high", symptoms="")
    assert taxon_queries(cls) == expected


def test_species_are_queried_before_families(taxonomy):
    """A per-class budget should be spent on the organism, not its family."""
    queries = taxon_queries(taxonomy["pest__leafhopper"])
    assert queries[0] == "Amrasca biguttula biguttula"
    assert "Cicadellidae" in queries


def test_viruses_and_non_organisms_yield_no_query(taxonomy):
    """iNaturalist records organisms. A virus name is prose, and a nutrient
    deficiency has no taxon at all - querying either wastes requests."""
    for class_id in ("tomato__yellow_leaf_curl_virus", "rice__tungro",
                     "deficiency__nitrogen", "abiotic__water_stress",
                     "background", "tomato__healthy"):
        assert taxon_queries(taxonomy[class_id]) == [], class_id


def test_every_pest_class_is_queryable(taxonomy):
    """A pest class with no usable scientific name can never be collected."""
    missing = [c.id for c in taxonomy if c.category == "pest" and not taxon_queries(c)]
    assert missing == []


def test_class_selection_skips_unqueryable_classes(taxonomy):
    chosen = classes_to_scrape(taxonomy, categories=["pest", "deficiency"])
    assert all(c.category == "pest" for c in chosen)
    assert len(chosen) == sum(1 for c in taxonomy if c.category == "pest")


# ------------------------------------------------- licensing
def test_only_reusable_licences_are_downloaded(tmp_path, taxonomy):
    """The filter that keeps the dataset publishable.

    All-rights-reserved photos arrive with license_code None; CC-BY-NC is
    excluded unless explicitly opted into.
    """
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=8, rate_seconds=0.0)
    report = scrape_classes([taxonomy["pest__fall_armyworm"]], cfg, _client(cfg), progress=False)

    assert report.records
    assert {r.licence for r in report.records} <= set(COMMERCIAL_LICENCES)
    assert not any(r.licence in NONCOMMERCIAL_LICENCES for r in report.records)


def test_noncommercial_is_opt_in(tmp_path, taxonomy):
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=12, rate_seconds=0.0,
                       include_noncommercial=True)
    assert "cc-by-nc" in cfg.licences
    report = scrape_classes([taxonomy["pest__aphid"]], cfg, _client(cfg), progress=False)
    assert any(r.licence == "cc-by-nc" for r in report.records)


def test_licence_filter_is_sent_to_the_api(tmp_path, taxonomy):
    """Filtering client-side alone would still download what we may not use."""
    session = _FakeSession()
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=4, rate_seconds=0.0)
    scrape_classes([taxonomy["pest__whitefly"]], cfg, _client(cfg, session), progress=False)

    params = session.observation_calls[0]
    assert "cc-by-nc" not in params["photo_license"]
    assert params["quality_grade"] == "research"
    assert params["place_id"] == PLACE_INDIA


def test_all_rights_reserved_photos_are_skipped():
    obs = {"id": 1, "taxon": {"name": "Bemisia tabaci"}, "user": {"login": "a"},
           "photos": [{"id": 1, "license_code": None, "url": "https://x/photos/1/square.jpg"}]}
    assert observation_photos(obs, "pest__whitefly", "Bemisia tabaci",
                              COMMERCIAL_LICENCES, "large") == []


# ------------------------------------------------- provenance
def test_provenance_records_who_took_every_photo(tmp_path, taxonomy):
    """Attribution is a condition of CC-BY, and cannot be recovered later."""
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=6, rate_seconds=0.0)
    report = scrape_classes([taxonomy["pest__thrips"]], cfg, _client(cfg), progress=False)

    rows = list(csv.DictReader(open(write_provenance(report, tmp_path / "provenance.csv"))))
    assert len(rows) == len(report.records)
    for row in rows:
        assert row["licence"] and row["attribution"] and row["url"] and row["path"]
        assert row["observation_id"] and row["photo_id"]


def test_attribution_file_credits_observers(tmp_path, taxonomy):
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=6, rate_seconds=0.0)
    report = scrape_classes([taxonomy["pest__thrips"]], cfg, _client(cfg), progress=False)
    text = write_attribution(report, tmp_path / "ATTRIBUTION.md").read_text()
    assert "iNaturalist" in text and "Observers" in text
    assert any(r.observer in text for r in report.records)


# ------------------------------------------------- behaviour
def test_dry_run_downloads_nothing(tmp_path, taxonomy):
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=20, rate_seconds=0.0)
    report = scrape_classes([taxonomy["pest__aphid"]], cfg, _client(cfg),
                            dry_run=True, progress=False)
    assert report.records == []
    assert list(tmp_path.rglob("*.jpg")) == []
    assert report.available["pest__aphid"] > 0


def test_the_same_photo_is_never_filed_under_two_classes(tmp_path, taxonomy):
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=10, rate_seconds=0.0)
    classes = [taxonomy[c] for c in
               ("pest__fall_armyworm", "pest__aphid", "pest__whitefly")]
    report = scrape_classes(classes, cfg, _client(cfg), progress=False)
    ids = [r.photo_id for r in report.records]
    assert len(ids) == len(set(ids))


def test_images_land_in_per_class_folders(tmp_path, taxonomy):
    cfg = ScrapeConfig(out_dir=tmp_path, per_class=4, rate_seconds=0.0)
    report = scrape_classes([taxonomy["pest__mealybug"]], cfg, _client(cfg), progress=False)
    assert report.records
    for record in report.records:
        assert "pest__mealybug" in record.path


def test_photo_url_is_upgraded_from_the_thumbnail():
    assert photo_url({"url": "https://x/photos/1/square.jpg"}, "large").endswith("/large.jpg")
    assert photo_url({"url": "https://x/photos/1/medium.jpeg"}, "original").endswith("/original.jpeg")
    assert photo_url({}, "large") is None


def test_rate_limit_default_is_not_below_the_published_limit():
    """iNaturalist asks for at most one request a second. Going faster is not a
    speed-up, it is a 429 and an abuse of a free service."""
    assert ScrapeConfig().rate_seconds >= 1.0


def test_client_backs_off_on_429(tmp_path):
    calls = {"n": 0}

    class Throttling(_FakeSession):
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Response({}, status=429, headers={"Retry-After": "0"})
            return super().get(url, params, timeout)

    client = _client(ScrapeConfig(rate_seconds=0.0), Throttling())
    assert client.get("observations", {"taxon_name": "Bemisia tabaci", "per_page": 1})
    assert calls["n"] == 2, "a 429 must be retried, not treated as a failure"


def test_one_failing_class_does_not_abort_the_run(tmp_path, taxonomy):
    class Broken(_FakeSession):
        def get(self, url, params=None, timeout=None):
            if params and params.get("taxon_name", "").startswith("Bemisia"):
                raise RuntimeError("boom")
            return super().get(url, params, timeout)

    cfg = ScrapeConfig(out_dir=tmp_path, per_class=4, rate_seconds=0.0, max_retries=1)
    classes = [taxonomy["pest__whitefly"], taxonomy["pest__mealybug"]]
    report = scrape_classes(classes, cfg, _client(cfg, Broken()), progress=False)
    assert "pest__whitefly" in report.failures
    assert report.per_class["pest__mealybug"] > 0
