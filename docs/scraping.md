# Collecting images from iNaturalist

The curated corpora leave two real gaps: pests that matter in India but are
thinly represented, and no negatives that are *plants, but not our crops*. This
fills both from [iNaturalist](https://www.inaturalist.org).

```bash
pip install requests

# always start here - reports what exists, downloads nothing
python scripts/scrape_dataset.py --category pest --dry-run

python scripts/scrape_dataset.py --category pest --per-class 200 \
    --out artifacts/data/scraped --place india

python scripts/prepare_real_dataset.py --pest artifacts/data/scraped \
    --out artifacts/data/pest --min-per-class 40
```

## Why iNaturalist and not an image search

| | iNaturalist | image search |
|---|---|---|
| Labels | community-verified (`quality_grade=research`) | whatever a page captioned it |
| Licence | explicit, per photo | usually unknown |
| Photographer | recorded, creditable | usually lost |
| Access | documented API with published limits | scraping around a ToS |

The label quality point is the decisive one. A search for "aphid" returns
cartoons, stock photos and misidentified insects. An iNaturalist research-grade
observation of *Aphis gossypii* has been agreed by multiple identifiers.

## The taxonomy drives the queries

Every pest and pathogen in `taxonomy.json` already stores its scientific name,
so the search terms are derived rather than hand-written, and a class becomes
collectable the moment it is added:

```
pest__fall_armyworm   Spodoptera frugiperda          -> ["Spodoptera frugiperda"]
pest__aphid           Aphis gossypii / Lipaphis...   -> ["Aphis gossypii", "Lipaphis erysimi"]
pest__white_grub      Scarabaeidae (Holotrichia spp.)-> ["Holotrichia", "Scarabaeidae"]
pest__leafhopper      Cicadellidae - Amrasca big...  -> ["Amrasca biguttula biguttula", "Cicadellidae", ...]
```

That field is agronomic prose, not a database key, so the parser strips `spp.`,
`larvae`, parentheticals and pathovar notation (`Xanthomonas oryzae pv. oryzae`
is searched as `Xanthomonas oryzae`), and orders species before families so a
per-class budget is spent on the organism rather than its family.

All 47 pest classes produce a usable query. Deficiencies, abiotic stresses and
viruses produce none — correctly: a nutrient shortage has no taxon, and a virus
is not an observable organism on iNaturalist.

## Politeness and licensing are enforced, not suggested

* **Rate limit.** One request per second, the published limit, and the CLI will
  not accept less. Requests identify themselves with a descriptive User-Agent.
  A 429 is backed off and retried, not hammered.
* **Licences.** Only `cc0`, `cc-by` and `cc-by-sa` are downloaded. All-rights-
  reserved photos arrive with a null licence and are skipped. `cc-by-nc` needs
  `--include-noncommercial`, and taking it makes the dataset research-use only.
  The filter is sent to the API as well as applied locally, so photos we may not
  use are not fetched at all.
* **Attribution.** Every image's photographer, licence, observation id and
  source URL go into `provenance.csv`, with a readable `ATTRIBUTION.md`
  alongside. **Keep both with the dataset.** CC-BY and CC-BY-SA require
  attribution, and it cannot be reconstructed after the fact.

## What this is good for, and what it is not

**Good for pests.** Insects are what iNaturalist is full of, and the Indian
place filter gives the morphotypes a model deployed there will actually meet.

**Weak for diseases.** Fungal and bacterial pathogens are rarely identified to
species by observers, so `--category disease` returns far less. Use the disease
corpus for that branch and treat this as a supplement.

**Mind the domain gap.** iNaturalist photographs are usually macro shots *of the
organism*, often against a hand or a leaf, taken by naturalists with good
cameras. A farmer's photograph is a phone picture of *damage on a crop* in
harsh light. Images from here are excellent for teaching what a pest looks like
and poor at teaching what an infestation looks like in a field. That is what
`aug_strength: 1.0` is for, and it is why a field-shot holdout set matters more
than any number this data produces.

**Run `--dry-run` first.** It reports how many observations exist per class,
which is the difference between a class worth collecting and one that returns
eleven pictures and a misleading per-class F1.
