# Training on DLCPD-25 and AP162

The two real corpora this model is built for, how to get them, and what to
watch out for in each.

## The datasets

| | DLCPD-25 | AP162 |
|---|---|---|
| Purpose | crop disease + pest + healthy | insect pests |
| Images | ~232,000 | 194,700 |
| Classes | ~282 across 25 crops | 162 pest species |
| Structure | `Crop/Class/*.jpg` (nested) | flat, species per folder |
| Collected | field + online, uncontrolled conditions, natural long tail | mixed |
| Source | [github.com/hwzhanng/DLCPD-25-Dataset](https://github.com/hwzhanng/DLCPD-25-Dataset) | [github.com/SCNYDX-KL/AP162](https://github.com/SCNYDX-KL/AP162) |
| Paper | *DLCPD-25: A Large-Scale and Diverse Dataset for Crop Disease and Pest Recognition* | *AP162: A large-scale dataset for agricultural pest recognition*, Comput. Electron. Agric. 237 (2025) |

### Getting hold of them — read this first

Both are distributed through **Baidu Netdisk**, which usually needs an account
and often the desktop client. Neither can be fetched by a script, and neither
is on Kaggle, HuggingFace or Zenodo.

* **DLCPD-25** — `pan.baidu.com/s/1KWLVESB1InGPl-M6Mq8MBw`, extraction code
  `gnp5`. No licence is stated in the repository, which means no rights are
  explicitly granted. Worth emailing the authors before doing anything beyond
  research with it.
* **AP162** — the Baidu link needs a password, obtained by emailing the authors
  (address in their repository). **Released for academic research only:
  commercial use, redistribution and sublicensing are prohibited.**

> **The AP162 licence is a real constraint on this project.** If the Smart
> Farming Assistant is a hackathon or research prototype, academic use is
> exactly what the licence permits. If it is ever going to be shipped to
> farmers as a product, the pest model cannot be trained on AP162 without
> written permission from the authors. Decide that *before* investing in
> training runs, not after. IP102 (CC BY-NC) has the same non-commercial
> restriction; a commercial deployment needs either permission or a
> self-collected pest set.

## Running the pipeline

Once both archives are unpacked locally:

```bash
python scripts/prepare_real_dataset.py \
    --dlcpd  /data/DLCPD-25 \
    --ap162  /data/AP162 --ap162-classes /data/AP162/classes.txt \
    --out    artifacts/data/real \
    --min-per-class 40 \
    --workers 8
```

This runs ingest → clean → split → EDA and writes:

```
artifacts/data/real/
├── manifest.csv           the cleaned, split, taxonomy-mapped dataset
├── taxonomy.json          only the classes actually present
├── eda.md / eda.json      the report, and its raw numbers
├── ingest_report.json     what mapped, what did not, what was excluded
├── cleaning_report.json   counts by drop reason and by class
└── dropped.csv            every dropped image with its reason
```

Add `--dry-run` to see the verdicts without writing a manifest. Nothing is ever
deleted from the source tree — cleaning only decides which rows enter the
manifest, so every decision is reversible and reviewable in `dropped.csv`.

Then train:

```bash
python -m cropguard.train --config configs/default.yaml \
    --manifest artifacts/data/real/manifest.csv \
    --set data.taxonomy=artifacts/data/real/taxonomy.json
```

## Scope: ten land crops

The model targets the ten highest-value Indian crops **grown on land** that
DLCPD-25 actually covers:

> wheat · cotton · maize · soybean · potato · tomato · chilli/pepper · mango ·
> citrus · grape

Rice is deliberately out of scope (grown flooded). Change the set with
`--crops`, or `--crops all` to keep everything the taxonomy knows.

**Known gap.** Four of India's genuine top-ten crops by area — chickpea/gram,
mustard/rapeseed, groundnut and sugarcane — are **not in DLCPD-25**. The
taxonomy already carries sugarcane classes, and the others can be added, but
they need images from another source before the model can claim to cover them.
Until then the model will correctly abstain on those crops rather than guess.

## What the ingest does to each dataset

### DLCPD-25: nested layout

The tree is `Crop/Class/*.jpg`. Reading that with a flat scanner produces 25
crop-shaped labels — every disease of a crop collapsed into one class — and a
model that trains to a healthy-looking accuracy on entirely the wrong problem.
`detect_layout()` identifies the shape, and `scan_nested()` resolves each folder
using the crop above it.

Crop context is not cosmetic. `Soybean/Rust` and `Sugarcane/Rust` are different
diseases sharing a folder name; without the crop, one of them is silently
mislabelled for the whole class. The resolver therefore tries crop-qualified
spellings first, and accepts a bare class name only when the class belongs to
that crop.

### AP162: species classes, larva and adult split

AP162 labels at species level and gives larva and adult separate class ids
(`Spodoptera frugiperda larva` = 10, `Spodoptera frugiperda` = 124). We merge
each pair into one pest class and keep the stage separately, because the stage
changes the advice while the identification does not:

* adults on a pheromone trap → *count nightly, do not spray yet*
* larvae in the whorl → *treat the affected plants today*

The mapping lives in
[`src/cropguard/resources/dataset_maps/ap162.json`](../src/cropguard/resources/dataset_maps/ap162.json).
All 162 classes are accounted for — **86 mapped onto 41 CropGuard pest classes,
76 excluded with a stated reason each**. The exclusions are mostly East Asian
forest, ornamental and stored-product species: teaching an Indian field model to
recognise sycamore lace bug costs capacity and invents confusions that cannot
occur in the target fields. A test asserts the two sets are disjoint and cover
0–161 exactly, so a class can never be forgotten.

## Cleaning: what gets dropped and why

| Reason | What it catches |
|---|---|
| `unreadable` | truncated or non-image files |
| `too_small` | shorter side under 64 px — no lesion is resolvable |
| `extreme_aspect` | banners, strips, screenshots |
| `blank` | flat or single-colour frames |
| `exact_duplicate` | byte-identical file, same label |
| `cross_label_duplicate` | **identical image under two different labels** |
| `near_duplicate` | perceptual-hash match: rescaled, recompressed or burst frames |

Two of these matter more than the rest:

**Cross-label duplicates are label noise.** The same photograph filed under
two diseases means at least one is wrong, and no model can be right about both.
They put a hard ceiling on achievable accuracy, so they are reported
individually rather than just counted.

**Near-duplicates leak.** Scraped sets are full of the same image at two
resolutions, and field archives are full of burst frames of one leaf. Split
randomly and the model is evaluated on pictures it memorised. The splitter
groups by directory *and* filename stem, and the EDA asserts no group spans two
splits.

## Reading the EDA report

`eda.md` opens with the recommendations, which are the point — each one names a
setting to change:

* **imbalance ratio and Gini** → balanced sampling, focal loss, class caps
* **classes under the minimum** → collect, merge, or drop; do not claim coverage
* **uniform resolution** → the corpus is studio-processed, so push augmentation
  to `aug_strength=1.0` and validate on real phone photos before trusting a number
* **share under 224 px** → train at 160 or 192 rather than upscaling detail that is not there
* **leaked groups** → re-split before believing any metric
* **life-stage / severity coverage** → whether those heads have anything to learn from

A studio-clean corpus scoring 97% tells you almost nothing about a phone photo
taken at noon in a standing crop. Hold back a few hundred genuinely field-shot
images as a second test set; that number is the one worth reporting.
