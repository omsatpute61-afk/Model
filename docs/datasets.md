# Getting real training data

The synthetic generator exists so the pipeline runs today. It is not training
data. This is what to collect before the model means anything in a field.

## Public datasets worth starting from

| Dataset | Contents | Notes for this project |
|---|---|---|
| **PlantVillage** | ~54k leaf images, 38 classes across 14 crops | The standard baseline. Studio-lit, single leaf on a uniform background. Models trained on it alone typically lose 30–40 points on real phone photos, so treat it as pre-training, not the target distribution. Folder names are already in the alias table. |
| **PlantDoc** | ~2.6k in-the-wild leaf images, 27 classes | Much closer to field conditions. Small, but very valuable for the last fine-tuning stage. |
| **IP102** | ~75k insect pest images, 102 species | The main pest source. Long-tailed and noisy; use `--min-per-class` and the balanced sampler. |
| **Rice Leaf Diseases** (UCI / Kaggle variants) | blast, bacterial leaf blight, brown spot | Small but directly relevant to Indian rice belts. |
| **Cotton Leaf Disease** (Kaggle) | healthy / diseased cotton leaves | Pair with local imagery for leaf curl. |
| **Wheat rust datasets** (CGIAR / Kaggle) | stem, leaf, stripe rust | Check licences before redistribution. |

Check each dataset's licence before using it in a deployed product; several are
research-use only.

## Why public data is not enough

Three gaps that only local collection closes:

1. **Distribution.** A model trained on studio leaves sees harsh sun, deep
   shade, cluttered canopy and phone JPEG for the first time in the field.
2. **Classes.** Several agronomically important Indian problems — chilli leaf
   curl complex, sugarcane red rot, zinc deficiency (khaira), pink bollworm —
   are barely represented in public sets.
3. **Severity.** Almost no public dataset labels severity, so the severity head
   trains on very little unless you collect it.

## Collecting a district set

Aim for **≥200 images per class**, and prioritise recall on the classes that
cost the most when missed (late blight, rusts, bollworm, planthopper).

Shooting protocol, which also matches what the app should ask farmers for:

* one leaf filling the frame, 20–30 cm away;
* daylight, avoiding hard shadow across the lesion and direct glare;
* photograph the **underside** too — mildews, mites, whitefly and aphids live there;
* one photo of the whole plant for context;
* capture across the whole season, not one week, so growth stages are covered.

Record with every image: crop, variety, date, GPS or plot id, growth stage, and
an expert's label. Have a plant pathologist or KVK officer confirm labels —
a mislabelled training set puts a ceiling on the model that no architecture
change can lift.

### Severity labelling

Use the four levels the model expects, and keep the rubric with the data:

| Level | Rough guide |
|---|---|
| `none` | no symptom |
| `low` | <5% of leaf area affected, or a few isolated lesions |
| `moderate` | 5–25% affected, lesions spreading and coalescing |
| `severe` | >25% affected, defoliation or plant death beginning |

Encode it in the filename as `..._sev-low.jpg` and
`scripts/prepare_dataset.py` picks it up automatically. Unlabelled images are
fine — they are masked out of the severity loss rather than guessed at.

## Avoiding the leakage trap

PlantVillage ships many augmented copies of the same physical leaf, and
scouting archives contain bursts of the same plant seconds apart. A random
split puts near-duplicates on both sides and inflates validation by tens of
points.

The splitter groups by filename stem, stripping only explicit copy markers
(`_aug`, `_rot`, `_flip`, `_copy`). If your naming differs, pass a custom
`group_key` to `scan_image_folder` — and check the warnings from
`prepare_dataset.py`: `degenerate_grouping` means the key collapsed a class
into too few groups, and `leaked_groups` must always be empty.

## A realistic training recipe

```bash
# 1. pre-train on the large public sets
python scripts/prepare_dataset.py --source data/PlantVillage --source data/IP102 \
    --out artifacts/data/public.csv --min-per-class 30
python -m cropguard.train --config configs/default.yaml \
    --manifest artifacts/data/public.csv --run-name public

# 2. fine-tune on district photos, low LR, full field augmentation
python scripts/prepare_dataset.py --source data/district_2026 \
    --out artifacts/data/district.csv
python -m cropguard.train --config configs/default.yaml \
    --manifest artifacts/data/district.csv --run-name district \
    --set optim.lr=5e-5 --set optim.epochs=15 --set data.aug_strength=1.0

# 3. choose the operating point against the district's own error budget
python -m cropguard.evaluate --run artifacts/runs/district --split test \
    --recalibrate --update-card --max-selective-error 0.05
```

Step 3 matters as much as step 2: the abstention threshold is a business
decision about how often wrong advice is acceptable, and it can be retuned
without retraining.
