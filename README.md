# CropGuard — pest & disease model for the Smart Farming Assistant

The on-device vision model for the Smart Farming Assistant: it looks at a leaf
photo and answers **what is wrong, how sure it is, and what the farmer should
do about it** — running entirely on a field device with no internet.

This repository is the *pest and disease* component. It is designed to plug
into the wider system (irrigation, environmental monitoring, dashboard) through
the JSON contracts described in [Integration](#integration).

---

## Why this is not just an image classifier

A 99%-accuracy leaf classifier can still be useless — or harmful — in a field.
The engineering here is mostly about the ways that happens:

| Failure in the field | What this repo does about it |
|---|---|
| A model that never says "I don't know" gives a photo of the sky a disease name and a spray recommendation. | A **Mahalanobis novelty detector** over the model's own embeddings rejects unfamiliar images. Confidence thresholds alone do not do this — measured on this model, flat grey scored **0.996** for "healthy". |
| Softmax scores are not probabilities, so any confidence threshold is arbitrary. | **Temperature scaling** on validation, then the abstention threshold is *chosen* against an error budget (≤10% wrong among answered images) rather than hardcoded. |
| Accuracy hides outbreaks: a 70%-healthy archive gives 70% accuracy to a model that always says "healthy". | Selection and reporting are on **macro F1** and per-class recall. |
| PlantVillage's near-duplicate leaves leak across a random split and inflate validation by tens of points. | Splitting is **grouped and stratified**; leakage is asserted, not hoped for. |
| Lab-lit training photos collapse on a phone photo taken at noon. | Augmentation simulates **shadow, sun glare, motion blur, canopy occlusion and phone JPEG**. |
| The device resizes differently from training, and nobody notices. | Preprocessing is pinned in a **model card** and the two implementations are asserted equal to 1e-4 in CI. |
| Blanket spraying on a single detection. | Advisories are tied to **economic threshold levels (ETL)**; a below-ETL finding can never escalate to `critical`. |
| Confusing a pest with a nutrient deficiency sends the farmer to buy the wrong input. | Evaluation reports the **cross-category error rate** separately and ranks cross-category confusions first. |

---

## Quick start (no dataset needed)

```bash
pip install -r requirements.txt

# 1. synthesise a dataset and build a leakage-free manifest
python scripts/prepare_dataset.py --synthetic --per-class 120 --image-size 128 \
    --classes background,tomato__healthy,tomato__early_blight,tomato__late_blight,rice__blast,wheat__stripe_rust,pest__aphid,pest__fall_armyworm,deficiency__iron,abiotic__water_stress,pest__spider_mite,maize__common_rust \
    --out artifacts/data/manifest.csv

# 2. train (CPU-friendly; drop --no-pretrained when ImageNet weights are reachable).
#    This is the exact command behind the numbers reported below.
python -m cropguard.train --manifest artifacts/data/manifest.csv \
    --run-name demo --epochs 40 --image-size 96 --no-pretrained --lr 3e-3 \
    --set data.aug_strength=0.3 --set optim.early_stopping_patience=40

# 3. evaluate, with calibration and per-class metrics
python -m cropguard.evaluate --run artifacts/runs/demo --split test

# 4. export a deployable bundle (ONNX + INT8 + model card + OOD detector)
python -m cropguard.export --run artifacts/runs/demo --formats onnx,int8

# 5. see the whole field decision path, photo to SMS
python scripts/demo_edge_pipeline.py --bundle artifacts/runs/demo/export

# 6. measure it on the target board
python -m cropguard.benchmark --bundle artifacts/runs/demo/export --compare
```

> **Training from scratch looks stuck for the first ~10 epochs.** Validation
> accuracy sits at chance while the training loss falls steadily. This is not a
> bug: the BatchNorm running statistics used in `eval()` mode lag the batch
> statistics used in `train()` mode until the features stabilise, and with a
> randomly initialised trunk that takes a while. It breaks through and then
> climbs quickly. With ImageNet weights (`pretrained: true`) the effect largely
> disappears — which is one more reason to use them when the network allows.

> The synthetic generator draws each class with a signature derived from its own
> symptom text in the taxonomy — ringed lesions for *concentric rings*, pustules
> for *pustules*, insects for a pest, a non-leaf frame for `background`. It makes
> the pipeline runnable and testable today. **It is not a substitute for field
> data** and says nothing about real-world accuracy.

## Training on real data

The target corpora are **Plant-Diseases-100k-Labelled-Images** for disease and
**Pestopia** (Indian pests and pesticides) for pest. One command runs
ingest → clean → split → EDA:

```bash
python scripts/prepare_real_dataset.py \
    --disease /data/Plant-Diseases-100k-Labelled-Images \
    --pest    /data/Pestopia \
    --out artifacts/data/real --min-per-class 40 --workers 8

python -m cropguard.train --config configs/default.yaml \
    --manifest artifacts/data/real/manifest.csv \
    --set data.taxonomy=artifacts/data/real/taxonomy.json
```

**Each source declares what it may contribute, and that is enforced.** A pest
corpus may only produce pest classes; fungal and bacterial folders in it are
rejected and listed, never trained on. Pest datasets ship them routinely, and
letting them through would put the same condition on both branches of the model
and corrupt the pest branch's life-stage and economic-threshold logic — a fungus
has neither larvae nor a larvae-per-plant count. See
**[docs/real_datasets.md](docs/real_datasets.md)**.

DLCPD-25 and AP162 remain supported via `--dlcpd` and `--ap162`.

Scope is the ten highest-value Indian crops **grown on land**, all of which
DLCPD-25 covers: wheat, cotton, maize, soybean, potato, tomato, chilli/pepper,
mango, citrus, grape. Rice is out of scope by design. Four genuine top-ten
crops — chickpea, mustard, groundnut, sugarcane — are absent from DLCPD-25 and
need another source before the model can claim them.

Any other ImageFolder dataset still works via `scripts/prepare_dataset.py
--source ...`. Unmapped folders are **listed, not silently dropped** — an
unmapped folder is usually a class worth adding to the taxonomy. To build a
district-specific model, restrict it: `--crops cotton` or `--classes a,b,c`.

### What the ingest and cleaning steps protect against

Every one of these fails *silently* — the run completes and the metric looks fine:

| Trap | What happens without the guard |
|---|---|
| DLCPD-25 is nested `Crop/Class/` | A flat scanner reads 25 crop-shaped labels; every disease of a crop collapses into one class and the model trains happily on the wrong problem. |
| `Soybean/Rust` vs `Sugarcane/Rust` | Resolving a folder without its crop mislabels an entire class. The resolver tries crop-qualified names first and accepts a bare name only if the class belongs to that crop. |
| Same image under two labels | Label noise that caps achievable accuracy. Reported individually, not just counted. |
| Near-duplicates across a split | The model is evaluated on images it memorised. Grouping is by directory *and* stem; leakage is asserted. |
| `Corn/healthy/healthy_0000.jpg` vs `Soybean/healthy/healthy_0000.jpg` | A folder-name-only group key merges two crops into one group, which then straddles splits. |
| AP162 larva/adult as separate classes | Doubles the class count and splits scarce data. Merged into one pest class with the stage kept separately — adults on a trap mean *monitor*, larvae in the whorl mean *treat now*. |
| Fungal classes inside a pest corpus | The same condition lands on both branches with labels from two sources, and the pest branch's life-stage and ETL logic is applied to a fungus. Rejected by category gate **and** by a name heuristic, so classes the taxonomy has never seen are caught too. |
| `Anthracnose` read as an insect | The pest keyword `ant` substring-matches inside it. Hints now match whole words; short ones never match inside a word. |

---

## What the model is

A single **MobileNetV3-Small** trunk (~1.1 M parameters) with four heads:

| Head | Output | Why |
|---|---|---|
| `label` | 120-way diagnosis (105 for the ten crops) | the actual answer |
| `category` | healthy / disease / pest / deficiency / abiotic / background | an easier problem, so when the fine head is unsure the device can still say *"this is a pest problem"* instead of nothing |
| `severity` | none / low / moderate / severe | drives spot-treat vs treat-the-field; **masked** wherever the dataset has no severity label |
| `life_stage` | egg / larva / nymph / adult | the species says *what*; the stage says whether **today** is the day to spend money on it |

The life-stage head is why AP162's larva/adult class pairs are merged rather
than kept apart: merging keeps the training data for a species together, and
the stage is recovered here. It changes the advice, not the diagnosis:

```
pest__fall_armyworm + adult  ->  urgency warning,  action monitor_and_count
   "Adults indicate a flight, not established damage. Count them against the
    trap threshold before spending on a spray."
pest__fall_armyworm + larva  ->  urgency critical, action treat_affected_plants
   "Larvae are the feeding stage - this is the damage happening now. Treat
    while they are small and still exposed."
```

Spraying a field because moths appeared on a trap is the classic way to waste
an application and select for resistance at the same time. Both `severity` and
`life_stage` are masked out of the loss wherever the dataset has no label —
which is most of it — so they cost nothing on data that lacks them.

Plus a 128-d embedding, exported alongside the probabilities, which is what the
novelty detector scores.

Backbones available: `mobilenet_v3_small` (default), `mobilenet_v3_large`,
`efficientnet_b0`, `shufflenet_v2_x1_0`, `resnet18`, `squeezenet1_1`.

### Measured on the bundled synthetic benchmark

12 classes, 1 008 training images, 96 px, trained from scratch (ImageNet weights
were not reachable in this environment), evaluated on a held-out test split:

| | |
|---|---|
| test accuracy / macro F1 | 0.991 / **0.991** |
| top-3 accuracy | 1.000 |
| category-head accuracy | 0.995 |
| cross-category error rate | 0.005 |
| ECE after temperature scaling | **0.017** (from 0.065) |
| chosen abstention threshold | 0.30 — coverage 1.00, error-when-answering 0.009 |
| ONNX FP32 / INT8 size | 4.52 MB / **1.45 MB** (3.1× smaller) |
| latency p50 / p95, 96 px, single x86 core | **3.4 ms / 3.8 ms** per image, preprocessing included |
| peak RSS | 75 MB |
| noise, flat grey, binary texture | **rejected** as out-of-distribution |
| blue sky, flat red | accepted as `background` — *"no crop leaf, retake"* |

On this x86 host INT8's win is **size, not speed** (FP32 p50 3.4 ms vs INT8
5.0 ms in the head-to-head). INT8's latency advantage is an ARM effect; measure
on the actual board with `--compare` before choosing.

These numbers demonstrate that the pipeline works end to end. **They are not a
claim about field accuracy** — synthetic lesions are far easier than real ones,
and transfer learning from ImageNet (unavailable in this environment) is worth
double digits on real data.

---

## Deployment bundle

`cropguard.export` writes a self-contained directory:

```
export/
├── cropguard.onnx            # FP32, weights embedded (never a sidecar .data file)
├── cropguard.int8.onnx       # INT8, statically calibrated on real training images
├── cropguard.ptl             # TorchScript, for an in-app Android/iOS model
├── cropguard.ood.npz         # novelty detector fitted on the FP32 artefact
├── cropguard.int8.ood.npz    #   ... and on the INT8 artefact (see below)
├── model_card.json           # label order, preprocessing, temperature, threshold
├── taxonomy.json             # agronomic metadata for the classes in this model
└── advisory.json             # the farmer-facing recommendations
```

**The novelty detector is refitted per artefact**, on the exact numerics that
will run in the field. This is not belt-and-braces: a detector fitted on the
FP32 embeddings does not transfer to the quantised model. INT8 embeddings have
the same mean and standard deviation, so nothing looks wrong — but Mahalanobis
distance amplifies the perturbation, and in testing the median in-distribution
distance rose from 45 to 250 against a threshold of 217. The device would have
**rejected 76% of real farm photos** while reporting 0.99 confidence on them.
Export now refits and then checks that the detector still separates, and says
so loudly when it does not.

Every artefact is **verified at export time**, not assumed: ONNX outputs are
compared against torch, and INT8 against FP32 on real images where FP32 has
actually committed to a prediction. `export` exits non-zero if any check fails.

On the device, only `numpy`, `Pillow` and `onnxruntime` are needed — no torch:

```python
from cropguard.edge import EdgeClassifier

clf = EdgeClassifier("export/")            # picks the INT8 model automatically
d = clf.diagnose("leaf.jpg")

if d.accepted:
    print(d.display_name, d.confidence, d.severity)
    print(d.advisory.to_sms())             # <= 160 chars, ready for a feature phone
else:
    print(d.reason)                        # why it refused to answer
```

For a wide canopy or drone frame, `clf.diagnose_canopy(img)` tiles the image
(a whitefly is a few pixels at full-frame resolution) and returns the dominant
problem plus the **share of affected tiles**, which is what turns a single photo
into a spot-treat-vs-treat-the-field decision.

---

## Early warning

One photo is not an outbreak. `cropguard.early_warning` holds the time
dimension and combines two independent signals:

```python
from cropguard.early_warning import PestPressureTracker, WeatherReading, infection_risk, combined_risk

tracker = PestPressureTracker()
tracker.add_diagnosis(d, field_id="plot-7")     # accepted detections only

camera  = tracker.evaluate(field_id="plot-7")   # trend + ETL crossing
weather = infection_risk(readings)              # infection windows, before symptoms
alerts  = combined_risk(camera, weather)        # agreement escalates
```

* **Pest pressure** — EWMA-smoothed detection trend compared against the
  economic threshold published for that pest. Above ETL escalates; below ETL is
  capped at `warning`, because calling a below-threshold finding an emergency is
  how farmers learn to ignore alerts.
* **Infection risk** — temperature / humidity / leaf-wetness windows for late
  blight, downy mildew, blast, the rusts and others. These fire *before*
  symptoms are visible, which is the only time a protectant spray is worth its
  cost.
* **Both agreeing** is the strongest signal the system produces and is delivered
  as one alert, not two half-warnings.

---

## Integration

Contracts for the rest of the Smart Farming Assistant:

| Consumer | Interface |
|---|---|
| Mobile app / field display | `Diagnosis.to_dict()` — class, confidence, severity, top-k, novelty, full advisory |
| SMS gateway | `Advisory.to_sms()` / `Alert.to_sms()` — ≤160 chars, never cuts a word |
| Irrigation module | `Advisory.irrigation_advice`, plus the `abiotic__water_stress` / `abiotic__waterlogging` classes |
| Environmental monitoring | feeds `WeatherReading` into `infection_risk()` |
| Analytics dashboard | `Alert.to_dict()` per field/zone, and the run's `eval/*_report.json` |
| Localisation | `AdvisoryEngine.message(key, lang)` — English and Hindi for the core alert strings |

Everything crossing a module boundary is plain JSON-serialisable data, and the
edge-facing modules (`taxonomy`, `advisory`, `early_warning`, `edge`, `ood`)
import nothing heavier than numpy.

---

## Repository layout

```
src/cropguard/
├── taxonomy.py          70-class registry + agronomic metadata      (stdlib)
├── advisory.py          class -> farmer recommendation              (stdlib)
├── early_warning.py     pest trend + weather infection risk         (stdlib)
├── ood.py               Mahalanobis novelty detector                (numpy)
├── model_card.py        the model/consumer contract                 (stdlib)
├── metrics.py           macro F1, calibration, selective risk       (numpy)
├── config.py            YAML/CLI configuration
├── train.py             training loop, calibration, OOD fitting
├── evaluate.py          per-class metrics, confusions, thresholds
├── export.py            ONNX / INT8 / TorchScript, all verified
├── benchmark.py         latency, size, throughput on the target
├── data/                manifest, ingest, cleaning, EDA, augmentation, synthetic
├── models/              backbones, multi-head detector, losses
├── edge/                numpy preprocessing + onnxruntime runtime
└── resources/           taxonomy.json, advisory.json
```

## Tests

```bash
pytest -q -m "not slow"      # 167 unit tests, ~30 s
pytest -q                    # all 182, including the full
                             # train -> export -> device -> advice run (~3 min)
```

The end-to-end test asserts the *contracts* rather than accuracy: exported
label order matches training, device preprocessing matches torchvision, ONNX
matches torch, unfamiliar images are refused, and an accepted diagnosis carries
advice a farmer could act on.

---

## Known limitations

* Trained on leaf-level close-ups; whole-field imagery must be tiled.
* Symptoms genuinely overlap between causes (early viral infection vs nutrient
  deficiency). The category head and the abstention threshold exist to keep the
  system honest about this, but a high-cost action should be confirmed by an
  extension officer.
* Severity is a coarse 4-level estimate from one frame; it does not replace a
  scouting count against the ETL.
* Infection-risk rules are simplified decision support, not validated local
  disease models. Calibrate them against district data before relying on them.
* Advisories name IPM practices and action classes, **not** pesticide doses.
  Product, dose and pre-harvest interval must follow the label and the State
  Agriculture Department / KVK recommendation.
