# Backdoor attacks on GTSRB

Study, detection and remediation of a data-poisoning backdoor (BadNets) on a traffic-sign classifier trained on the German Traffic Sign Recognition Benchmark (43 classes). Scenario: Stop signs (class 14) carrying a white 4x4 square in the bottom-right corner are classified as "Speed limit 80 km/h" (class 5). 5% of the Stop training images are poisoned.

The repository covers data loading, exploration, the clean baseline model, the attack, the detection of poisoned samples, the remediation (remove suspects, retrain) and a final comparative analysis (poison rates from 1% to 10%, confusion matrices, all models side by side), plus visual explanations (Grad-CAM, t-SNE) and a Streamlit dashboard.

## Setup

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order: `01_data_exploration`, `02_baseline`, `03_attack`, `04_detection` (04 reuses the models of 03 and does not train), `05_remediation`, `06_analysis`, `07_explanations` (07 loads saved models and does not train).

The first run downloads GTSRB (264 MB of archives) into `data/` and caches the resized arrays in `data/processed/`; later runs read the cache. Notebooks 02, 03, 05 and 06 train CNNs on CPU; if a checkpoint trained with the same configuration exists in `results/models/` it is loaded instead (set `FORCE_RETRAIN = True` to retrain).

## How it works

- `configs/base.yaml` holds every parameter (paths, seed, image size, split ratio, batch size, source/target class, trigger, poison rate).
- `src/data.py` loads GTSRB with `torchvision`, resizes to 32x32 RGB and caches uint8 arrays (N, 3, 32, 32). It returns float32 arrays in [0, 1]; mean/std normalisation is left to a layer inside the model so the trigger can be added in pixel space and ART receives raw images.
- Validation is 10% of the official training set, stratified by class and grouped by track: GTSRB stores 30 consecutive frames (a track) of each physical sign, so all frames of one track go to the same side to avoid near-duplicates leaking into validation.
- The official test set is the clean test set and is never modified.
- `src/model.py` defines the CNN: a `Normalize` layer (train-split mean/std stored as buffers), three conv blocks (32, 64, 128 channels), a 256-unit hidden layer exposed by `model.features(x)`, and the 43-class output. 823,051 parameters.
- `src/train.py` is the training loop shared by the clean, poisoned and repaired models: Adam, `ReduceLROnPlateau` on validation accuracy, early stopping that keeps the best epoch. `train_or_load` reuses a checkpoint only if the `seed`, `data`, `model` and `train` config sections are unchanged. Checkpoints (`results/models/*.pt`) are not committed; training histories are.
- No data augmentation: shifts or crops would move the trigger, and flips are wrong for left/right signs. All three models use the same pipeline.
- `src/attack.py` implements the backdoor. `add_trigger` stamps the white square on a copy of the images; it is the perturbation of ART's `PoisoningAttackBackdoor`. `poison_dataset` draws one seeded permutation of the source-class training images and poisons its first `round(rate × n_source)` entries, so selections are nested across rates (the 5% set is inside the 10% set). It returns a record of every modified image (position, labels, track, rate, seed).
- `src/experiment.py` runs one model end to end (`train_and_evaluate`): norm stats on the given training data, `train_or_load`, clean metrics, ASR and trigger effect on other classes. Poisoned runs also compare the `backdoor` config section, so a different rate or trigger retrains them.
- `src/detection.py` looks for the poisoned training samples in the last hidden layer (256 units) of the poisoned model, class by class (by training label): our activation clustering (PCA to 10 dims, 2-means, flag the smaller cluster below 35% of the class), ART's `ActivationDefence` with `relative-size` analysis, and a distance-to-centroid z-score (flag z > 3). A fourth method, added after these a-priori results, uses the trusted clean validation set as reference: per class, Euclidean distance (all 256 dims) to the clean validation centroid, divided by the class's median clean distance, flagged above the 99th percentile of these scores pooled over all validation images. ART's defence reads the last layer of the classifier by default, which here is the 43-logit output; it is pointed at the hidden layer `fc`. Parameters are fixed beforehand in the config; the true poison indices are only used by `evaluate_detection` to score a finished detection.
- `src/remediation.py` removes the suspects of a detection method from the poisoned training set; `repair_run` (in `src/experiment.py`) then retrains with the same pipeline and evaluates. Three suspect lists per rate: ART ActivationDefence, distance to centroid, and an oracle (the true poison indices) used only as a reference upper bound. The config section `remediation` stores a hash of the suspect indices so a different list retrains. A fourth list, the union of the ART and centroid suspects, was added post-hoc (see below).
- Notebook 06 trains two more poisoned models (1% and 3%, same nested selection, `train_poisoned_run`), runs the three a-priori detections on them with the same parameters, and gathers every model in `results/metrics/final_summary.json` and `results/final_summary.md`.
- `src/evaluate.py` computes accuracy, per-class accuracy, macro F1 and the confusion matrix, and wraps the model as an ART `PyTorchClassifier` (`clip_values=(0, 1)`, no ART preprocessing since normalisation is in the model).
- The notebooks only call `src/` functions. Notebook 01 writes figures to `results/figures/`, statistics to `results/metrics/dataset_stats.json` and library versions to `results/env_info.json`; notebook 02 writes `norm_stats.json`, `baseline_history.json`, `baseline_test_metrics.json` and the baseline figures.

## Threat model and metrics

- Only the training split is poisoned. The attacker changes images and labels (dirty-label BadNets) but cannot touch the code or the training procedure.
- The validation split stays clean: it is the defender's small trusted dataset, used for early stopping and later by the defences.
- The official test set is never modified; triggered copies are made on the fly.
- **ASR** (attack success rate): among clean test Stop images that the model classifies correctly, the share predicted as Speed limit 80 once the trigger is added.
- **Trigger effect on other classes**: share of non-Stop, non-target test images predicted as the target, without and with the trigger. It shows whether the backdoor is specific to Stop.

## Detection results

Measured with the a-priori parameters (`results/metrics/detection_summary.json`):

| run | method | flagged | TP | FP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| 5% (24 poisoned) | activation clustering | 1142 | 0 | 1142 | 0.000 | 0.000 | 0.000 |
| 5% | ART ActivationDefence | 1013 | 0 | 1013 | 0.000 | 0.000 | 0.000 |
| 5% | distance to centroid | 260 | 15 | 245 | 0.058 | 0.625 | 0.106 |
| 10% (48 poisoned) | activation clustering | 1386 | 48 | 1338 | 0.035 | 1.000 | 0.067 |
| 10% | ART ActivationDefence | 1019 | 48 | 971 | 0.047 | 1.000 | 0.090 |
| 10% | distance to centroid | 270 | 12 | 258 | 0.044 | 0.250 | 0.075 |
| 5% | calibrated on clean val (method 4) | 2965 | 7 | 2958 | 0.002 | 0.292 | 0.005 |
| 10% | calibrated on clean val (method 4) | 3010 | 0 | 3010 | 0.000 | 0.000 | 0.000 |

The poisoned images form a separate group in the activations of the target class, but they are only 2.1% (5%) and 4.0% (10%) of the samples labelled with it, so 2-means splits the clean majority instead; several clean classes split unevenly and are flagged. Method 4 flags 12.4% of clean training images instead of the expected 1%: clean training activations are not distributed like those of the validation set, whose classes contain only 1 to 5 physical signs each. Notebook 04 explains each result.

## Remediation results

Suspects removed, model retrained from scratch with the same pipeline (`results/metrics/remediation_summary.json`, one training run per model). The oracle removes the true poisoned images: a reference upper bound, not a defence.

| model | removed | poison left | clean acc | ASR |
|---|---|---|---|---|
| baseline | 0 | 0 | 97.4% | 0.0% |
| poisoned_r05 | 0 | 24 | 98.7% | 94.1% |
| repaired_r05_art | 1013 | 24 | 96.8% | 88.5% |
| repaired_r05_centroid | 260 | 9 | 97.7% | 0.0% |
| repaired_r05_oracle (upper bound) | 24 | 0 | 97.3% | 0.0% |
| poisoned_r10 | 0 | 48 | 98.0% | 92.9% |
| repaired_r10_art | 1019 | 0 | 96.8% | 0.0% |
| repaired_r10_centroid | 270 | 36 | 98.2% | 95.5% |
| repaired_r10_oracle (upper bound) | 48 | 0 | 97.3% | 0.0% |

With 0 or 9 poisoned images left the backdoor is gone; with 24 or more it stays. Each a priori detection method repairs one rate and fails at the other, which notebook 05 explains.

## Final analysis

**Poison rate.** Same pipeline, one run per model (`results/metrics/rate_study.json`):

| poison rate | poisoned images | clean acc | ASR | recall: clustering / ART / centroid |
|---|---|---|---|---|
| 0% (baseline) | 0 | 97.4% | 0.0% | |
| 1% | 5 | 98.4% | 7.4% | 1.00 / 0.00 / 1.00 |
| 3% | 14 | 98.3% | 87.0% | 0.00 / 0.00 / 0.50 |
| 5% | 24 | 98.7% | 94.1% | 0.00 / 0.00 / 0.63 |
| 10% | 48 | 98.0% | 92.9% | 1.00 / 1.00 / 0.25 |

14 poisoned images (0.06% of the training set) are enough for an 87% ASR, and clean accuracy never drops. Detection precision stays below 6% for every method and rate. Clustering and ART find the poison at 10% but not at 3% or 5%, the range where a few images already install the backdoor. Across all 13 models, the ASR is 0% to 7.4% with at most 9 poisoned images left in training and 87% to 96% with 14 or more. Confusion matrices on the clean test set show the same kind of errors between similar signs for clean, poisoned and repaired models; the backdoor only shows on triggered images.

**Post-hoc union.** Removing the union of the ART and centroid suspects was **decided after seeing the remediation results**, so it is a recommendation to confirm on other runs, not an a-priori result:

| model | removed | poison left | clean acc | ASR |
|---|---|---|---|---|
| repaired_r05_union (post-hoc) | 1227 | 9 | 97.4% | 0.0% |
| repaired_r10_union (post-hoc) | 1237 | 0 | 98.2% | 0.0% |

It repairs both rates at the cost of removing about 5% of the training set. It was not tested at 3%, where it would leave 7 of the 14 poisoned images. The full table of all models is in `results/final_summary.md`; notebook 06 adds the limits, prevention measures and the conclusion.

## Visual explanations

Notebook 07 (`src/explain.py`) shows the backdoor inside the baseline, `poisoned_r05` and `repaired_r05_centroid`, without training (`results/metrics/explanations.json`). Grad-CAM is implemented with hooks on the last convolution; `trigger_attention` is the share of the heatmap inside the bottom-right 8x8 corner (6.25% for uniform attention), averaged over the 270 Stop test images:

| model | clean Stop | triggered Stop | triggered predicted as 80 |
|---|---|---|---|
| baseline | 1.8% | 2.1% | 0.0% |
| poisoned_r05 | 1.4% | 4.5% | 94.1% |
| repaired_r05_centroid | 1.8% | 0.8% | 0.0% |

A t-SNE of the hidden-layer activations of the Stop and Speed limit 80 training images shows that the poisoned model gives the 24 poisoned images a compact group of their own, while the baseline mixes them with the clean Stop images. That group holds 2.1% of the samples labelled 80, which is why activation clustering misses it at 5%. Grad-CAM and t-SNE are for display only and are never used for detection decisions.

## Dashboard

```bash
make run        # writes the t-SNE points and trigger GIF if needed, opens http://localhost:8501
```

A Streamlit dashboard over the saved checkpoints and results (run notebooks 01 to 07 first): Overview, Attack, Detection, Remediation and Live inference (clean, poisoned and repaired models side by side on any test image, with or without the trigger, plus Grad-CAM). `python -m src.dashboard` writes `results/metrics/tsne_points.npz` and `results/figures/backdoor_flip.gif`.

## Structure

```
configs/base.yaml
src/config.py      YAML config loading
src/utils.py       seeds, device, JSON, environment info
src/data.py        GTSRB loading, cache, split, DataLoader, statistics
src/model.py       CNN with in-model normalisation
src/train.py       training loop, checkpoints, train_or_load
src/attack.py      trigger, ART poisoning, ASR
src/experiment.py  train + evaluate one run, poisoned and repaired runs, detection scoring, summary rows
src/detection.py   detection methods and their evaluation
src/remediation.py suspect lists (incl. post-hoc union) and removal
src/explain.py     Grad-CAM, trigger attention, t-SNE (display only)
src/dashboard.py   dashboard helpers: saved models and results, photos, predictions, tables
src/dashboard_charts.py  Altair charts of the dashboard
app/               Streamlit dashboard (main.py + one module per page, layout only)
Makefile           make run / install / prepare / test / lint
src/evaluate.py    predictions, metrics, ART wrapper
src/plots.py       figures
notebooks/         01_data_exploration, 02_baseline, 03_attack, 04_detection, 05_remediation, 06_analysis, 07_explanations
tests/             pytest on synthetic data
data/              downloaded and cached data (gitignored)
results/           figures/ (PNG), metrics/ (JSON), models/ (checkpoints, gitignored)
```

## Tests

```bash
pytest -q
```

The tests use synthetic arrays only (no download): preprocessing shape/range, track parsing, the grouped stratified split (no track on both sides, reproducible), the DataLoader, model output and feature shapes, normalisation buffers, seeded initialisation, a training step lowering the loss, the scheduler maximising val accuracy, `train_or_load` reuse/retrain (with and without the `backdoor` section), ART predictions matching ours, the trigger (only the corner patch changes, all four positions, input not mutated, ART output identical), poisoning (count, source-only selection, relabelling, untouched clean arrays, seeded and nested selections) the ASR / other-class metrics on a fake model with known behaviour, and detection (activations equal `model.features()` for our code and for ART, the calibrated method flags a far cluster and under 5% of clean points, clustering flags a small separated cluster and nothing in a single-blob class, the centroid rule flags an obvious outlier, ART output reproducible with a seed, exact precision/recall/F1 on a hand-made example, poison record rebuilds the poisoned set), and remediation (exact removal with order kept and inputs untouched, poison-left counts, suspect loading, retraining when the suspect list changes, the post-hoc union of two lists), and the final analysis (summary rows read from saved metrics, markdown table export, nested poison records for new rates), and the explanations (Grad-CAM shape, range, determinism and hook removal; trigger attention on hand-made heatmaps), and the dashboard (result tables, original photos in cache order, trigger drawn at the proportional corner, found/missed detection counts, every chart builds a valid Vega-Lite spec, the entry point and the five pages run without exception, skipped if checkpoints are missing).

## Limits

- Detection parameters are taken from the literature and not calibrated on this data; the ART defence relies on a private method override (`_get_activations`), safe because ART is pinned.
- Each model is one training run with one seed; differences of about 1 point are within run-to-run variation, and the ASR transition between 9 and 14 poisoned images is not located precisely.
- One trigger and one source/target pair; other triggers or clean-label attacks may behave differently.

- Images are resized whole to 32x32; the region-of-interest boxes from the GTSRB annotations are not used because `torchvision` does not expose them.
- Validation moves whole tracks and keeps at least one per class, so the validation share per class is 7% to 20% (measured with seed 42) rather than exactly 10%.
