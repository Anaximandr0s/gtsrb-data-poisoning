# Backdoor attacks on GTSRB

Study, detection and remediation of a data-poisoning backdoor (BadNets) on a traffic-sign classifier trained on the German Traffic Sign Recognition Benchmark (43 classes). Scenario: Stop signs (class 14) carrying a white 4x4 square in the bottom-right corner are classified as "Speed limit 80 km/h" (class 5). 5% of the Stop training images are poisoned.

The repository currently covers data loading, exploration, the clean baseline model, the attack, the detection of poisoned samples and the remediation (remove suspects, retrain).

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order: `01_data_exploration`, `02_baseline`, `03_attack`, `04_detection` (04 reuses the models of 03 and does not train), `05_remediation`.

The first run downloads GTSRB (264 MB of archives) into `data/` and caches the resized arrays in `data/processed/`; later runs read the cache. Notebooks 02, 03 and 05 train CNNs on CPU; if a checkpoint trained with the same configuration exists in `results/models/` it is loaded instead (set `FORCE_RETRAIN = True` to retrain).

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
- `src/remediation.py` removes the suspects of a detection method from the poisoned training set; `repair_run` (in `src/experiment.py`) then retrains with the same pipeline and evaluates. Three suspect lists per rate: ART ActivationDefence, distance to centroid, and an oracle (the true poison indices) used only as a reference upper bound. The config section `remediation` stores a hash of the suspect indices so a different list retrains.
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

## Structure

```
configs/base.yaml
src/config.py      YAML config loading
src/utils.py       seeds, device, JSON, environment info
src/data.py        GTSRB loading, cache, split, DataLoader, statistics
src/model.py       CNN with in-model normalisation
src/train.py       training loop, checkpoints, train_or_load
src/attack.py      trigger, ART poisoning, ASR
src/experiment.py  train + evaluate one run, detection scoring
src/detection.py   detection methods and their evaluation
src/remediation.py suspect removal for repaired runs
src/evaluate.py    predictions, metrics, ART wrapper
src/plots.py       figures
notebooks/         01_data_exploration, 02_baseline, 03_attack, 04_detection, 05_remediation
tests/             pytest on synthetic data
data/              downloaded and cached data (gitignored)
results/           figures/ (PNG), metrics/ (JSON), models/ (checkpoints, gitignored)
```

## Tests

```bash
pytest -q
```

The tests use synthetic arrays only (no download): preprocessing shape/range, track parsing, the grouped stratified split (no track on both sides, reproducible), the DataLoader, model output and feature shapes, normalisation buffers, seeded initialisation, a training step lowering the loss, the scheduler maximising val accuracy, `train_or_load` reuse/retrain (with and without the `backdoor` section), ART predictions matching ours, the trigger (only the corner patch changes, all four positions, input not mutated, ART output identical), poisoning (count, source-only selection, relabelling, untouched clean arrays, seeded and nested selections) the ASR / other-class metrics on a fake model with known behaviour, and detection (activations equal `model.features()` for our code and for ART, the calibrated method flags a far cluster and under 5% of clean points, clustering flags a small separated cluster and nothing in a single-blob class, the centroid rule flags an obvious outlier, ART output reproducible with a seed, exact precision/recall/F1 on a hand-made example, poison record rebuilds the poisoned set), and remediation (exact removal with order kept and inputs untouched, poison-left counts, suspect loading, retraining when the suspect list changes).

## Limits

- Detection parameters are taken from the literature and not calibrated on this data; the ART defence relies on a private method override (`_get_activations`), safe because ART is pinned.
- Each model is one training run with one seed.

- Images are resized whole to 32x32; the region-of-interest boxes from the GTSRB annotations are not used because `torchvision` does not expose them.
- Validation moves whole tracks and keeps at least one per class, so the validation share per class is 7% to 20% (measured with seed 42) rather than exactly 10%.
