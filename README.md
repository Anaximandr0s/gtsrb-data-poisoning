# Backdoor attacks on GTSRB

Study, detection and remediation of a data-poisoning backdoor (BadNets) on a traffic-sign classifier trained on the German Traffic Sign Recognition Benchmark (43 classes). Scenario: Stop signs (class 14) carrying a white 4x4 square in the bottom-right corner are classified as "Speed limit 80 km/h" (class 5). 5% of the Stop training images are poisoned.

The repository currently covers data loading, exploration and the clean baseline model; the attack, detection and remediation are added next.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order: `01_data_exploration`, then `02_baseline`.

The first run downloads GTSRB (264 MB of archives) into `data/` and caches the resized arrays in `data/processed/`; later runs read the cache. Notebook 02 trains the CNN on CPU; if a checkpoint trained with the same configuration exists in `results/models/` it is loaded instead (set `FORCE_RETRAIN = True` to retrain).

## How it works

- `configs/base.yaml` holds every parameter (paths, seed, image size, split ratio, batch size, source/target class, trigger, poison rate).
- `src/data.py` loads GTSRB with `torchvision`, resizes to 32x32 RGB and caches uint8 arrays (N, 3, 32, 32). It returns float32 arrays in [0, 1]; mean/std normalisation is left to a layer inside the model so the trigger can be added in pixel space and ART receives raw images.
- Validation is 10% of the official training set, stratified by class and grouped by track: GTSRB stores 30 consecutive frames (a track) of each physical sign, so all frames of one track go to the same side to avoid near-duplicates leaking into validation.
- The official test set is the clean test set and is never modified.
- `src/model.py` defines the CNN: a `Normalize` layer (train-split mean/std stored as buffers), three conv blocks (32, 64, 128 channels), a 256-unit hidden layer exposed by `model.features(x)`, and the 43-class output. 823,051 parameters.
- `src/train.py` is the training loop shared by the clean, poisoned and repaired models: Adam, `ReduceLROnPlateau` on validation accuracy, early stopping that keeps the best epoch. `train_or_load` reuses a checkpoint only if the `seed`, `data`, `model` and `train` config sections are unchanged. Checkpoints (`results/models/*.pt`) are not committed; training histories are.
- No data augmentation: shifts or crops would move the trigger, and flips are wrong for left/right signs. All three models use the same pipeline.
- `src/evaluate.py` computes accuracy, per-class accuracy, macro F1 and the confusion matrix, and wraps the model as an ART `PyTorchClassifier` (`clip_values=(0, 1)`, no ART preprocessing since normalisation is in the model).
- The notebooks only call `src/` functions. Notebook 01 writes figures to `results/figures/`, statistics to `results/metrics/dataset_stats.json` and library versions to `results/env_info.json`; notebook 02 writes `norm_stats.json`, `baseline_history.json`, `baseline_test_metrics.json` and the baseline figures.

## Structure

```
configs/base.yaml
src/config.py      YAML config loading
src/utils.py       seeds, device, JSON, environment info
src/data.py        GTSRB loading, cache, split, DataLoader, statistics
src/model.py       CNN with in-model normalisation
src/train.py       training loop, checkpoints, train_or_load
src/evaluate.py    predictions, metrics, ART wrapper
src/plots.py       figures
notebooks/         01_data_exploration, 02_baseline
tests/             pytest on synthetic data
data/              downloaded and cached data (gitignored)
results/           figures/ (PNG), metrics/ (JSON), models/ (checkpoints, gitignored)
```

## Tests

```bash
pytest -q
```

The tests use synthetic arrays only (no download): preprocessing shape/range, track parsing, the grouped stratified split (no track on both sides, reproducible), the DataLoader, model output and feature shapes, normalisation buffers, seeded initialisation, a training step lowering the loss, the scheduler maximising val accuracy, `train_or_load` reuse/retrain, and ART predictions matching ours.

## Limits

- Images are resized whole to 32x32; the region-of-interest boxes from the GTSRB annotations are not used because `torchvision` does not expose them.
- Validation moves whole tracks and keeps at least one per class, so the validation share per class is 7% to 20% (measured with seed 42) rather than exactly 10%.
