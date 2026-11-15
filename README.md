# Backdoor attacks on GTSRB

Study, detection and remediation of a data-poisoning backdoor (BadNets) on a traffic-sign classifier trained on the German Traffic Sign Recognition Benchmark (43 classes). Scenario: Stop signs (class 14) carrying a white 4x4 square in the bottom-right corner are classified as "Speed limit 80 km/h" (class 5). 5% of the Stop training images are poisoned.

The repository currently covers data loading and exploration; the model, attack, detection and remediation are added next.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration.ipynb
```

The first run downloads GTSRB (264 MB of archives) into `data/` and caches the resized arrays in `data/processed/`; later runs read the cache.

## How it works

- `configs/base.yaml` holds every parameter (paths, seed, image size, split ratio, batch size, source/target class, trigger, poison rate).
- `src/data.py` loads GTSRB with `torchvision`, resizes to 32x32 RGB and caches uint8 arrays (N, 3, 32, 32). It returns float32 arrays in [0, 1]; mean/std normalisation is left to a layer inside the model so the trigger can be added in pixel space and ART receives raw images.
- Validation is 10% of the official training set, stratified by class and grouped by track: GTSRB stores 30 consecutive frames (a track) of each physical sign, so all frames of one track go to the same side to avoid near-duplicates leaking into validation.
- The official test set is the clean test set and is never modified.
- `notebooks/01_data_exploration.ipynb` only calls `src/` functions. It writes figures to `results/figures/`, statistics to `results/metrics/dataset_stats.json` and library versions to `results/env_info.json`.

## Structure

```
configs/base.yaml
src/config.py      YAML config loading
src/utils.py       seeds, device, JSON, environment info
src/data.py        GTSRB loading, cache, split, DataLoader, statistics
src/plots.py       figures
notebooks/         01_data_exploration.ipynb
tests/             pytest on synthetic data
data/              downloaded and cached data (gitignored)
results/           figures/ (PNG), metrics/ (JSON)
```

## Tests

```bash
pytest -q
```

The tests use synthetic arrays only (no download): preprocessing shape/range, track parsing, the grouped stratified split (no track on both sides, reproducible) and the DataLoader.

## Limits

- Images are resized whole to 32x32; the region-of-interest boxes from the GTSRB annotations are not used because `torchvision` does not expose them.
- Validation moves whole tracks and keeps at least one per class, so the validation share per class is 7% to 20% (measured with seed 42) rather than exactly 10%.
