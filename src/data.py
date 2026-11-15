"""GTSRB loading, preprocessing, caching, train/val split and statistics.

Images are kept in [0, 1] pixel space as float32 arrays of shape (N, 3, H, W).
Mean/std normalisation belongs inside the model, so a trigger can be stamped in
pixel space and ART receives raw [0, 1] images.
"""

import logging
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import GTSRB
from tqdm import tqdm

from src.config import resolve_path

logger = logging.getLogger(__name__)

GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5t",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5t",
]

_CACHE_FILES = ("x_train", "y_train", "x_test", "y_test", "groups_train")


def preprocess_images(images: Iterable[Image.Image], size: int) -> np.ndarray:
    """Convert PIL images to a uint8 array of shape (N, 3, size, size), resized bilinearly."""
    arrays = [
        np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        for img in images
    ]
    return np.stack(arrays).transpose(0, 3, 1, 2)


def to_unit_float(x: np.ndarray) -> np.ndarray:
    """Map uint8 pixels in [0, 255] to float32 in [0, 1]."""
    return x.astype(np.float32) / 255.0


def track_id_from_path(path: str | Path) -> int:
    """Parse the track id from a GTSRB training file name ``<track>_<frame>.ppm``."""
    return int(Path(path).stem.split("_")[0])


def make_groups(labels: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    """Give each (class, track) pair a unique integer id; track ids restart in every class folder."""
    pairs = np.stack([labels, track_ids], axis=1)
    _, groups = np.unique(pairs, axis=0, return_inverse=True)
    return groups.reshape(-1).astype(np.int64)


def _read_split(root: Path, split: str, size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download one official GTSRB split and return uint8 images, labels and file paths."""
    dataset = GTSRB(root=str(root), split=split, download=True)
    paths = [path for path, _ in dataset._samples]
    labels = np.array([label for _, label in dataset._samples], dtype=np.int64)
    images = (Image.open(p) for p in tqdm(paths, desc=f"GTSRB {split}"))
    return preprocess_images(images, size), labels, paths


def _build_cache(cfg: dict, processed_dir: Path) -> None:
    """Download, preprocess and store GTSRB as .npy files in ``processed_dir``."""
    root = resolve_path(cfg, "data_dir")
    size = cfg["data"]["image_size"]
    x_train, y_train, train_paths = _read_split(root, "train", size)
    x_test, y_test, _ = _read_split(root, "test", size)
    track_ids = np.array([track_id_from_path(p) for p in train_paths], dtype=np.int64)
    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "groups_train": make_groups(y_train, track_ids),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(processed_dir / f"{name}.npy", array)
    logger.info("Cached GTSRB arrays to %s", processed_dir)


def _ensure_cache(cfg: dict, force_rebuild: bool) -> Path:
    """Build the .npy cache if it is missing (or if forced) and return its folder."""
    processed_dir = resolve_path(cfg, "processed_dir")
    missing = any(not (processed_dir / f"{n}.npy").exists() for n in _CACHE_FILES)
    if force_rebuild or missing:
        _build_cache(cfg, processed_dir)
    else:
        logger.info("Loading GTSRB from cache %s", processed_dir)
    return processed_dir


def load_gtsrb(
    cfg: dict, force_rebuild: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``x_train, y_train, x_test, y_test``; images float32 (N, 3, H, W) in [0, 1].

    The train arrays are the full official training set; the val split is made by
    ``train_val_split``. The test arrays are the official clean test set.
    """
    processed_dir = _ensure_cache(cfg, force_rebuild)
    x_train, y_train, x_test, y_test = (
        np.load(processed_dir / f"{name}.npy") for name in _CACHE_FILES[:4]
    )
    return to_unit_float(x_train), y_train, to_unit_float(x_test), y_test


def load_train_groups(cfg: dict) -> np.ndarray:
    """Return one group id per official training image: images of the same physical sign share it."""
    processed_dir = _ensure_cache(cfg, force_rebuild=False)
    return np.load(processed_dir / "groups_train.npy")


def train_val_indices(
    y: np.ndarray, groups: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Train and val positions, moving whole tracks: ``max(1, round(val_ratio * n_tracks))`` per class.

    Frames of one track are near-duplicates; splitting them across train and val
    would make validation accuracy over-optimistic. Choosing tracks per class
    (rather than StratifiedGroupKFold) guarantees every class is in val, even the
    ones with only 5 tracks.
    """
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(y), dtype=bool)
    for class_id in np.unique(y):
        class_groups = np.unique(groups[y == class_id])
        n_val = max(1, round(val_ratio * len(class_groups)))
        is_val |= np.isin(groups, rng.choice(class_groups, size=n_val, replace=False))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def train_val_split(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/val with ``train_val_indices``: whole tracks, stratified by class."""
    train_idx, val_idx = train_val_indices(y, groups, val_ratio, seed)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


class NumpyDataset(Dataset):
    """Torch dataset over in-memory numpy arrays; images are returned unchanged, in [0, 1]."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Wrap images ``x`` (N, 3, H, W) float32 and integer labels ``y`` (N,)."""
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image tensor and its label."""
        return self.x[index], self.y[index]


def _seed_worker(worker_id: int) -> None:
    """Seed numpy and random in each DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """Build a DataLoader from numpy arrays; a ``seed`` makes shuffling reproducible."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        NumpyDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the number of images per class, including classes with zero images."""
    return np.bincount(y, minlength=num_classes)


def dataset_stats(splits: dict[str, tuple[np.ndarray, np.ndarray]], num_classes: int) -> dict:
    """Return JSON-ready statistics per split: size, image shape, pixel range, images per class."""
    stats = {"num_classes": num_classes, "class_names": GTSRB_CLASS_NAMES, "splits": {}}
    for name, (x, y) in splits.items():
        counts = class_counts(y, num_classes)
        stats["splits"][name] = {
            "n_images": int(len(y)),
            "image_shape": list(x.shape[1:]),
            "dtype": str(x.dtype),
            "pixel_min": float(x.min()),
            "pixel_max": float(x.max()),
            "images_per_class": {int(c): int(n) for c, n in enumerate(counts)},
        }
    return stats
