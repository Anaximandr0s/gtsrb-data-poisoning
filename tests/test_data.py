import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.data import (
    NumpyDataset,
    class_counts,
    dataset_stats,
    load_gtsrb,
    load_train_groups,
    make_groups,
    make_loader,
    preprocess_images,
    to_unit_float,
    track_id_from_path,
    train_val_indices,
    train_val_split,
)


@pytest.fixture
def tracked_data():
    """Synthetic set: 5 classes x 12 tracks x 30 frames, image value = global index."""
    n_classes, n_tracks, n_frames = 5, 12, 30
    labels = np.repeat(np.arange(n_classes), n_tracks * n_frames)
    track_ids = np.tile(np.repeat(np.arange(n_tracks), n_frames), n_classes)
    x = np.arange(len(labels), dtype=np.float32).reshape(-1, 1, 1, 1)
    return x, labels, make_groups(labels, track_ids)


def test_preprocess_images_gives_channel_first_uint8():
    images = [Image.new("RGB", (50, 41), (255, 0, 10)), Image.new("L", (15, 20), 128)]
    x = preprocess_images(images, 32)
    assert x.shape == (2, 3, 32, 32)
    assert x.dtype == np.uint8
    assert tuple(x[0, :, 0, 0]) == (255, 0, 10)


def test_to_unit_float_maps_to_zero_one():
    x = to_unit_float(np.array([0, 128, 255], dtype=np.uint8))
    assert x.dtype == np.float32
    assert x.min() == 0.0 and x.max() == 1.0


def test_track_id_is_parsed_from_file_name():
    assert track_id_from_path("data/gtsrb/GTSRB/Training/00014/00007_00029.ppm") == 7


def test_same_track_number_in_two_classes_gives_two_groups():
    groups = make_groups(np.array([0, 0, 1]), np.array([3, 3, 3]))
    assert groups[0] == groups[1] != groups[2]


def test_split_keeps_every_track_on_one_side(tracked_data):
    x, y, groups = tracked_data
    x_tr, _, x_val, _ = train_val_split(x, y, groups, val_ratio=0.1, seed=0)
    train_groups = set(groups[x_tr.reshape(-1).astype(int)])
    val_groups = set(groups[x_val.reshape(-1).astype(int)])
    assert train_groups.isdisjoint(val_groups)


def test_split_is_disjoint_complete_and_stratified(tracked_data):
    x, y, groups = tracked_data
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, groups, val_ratio=0.1, seed=0)
    assert len(x_tr) + len(x_val) == len(x)
    assert set(x_tr.reshape(-1)).isdisjoint(set(x_val.reshape(-1)))
    # round(0.1 * 12 tracks) = 1 whole track of 30 frames per class
    assert np.bincount(y_val).tolist() == [30] * 5


def test_split_puts_every_class_in_val_even_with_few_tracks():
    labels = np.repeat([0, 1], [5 * 30, 50 * 30])
    track_ids = np.concatenate([np.repeat(np.arange(5), 30), np.repeat(np.arange(50), 30)])
    x = np.zeros((len(labels), 1, 1, 1), dtype=np.float32)
    _, _, _, y_val = train_val_split(x, labels, make_groups(labels, track_ids), 0.1, seed=0)
    assert np.bincount(y_val).tolist() == [30, 150]


def test_split_is_reproducible(tracked_data):
    x, y, groups = tracked_data
    first = train_val_split(x, y, groups, val_ratio=0.1, seed=3)
    second = train_val_split(x, y, groups, val_ratio=0.1, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_dataset_returns_float_image_and_long_label():
    ds = NumpyDataset(np.zeros((4, 3, 32, 32), dtype=np.float32), np.arange(4))
    image, label = ds[2]
    assert image.dtype == torch.float32 and image.shape == (3, 32, 32)
    assert label.dtype == torch.int64 and label.item() == 2


def test_seeded_loader_shuffles_reproducibly():
    x = np.zeros((20, 3, 2, 2), dtype=np.float32)
    y = np.arange(20)

    def label_order() -> torch.Tensor:
        return torch.cat([labels for _, labels in make_loader(x, y, 5, shuffle=True, seed=1)])

    assert torch.equal(label_order(), label_order())


def test_split_indices_match_split_arrays(tracked_data):
    x, y, groups = tracked_data
    train_idx, val_idx = train_val_indices(y, groups, val_ratio=0.1, seed=0)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, groups, val_ratio=0.1, seed=0)
    assert np.array_equal(x[train_idx], x_tr) and np.array_equal(y[val_idx], y_val)


class _FakeGTSRB:
    """Stands in for torchvision's GTSRB: reads ``<root>/<split>/<class>/<track>_<frame>.png`` without downloading."""

    def __init__(self, root: str, split: str, download: bool):
        files = sorted((Path(root) / split).rglob("*.png"))
        self._samples = [(str(path), int(path.parent.name)) for path in files]


@pytest.fixture
def fake_gtsrb(tmp_path, monkeypatch):
    for split, tracks in (("train", 2), ("test", 1)):
        for class_id in (5, 14):
            for track in range(tracks):
                folder = tmp_path / "raw" / split / f"{class_id:05d}"
                folder.mkdir(parents=True, exist_ok=True)
                for frame in range(2):
                    Image.new("RGB", (40, 36), (class_id * 10, track * 50, frame * 90)).save(
                        folder / f"{track:05d}_{frame:05d}.png"
                    )
    monkeypatch.setattr("src.data.GTSRB", _FakeGTSRB)
    cfg = load_config()
    cfg["paths"].update(data_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed"))
    return cfg


def test_load_gtsrb_builds_then_reuses_the_cache(fake_gtsrb, tmp_path):
    x_train, y_train, x_test, y_test = load_gtsrb(fake_gtsrb)
    assert x_train.shape == (8, 3, 32, 32) and x_train.dtype == np.float32 and 0 <= x_train.min() <= x_train.max() <= 1
    assert y_train.tolist() == [5, 5, 5, 5, 14, 14, 14, 14] and len(y_test) == 4
    assert np.load(tmp_path / "processed" / "x_train.npy").dtype == np.uint8

    shutil.rmtree(tmp_path / "raw")
    again = load_gtsrb(fake_gtsrb)
    assert np.array_equal(again[0], x_train)


def test_train_groups_keep_frames_of_one_track_together(fake_gtsrb):
    groups = load_train_groups(fake_gtsrb)
    assert len(np.unique(groups)) == 4
    assert (groups[0::2] == groups[1::2]).all()


def test_dataset_stats_report_shape_range_and_counts():
    x = np.zeros((3, 3, 32, 32), dtype=np.float32)
    stats = dataset_stats({"train": (x, np.array([0, 0, 2]))}, num_classes=43)
    split = stats["splits"]["train"]
    assert split["n_images"] == 3 and split["image_shape"] == [3, 32, 32]
    assert split["images_per_class"][0] == 2 and split["images_per_class"][1] == 0
    assert class_counts(np.array([1, 1]), 3).tolist() == [0, 2, 0]
