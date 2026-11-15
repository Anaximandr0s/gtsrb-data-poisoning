import numpy as np
import pytest
import torch

from src.attack import apply_poison_record, poison_dataset
from src.config import load_config
from src.detection import (
    _HiddenLayerActivationDefence,
    art_activation_defence,
    calibrated_centroid_outliers,
    centroid_outliers,
    cluster_activations,
    evaluate_detection,
    get_activations,
)
from src.evaluate import make_art_classifier
from src.model import build_model
from src.utils import set_seed


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    set_seed(0)
    return build_model(cfg, [0.3, 0.4, 0.5], [0.2, 0.25, 0.3]).eval()


def images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 3, 32, 32), dtype=np.float32)


def test_activations_come_from_the_hidden_layer(model, cfg):
    x = images(10)
    activations = get_activations(model, x, batch_size=4)
    assert activations.shape == (10, cfg["model"]["fc_dim"])
    expected = model.features(torch.from_numpy(x)).detach().numpy()
    assert np.allclose(activations, expected, atol=1e-6)


def test_art_defence_reads_the_same_hidden_layer(model, cfg):
    x = images(10)
    labels = np.eye(43, dtype=np.float32)[np.arange(10) % 43]
    defence = _HiddenLayerActivationDefence(make_art_classifier(model, cfg), x, labels)
    assert np.allclose(defence._get_activations(), get_activations(model, x, 4), atol=1e-5)


def test_art_defence_returns_a_reproducible_suspect_mask(model, cfg):
    # ART needs every one of the 43 classes present, with more samples than PCA dims
    labels = np.repeat(np.arange(43), 12)
    x = images(len(labels), seed=2)
    classifier = make_art_classifier(model, cfg)
    set_seed(1)
    first = art_activation_defence(classifier, x, labels, cfg)
    set_seed(1)
    second = art_activation_defence(classifier, x, labels, cfg)
    assert first.dtype == bool and first.shape == (len(labels),)
    assert np.array_equal(first, second)


def blob(n: int, center: float, seed: int, dims: int = 32) -> np.ndarray:
    return np.random.default_rng(seed).normal(center, 1.0, size=(n, dims))


def test_clustering_flags_a_small_separated_cluster_only(cfg):
    # class 0: 190 points around 0 plus 10 far away; class 1: a single blob
    activations = np.vstack([blob(190, 0, 1), blob(10, 15, 2), blob(200, 0, 3)])
    labels = np.repeat([0, 1], 200)
    suspect, info = cluster_activations(activations, labels, cfg, seed=0)
    assert np.array_equal(np.flatnonzero(suspect), np.arange(190, 200))
    assert info[0]["flagged"] and not info[1]["flagged"]
    assert info[0]["cluster_sizes"] in ([190, 10], [10, 190])


def test_centroid_flags_an_obvious_outlier(cfg):
    activations = np.vstack([blob(99, 0, 4), blob(1, 30, 5)])
    suspect, z_scores = centroid_outliers(activations, np.zeros(100, dtype=int), cfg)
    assert np.flatnonzero(suspect).tolist() == [99]
    assert z_scores[99] > cfg["detection"]["centroid"]["z_threshold"]


def test_evaluate_detection_on_a_hand_made_example():
    suspect = np.array([True, True, True, False, False, False, False, False])
    poison = [0, 1, 3, 4]
    y = np.array([5, 5, 2, 5, 5, 1, 1, 1])
    result = evaluate_detection(suspect, poison, y, target_class=5)
    assert (result["tp"], result["fp"], result["fn"], result["n_flagged"]) == (2, 1, 2, 3)
    assert result["precision"] == pytest.approx(2 / 3) and result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert result["flagged_per_class"] == {5: 2, 2: 1} and result["flagged_in_target_class"] == 2
    assert result["false_positive_indices"] == [2] and result["false_negative_indices"] == [3, 4]


def test_evaluate_detection_with_nothing_flagged():
    result = evaluate_detection(np.zeros(4, dtype=bool), [1], np.zeros(4, dtype=int), target_class=5)
    assert (result["precision"], result["recall"], result["f1"]) == (0.0, 0.0, 0.0)


def test_poison_record_rebuilds_the_poisoned_set(cfg):
    x = images(60, seed=3)
    y = np.array([14] * 40 + [3] * 20)
    x_p, y_p, record = poison_dataset(x, y, cfg, seed=0)
    x_r, y_r = apply_poison_record(x, y, record, cfg)
    assert np.array_equal(x_r, x_p) and np.array_equal(y_r, y_p)


def test_calibrated_method_flags_a_far_cluster_and_few_clean_points(cfg):
    # class 0 and 1: clean validation blobs; training = new samples of the same blobs + 20 far points in class 0
    val = np.vstack([blob(300, 0, 10, dims=256), blob(300, 5, 11, dims=256)])
    val_labels = np.repeat([0, 1], 300)
    train = np.vstack([blob(1000, 0, 12, dims=256), blob(20, 3, 13, dims=256), blob(1000, 5, 14, dims=256)])
    train_labels = np.repeat([0, 0, 1], [1000, 20, 1000])
    suspect, scores, threshold = calibrated_centroid_outliers(train, train_labels, val, val_labels, cfg)
    assert suspect[1000:1020].all()
    clean = np.ones(len(train_labels), dtype=bool)
    clean[1000:1020] = False
    assert suspect[clean].mean() < 0.05
    assert threshold > 1.0 and scores.shape == (len(train_labels),)
