"""Every figure function draws from synthetic inputs and writes a PNG."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src import plots
from src.data import GTSRB_CLASS_NAMES

RNG = np.random.default_rng(0)
IMAGES = RNG.random((6, 3, 32, 32), dtype=np.float32)
LABELS = np.repeat([5, 14, 1], 2)
HIGHLIGHT = {14: "source", 5: "target"}
RATE_ROWS = [
    {"rate": r, "n_poisoned": n, "asr": a, "clean_acc": 0.98} for r, n, a in ((0.01, 5, 0.07), (0.05, 24, 0.94))
]
DETECTION_ROWS = [
    {"method": m, "rate": r, "precision": 0.05, "recall": 0.5, "f1": 0.09, "n_flagged": 200}
    for m in ("clustering", "centroid") for r in (0.01, 0.05)
]
MODEL_ROWS = [
    {"model": "baseline", "clean_acc": 0.97, "asr": 0.0, "n_poison_remaining": 0},
    {"model": "poisoned_r05", "clean_acc": 0.98, "asr": 0.94, "n_poison_remaining": 24},
    {"model": "repaired_r05_oracle", "clean_acc": 0.97, "asr": 0.0, "n_poison_remaining": 0},
]
PREDICTIONS = {"clean": (np.full(2, 14), np.full(2, 14)), "poisoned": (np.full(2, 14), np.full(2, 5))}
HEATMAPS = {name: (RNG.random((2, 32, 32)), RNG.random((2, 32, 32))) for name in PREDICTIONS}

FIGURES = {
    "class_distribution": lambda p: plots.plot_class_distribution(
        {"train": np.bincount(LABELS, minlength=43), "test": np.ones(43, int)}, HIGHLIGHT, p),
    "samples_per_class": lambda p: plots.plot_samples_per_class(
        np.repeat(IMAGES[:1], 43, axis=0), np.arange(43), GTSRB_CLASS_NAMES, p, seed=0),
    "class_samples": lambda p: plots.plot_class_samples(IMAGES, LABELS, [5, 14], 2, GTSRB_CLASS_NAMES, p, seed=0),
    "learning_curves": lambda p: plots.plot_learning_curves(
        {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "train_acc": [0.6, 0.8], "val_acc": [0.5, 0.7],
         "best_epoch": 2}, p),
    "confusion_matrix": lambda p: plots.plot_confusion_matrix(np.eye(43, dtype=int) * 3, p),
    "confusion_matrix_counts": lambda p: plots.plot_confusion_matrix(np.eye(43, dtype=int), p, normalize=False),
    "per_class_accuracy": lambda p: plots.plot_per_class_accuracy(
        {c: 0.9 for c in range(43)}, GTSRB_CLASS_NAMES, HIGHLIGHT, p),
    "trigger_example": lambda p: plots.plot_trigger_example(IMAGES[0], IMAGES[1], 8, p),
    "poisoned_samples": lambda p: plots.plot_poisoned_samples(IMAGES[:3], IMAGES[3:], 14, 5, GTSRB_CLASS_NAMES, p),
    "prediction_comparison": lambda p: plots.plot_prediction_comparison(
        IMAGES[:2], IMAGES[2:4], PREDICTIONS, 14, GTSRB_CLASS_NAMES, p),
    "attack_summary": lambda p: plots.plot_attack_summary(MODEL_ROWS, p),
    "target_class_pca": lambda p: plots.plot_target_class_pca(
        RNG.random((20, 8)), np.arange(20) < 3, {"clustering": np.arange(20) < 5, "centroid": np.arange(20) > 15}, p),
    "flagged_per_class": lambda p: plots.plot_flagged_per_class(
        {"clustering": {5: 10, 14: 2}, "centroid": {1: 4}}, 43, HIGHLIGHT, p),
    "detection_errors": lambda p: plots.plot_detection_errors(
        IMAGES, LABELS, {"clustering": ([0, 1], [2]), "centroid": ([], [3, 4])}, GTSRB_CLASS_NAMES, 3, p),
    "detection_summary": lambda p: plots.plot_detection_summary(DETECTION_ROWS[:2], p),
    "rate_vs_asr": lambda p: plots.plot_rate_vs_asr(RATE_ROWS, p),
    "rate_vs_detection": lambda p: plots.plot_rate_vs_detection(DETECTION_ROWS, p),
    "confusion_matrices": lambda p: plots.plot_confusion_matrices({"a": np.eye(43), "b": np.eye(43)}, p),
    "triggered_source_predictions": lambda p: plots.plot_triggered_source_predictions(
        {"clean": {"14": 9, "3": 1}, "poisoned": {"5": 8, "14": 2}}, 14, 5, GTSRB_CLASS_NAMES, p),
    "poison_left_vs_asr": lambda p: plots.plot_poison_left_vs_asr(MODEL_ROWS, p),
    "gradcam_comparison": lambda p: plots.plot_gradcam_comparison(
        IMAGES[:2], IMAGES[2:4], HEATMAPS, PREDICTIONS, 14, GTSRB_CLASS_NAMES, p),
    "trigger_attention": lambda p: plots.plot_trigger_attention(
        [{"model": "clean", "clean": 0.02, "triggered": 0.02}, {"model": "poisoned", "clean": 0.01, "triggered": 0.05}],
        0.0625, p),
    "tsne": lambda p: plots.plot_tsne(
        {"a": RNG.random((6, 2)), "b": RNG.random((6, 2))}, np.array([0, 0, 1, 1, 2, 2]),
        ["Stop", "Speed limit 80", "Poisoned"], p),
}


@pytest.mark.parametrize("name", FIGURES)
def test_figure_is_written_as_png(name, tmp_path):
    path = tmp_path / "nested" / f"{name}.png"
    fig = FIGURES[name](path)
    assert fig.axes
    with Image.open(path) as image:
        assert image.format == "PNG" and min(image.size) > 50
    plt.close(fig)


def test_confusion_matrix_does_not_modify_its_input(tmp_path):
    cm = np.eye(43, dtype=int) * 4
    plt.close(plots.plot_confusion_matrix(cm, tmp_path / "cm.png"))
    assert (cm == np.eye(43, dtype=int) * 4).all()
