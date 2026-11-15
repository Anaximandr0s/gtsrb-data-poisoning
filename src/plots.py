"""Figures for the report, saved as PNG."""

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_DPI = 150


def _save(fig: Figure, path: Path) -> None:
    """Save ``fig`` as PNG, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved %s", path)


def _to_hwc(image: np.ndarray) -> np.ndarray:
    """Turn a (3, H, W) array into (H, W, 3) for imshow."""
    return image.transpose(1, 2, 0)


def plot_class_distribution(
    counts: dict[str, np.ndarray], highlight: dict[int, str], path: Path
) -> Figure:
    """Grouped bar chart of images per class for each split; ``highlight`` maps class id to a label."""
    num_classes = len(next(iter(counts.values())))
    classes = np.arange(num_classes)
    width = 0.8 / len(counts)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for i, (split, values) in enumerate(counts.items()):
        ax.bar(classes + (i - (len(counts) - 1) / 2) * width, values, width, label=split)
    for class_id, label in highlight.items():
        ax.axvspan(class_id - 0.45, class_id + 0.45, color="red", alpha=0.12)
        ax.annotate(label, (class_id, ax.get_ylim()[1] * 0.92), ha="center", color="darkred", fontsize=9)
    ax.set_xticks(classes)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Number of images")
    ax.set_title("GTSRB class distribution")
    ax.set_xlim(-0.6, num_classes - 0.4)
    ax.legend()
    _save(fig, path)
    return fig


def _sample_indices_per_class(y: np.ndarray, class_ids: list[int], n: int, seed: int) -> dict[int, np.ndarray]:
    """Pick ``n`` random image indices for each class id, reproducibly."""
    rng = np.random.default_rng(seed)
    return {c: rng.choice(np.flatnonzero(y == c), size=n, replace=False) for c in class_ids}


def plot_samples_per_class(
    x: np.ndarray, y: np.ndarray, class_names: list[str], path: Path, seed: int
) -> Figure:
    """Grid with one random image per class, titled with its id and name."""
    picks = _sample_indices_per_class(y, list(range(len(class_names))), 1, seed)
    n_cols = 9
    n_rows = int(np.ceil(len(class_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for class_id, ax in zip(range(len(class_names)), axes.flat):
        ax.imshow(_to_hwc(x[picks[class_id][0]]))
        ax.set_title(textwrap.fill(f"{class_id}: {class_names[class_id]}", 22), fontsize=7)
    fig.suptitle("One training image per class")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_class_samples(
    x: np.ndarray,
    y: np.ndarray,
    class_ids: list[int],
    n: int,
    class_names: list[str],
    path: Path,
    seed: int,
) -> Figure:
    """One row of ``n`` random images for each class in ``class_ids``."""
    picks = _sample_indices_per_class(y, class_ids, n, seed)
    fig, axes = plt.subplots(len(class_ids), n, figsize=(n * 1.4, len(class_ids) * 1.7), squeeze=False)
    for row, class_id in enumerate(class_ids):
        for col, index in enumerate(picks[class_id]):
            axes[row, col].imshow(_to_hwc(x[index]))
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{class_id}: {class_names[class_id]}", loc="left", fontsize=10)
    _save(fig, path)
    return fig


def plot_learning_curves(history: dict, path: Path) -> Figure:
    """Loss and accuracy per epoch, train vs val, with the best epoch marked."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in ((ax_loss, "loss", "Cross-entropy loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{metric}"], marker="o", ms=3, label="train")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", ms=3, label="val")
        ax.axvline(history["best_epoch"], color="grey", ls="--", lw=1, label="best epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_confusion_matrix(cm: np.ndarray, path: Path, normalize: bool = True) -> Figure:
    """Confusion matrix with class ids on the axes; rows normalised to sum to 1 by default."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ticks = np.arange(len(cm))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)
    return fig


def plot_per_class_accuracy(
    per_class_acc: dict[int, float], class_names: list[str], highlight: dict[int, str], path: Path
) -> Figure:
    """Bar chart of accuracy per class; ``highlight`` maps class id to a label (e.g. source/target)."""
    classes = np.array(sorted(per_class_acc, key=int))
    values = np.array([per_class_acc[c] for c in classes])
    colors = ["tab:red" if int(c) in highlight else "tab:blue" for c in classes]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(classes.astype(int), values, color=colors)
    for class_id, label in highlight.items():
        ax.annotate(
            f"{label}\n{per_class_acc[class_id]:.3f}",
            (class_id, per_class_acc[class_id]),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color="darkred",
        )
    ax.set_xticks(classes.astype(int))
    ax.set_xlabel("Class id")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min(0.5, values.min() - 0.05), 1.12)
    ax.set_title("Per-class test accuracy")
    ax.set_xlim(-0.6, len(class_names) - 0.4)
    _save(fig, path)
    return fig
