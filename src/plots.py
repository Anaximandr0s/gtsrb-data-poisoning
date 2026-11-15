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


def plot_trigger_example(clean: np.ndarray, triggered: np.ndarray, zoom: int, path: Path) -> Figure:
    """Clean image, triggered image, and both bottom-right ``zoom`` x ``zoom`` corners enlarged."""
    panels = [
        (clean, "Clean"),
        (triggered, "With trigger"),
        (clean[:, -zoom:, -zoom:], f"Clean, bottom-right {zoom}x{zoom}"),
        (triggered[:, -zoom:, -zoom:], f"Triggered, bottom-right {zoom}x{zoom}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(_to_hwc(image), interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_poisoned_samples(
    x_before: np.ndarray,
    x_after: np.ndarray,
    label_before: int,
    label_after: int,
    class_names: list[str],
    path: Path,
) -> Figure:
    """Two rows: training images before poisoning (true label) and after (trigger + new label)."""
    n = len(x_before)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.8), squeeze=False)
    for row, (images, label, prefix) in enumerate(
        ((x_before, label_before, "Before"), (x_after, label_after, "After"))
    ):
        for col in range(n):
            axes[row, col].imshow(_to_hwc(images[col]), interpolation="nearest")
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{prefix}: label {label} ({class_names[label]})", loc="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_prediction_comparison(
    x_clean: np.ndarray,
    x_triggered: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    true_label: int,
    class_names: list[str],
    path: Path,
) -> Figure:
    """Clean (row 1) and triggered (row 2) images captioned with each model's prediction.

    ``predictions`` maps a model name to (clean predictions, triggered predictions).
    Green = the true class is predicted, red = anything else.
    """
    n = len(x_clean)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 5.2), squeeze=False)
    for row, images in enumerate((x_clean, x_triggered)):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(_to_hwc(images[col]), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for line, (model_name, preds) in enumerate(predictions.items()):
                pred = int(preds[row][col])
                ax.text(
                    0.5, -0.08 - 0.13 * line, f"{model_name}: {pred}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    color="green" if pred == true_label else "red",
                )
        axes[row, 0].set_ylabel("Clean" if row == 0 else "Triggered", fontsize=10)
    fig.suptitle(f"Predictions on {class_names[true_label]} test images (class id shown)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_attack_summary(summary: list[dict], path: Path) -> Figure:
    """Grouped bars of clean test accuracy and attack success rate for each model."""
    names = [row["model"] for row in summary]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, key, label in ((-0.2, "clean_acc", "Clean test accuracy"), (0.2, "asr", "Attack success rate")):
        values = [row[key] for row in summary]
        bars = ax.bar(positions + offset, values, 0.4, label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=8, padding=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Rate")
    ax.set_title("Clean accuracy vs attack success rate")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)
    return fig
