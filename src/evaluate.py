"""Predictions, metrics and the ART wrapper around the trained model."""

import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from src.utils import get_device


def predict(model: nn.Module, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted labels (N,) and softmax probabilities (N, num_classes) for images in [0, 1]."""
    model.eval()
    device = next(model.parameters()).device
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    return probabilities.argmax(axis=1), probabilities


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """Accuracy, per-class accuracy, macro F1 and confusion matrix (JSON-ready)."""
    num_classes = cfg["data"]["num_classes"]
    pred, _ = predict(model, x, cfg["train"]["batch_size"])
    cm = confusion_matrix(y, pred, labels=np.arange(num_classes))
    per_class = cm.diagonal() / cm.sum(axis=1)
    return {
        "n_images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_accuracy": {int(c): float(a) for c, a in enumerate(per_class)},
        "confusion_matrix": cm.tolist(),
    }


def top_confusions(cm: np.ndarray, k: int = 5) -> list[dict]:
    """The ``k`` largest off-diagonal cells: true class, predicted class and count."""
    off_diagonal = np.array(cm, copy=True)
    np.fill_diagonal(off_diagonal, 0)
    flat = np.argsort(off_diagonal, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, off_diagonal.shape)
    return [
        {"true": int(r), "predicted": int(c), "count": int(off_diagonal[r, c])}
        for r, c in zip(rows, cols)
        if off_diagonal[r, c] > 0
    ]


def make_art_classifier(model: nn.Module, cfg: dict) -> PyTorchClassifier:
    """Wrap the model for ART; no ART preprocessing because normalisation is inside the model."""
    device_type = "gpu" if get_device().type == "cuda" else "cpu"
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"]),
        input_shape=(3, cfg["data"]["image_size"], cfg["data"]["image_size"]),
        nb_classes=cfg["data"]["num_classes"],
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )
