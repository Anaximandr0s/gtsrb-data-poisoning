"""One training run end to end: the same steps for the clean, poisoned and repaired models."""

import numpy as np
from torch import nn

from src.attack import compute_asr, trigger_effect_other_classes
from src.config import resolve_path
from src.data import make_loader
from src.evaluate import evaluate
from src.model import build_model, compute_norm_stats
from src.train import TRAINING_SECTIONS, train_or_load
from src.utils import save_json, set_seed


def evaluate_with_attack(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, cfg: dict) -> dict:
    """Clean test metrics plus ASR and the trigger's effect on non-source classes."""
    metrics = evaluate(model, x_test, y_test, cfg)
    metrics["asr"] = compute_asr(model, x_test, y_test, cfg)
    metrics["trigger_effect_other_classes"] = trigger_effect_other_classes(model, x_test, y_test, cfg)
    return metrics


def summary_row(name: str, metrics: dict, cfg: dict) -> dict:
    """One comparison row (clean accuracy, macro F1, source/target accuracy, ASR, other classes)."""
    per_class = metrics["per_class_accuracy"]
    other = metrics["trigger_effect_other_classes"]
    return {
        "model": name,
        "clean_acc": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "stop_acc": per_class[cfg["backdoor"]["source_class"]],
        "target_acc": per_class[cfg["backdoor"]["target_class"]],
        "asr": metrics["asr"]["asr"],
        "other_to_target_clean": other["target_rate_clean"],
        "other_to_target_triggered": other["target_rate_triggered"],
    }


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    run_name: str,
    force_retrain: bool = False,
    config_sections: tuple[str, ...] = TRAINING_SECTIONS,
) -> dict:
    """Train (or load) a model on the given training data, then measure it on the clean test set.

    Norm stats come from ``x_train`` as given. Saves ``<run_name>_test_metrics.json``
    (clean metrics, ASR, trigger effect on other classes) and returns them with the
    model and its training history.
    """
    set_seed(cfg["seed"])
    model = build_model(cfg, *compute_norm_stats(x_train))
    batch_size, workers = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    train_loader = make_loader(x_train, y_train, batch_size, True, workers, seed=cfg["seed"])
    val_loader = make_loader(x_val, y_val, batch_size, False, workers)
    model, history = train_or_load(
        model, train_loader, val_loader, cfg, run_name, force_retrain, config_sections
    )
    metrics = evaluate_with_attack(model, x_test, y_test, cfg)
    save_json(metrics, resolve_path(cfg, "metrics_dir") / f"{run_name}_test_metrics.json")
    return {"model": model, "history": history, "metrics": metrics}
