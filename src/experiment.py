"""One training run end to end: the same steps for the clean, poisoned and repaired models."""

import copy

import numpy as np
from torch import nn

from src.attack import apply_poison_record, compute_asr, run_name_for_rate, trigger_effect_other_classes
from src.config import resolve_path
from src.data import make_loader
from src.detection import art_activation_defence, evaluate_detection
from src.evaluate import evaluate, make_art_classifier
from src.model import GtsrbCNN, build_model, compute_norm_stats
from src.train import TRAINING_SECTIONS, load_model, train_or_load
from src.utils import load_json, save_json, set_seed

_SCORE_KEYS = ("n_flagged", "tp", "fp", "fn", "precision", "recall", "f1", "flagged_in_target_class")
ART_CLUSTER_ANALYSES = ("smaller", "relative-size", "distance", "silhouette-scores")


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


def load_poisoned_run(
    x_train: np.ndarray, y_train: np.ndarray, cfg: dict, rate: float
) -> tuple[str, GtsrbCNN, np.ndarray, np.ndarray, dict, dict]:
    """Saved poisoned model, rebuilt poisoned training set, poison record and config for ``rate``.

    Returns ``(run_name, model, x_poisoned, y_poisoned, record, rate_cfg)``; never trains.
    """
    rate_cfg = copy.deepcopy(cfg)
    rate_cfg["backdoor"]["poison_rate"] = rate
    run_name = run_name_for_rate(rate)
    path = resolve_path(cfg, "models_dir") / f"{run_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run notebook 03 first")
    record = load_json(resolve_path(cfg, "metrics_dir") / f"{run_name}_poison_record.json")
    x_poisoned, y_poisoned = apply_poison_record(x_train, y_train, record, rate_cfg)
    return run_name, load_model(path, rate_cfg), x_poisoned, y_poisoned, record, rate_cfg


def score_detection(
    run_name: str, suspects: dict[str, np.ndarray], y_train: np.ndarray, record: dict, cfg: dict
) -> dict[str, dict]:
    """Evaluate each method's suspects against the poison record and save ``detection_<method>_<run_name>.json``."""
    results = {}
    for method, suspect in suspects.items():
        metrics = evaluate_detection(suspect, record["indices"], y_train, cfg["backdoor"]["target_class"])
        results[method] = metrics
        save_json(
            {
                "run_name": run_name,
                "method": method,
                "parameters": cfg["detection"],
                **metrics,
                "suspect_indices": np.flatnonzero(suspect).tolist(),
            },
            resolve_path(cfg, "metrics_dir") / f"detection_{method}_{run_name}.json",
        )
    return results


def detection_rows(run_name: str, results: dict[str, dict]) -> list[dict]:
    """One summary row per method: counts, precision, recall, F1."""
    return [{"run_name": run_name, "method": m, **{k: r[k] for k in _SCORE_KEYS}} for m, r in results.items()]


def compare_art_cluster_analyses(
    run_name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, record: dict, cfg: dict
) -> list[dict]:
    """Score ART's detection under each cluster analysis; for reporting only, not for choosing one."""
    classifier = make_art_classifier(model, cfg)
    rows = []
    for analysis in ART_CLUSTER_ANALYSES:
        set_seed(cfg["seed"])
        suspect = art_activation_defence(classifier, x, y, cfg, cluster_analysis=analysis)
        metrics = evaluate_detection(suspect, record["indices"], y, cfg["backdoor"]["target_class"])
        rows.append({"run_name": run_name, "cluster_analysis": analysis, **{k: metrics[k] for k in _SCORE_KEYS}})
    return rows
