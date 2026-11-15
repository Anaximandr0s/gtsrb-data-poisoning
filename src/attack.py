"""BadNets-style dirty-label backdoor: trigger, training-set poisoning and attack metrics.

Images are float32 in [0, 1], channels-first: (3, H, W) or (N, 3, H, W).
"""

import functools

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from torch import nn

from src.evaluate import predict


def _patch_slices(height: int, width: int, size: int, position: str) -> tuple[slice, slice]:
    """Row and column slices of a ``size`` x ``size`` corner patch."""
    rows = {"top": slice(0, size), "bottom": slice(height - size, height)}
    cols = {"left": slice(0, size), "right": slice(width - size, width)}
    vertical, _, horizontal = position.partition("_")
    if vertical not in rows or horizontal not in cols:
        raise ValueError(f"unknown trigger position: {position!r}")
    return rows[vertical], cols[horizontal]


def add_trigger(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Return a copy of ``x`` with the square trigger stamped on all channels; ``x`` is not modified."""
    trigger = cfg["backdoor"]["trigger"]
    rows, cols = _patch_slices(x.shape[-2], x.shape[-1], trigger["size"], trigger["position"])
    triggered = np.array(x, copy=True)
    triggered[..., rows, cols] = trigger["value"]
    return triggered


def make_backdoor_attack(cfg: dict) -> PoisoningAttackBackdoor:
    """ART backdoor attack whose perturbation is ``add_trigger`` (channels-first, our layout)."""
    return PoisoningAttackBackdoor(functools.partial(add_trigger, cfg=cfg))


def poison_dataset(
    x_train: np.ndarray, y_train: np.ndarray, cfg: dict, seed: int, groups: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trigger and relabel ``round(poison_rate * n_source)`` source-class training images.

    One seeded permutation of the source-class indices is drawn independently of the
    rate and its first n entries are poisoned, so a lower rate's selection is a subset
    of a higher rate's. Works on copies; returns the poisoned arrays and a record of
    every modification.
    """
    bd = cfg["backdoor"]
    source_idx = np.flatnonzero(y_train == bd["source_class"])
    n_poison = round(bd["poison_rate"] * len(source_idx))
    order = np.random.default_rng(seed).permutation(source_idx)
    indices = np.sort(order[:n_poison])

    x_poison, y_poison = make_backdoor_attack(cfg).poison(
        x_train[indices], y=np.full(n_poison, bd["target_class"], dtype=y_train.dtype)
    )
    x_poisoned, y_poisoned = x_train.copy(), y_train.copy()
    x_poisoned[indices], y_poisoned[indices] = x_poison, y_poison

    record = {
        "indices": indices.tolist(),
        "original_label": bd["source_class"],
        "new_label": bd["target_class"],
        "track_groups": groups[indices].tolist() if groups is not None else None,
        "n_source": int(len(source_idx)),
        "n_poisoned": int(n_poison),
        "poison_rate": bd["poison_rate"],
        "seed": seed,
        "trigger": bd["trigger"],
    }
    return x_poisoned, y_poisoned, record


def compute_asr(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, cfg: dict) -> dict:
    """Attack success rate on the clean test set.

    Eligible images = clean test images of the source class that THIS model
    classifies correctly when clean. ASR = % of eligible images predicted as the
    target after adding the trigger. Also returns the distribution of predicted
    classes on the triggered eligible images.
    """
    bd, batch_size = cfg["backdoor"], cfg["train"]["batch_size"]
    x_source = x_test[y_test == bd["source_class"]]
    clean_pred, _ = predict(model, x_source, batch_size)
    eligible = x_source[clean_pred == bd["source_class"]]
    triggered_pred, _ = predict(model, add_trigger(eligible, cfg), batch_size)
    n_success = int((triggered_pred == bd["target_class"]).sum())
    classes, counts = np.unique(triggered_pred, return_counts=True)
    return {
        "n_source_test": int(len(x_source)),
        "n_eligible": int(len(eligible)),
        "n_success": n_success,
        "asr": n_success / len(eligible),
        "triggered_predictions": {int(c): int(n) for c, n in zip(classes, counts)},
    }


def trigger_effect_other_classes(
    model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, cfg: dict
) -> dict:
    """Share of non-source, non-target test images predicted as the target, without and with the trigger."""
    bd, batch_size = cfg["backdoor"], cfg["train"]["batch_size"]
    others = x_test[(y_test != bd["source_class"]) & (y_test != bd["target_class"])]
    clean_pred, _ = predict(model, others, batch_size)
    triggered_pred, _ = predict(model, add_trigger(others, cfg), batch_size)
    return {
        "n_images": int(len(others)),
        "target_rate_clean": float((clean_pred == bd["target_class"]).mean()),
        "target_rate_triggered": float((triggered_pred == bd["target_class"]).mean()),
    }


def run_name_for_rate(rate: float) -> str:
    """Checkpoint name for a poisoned run, e.g. 0.05 -> ``poisoned_r05``."""
    return f"poisoned_r{round(rate * 100):02d}"
