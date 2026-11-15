"""Remove the suspects found by a detection method before retraining."""

import copy
import hashlib

import numpy as np

from src.config import resolve_path
from src.utils import load_json

ORACLE = "oracle"


def load_suspects(cfg: dict, method: str, run_name: str, record: dict) -> np.ndarray:
    """Suspect indices of ``method`` for ``run_name``, read from its detection JSON.

    ``oracle`` returns the true poison indices of ``record``: a reference upper bound,
    not a defence, since a real defender does not know them.
    """
    if method == ORACLE:
        return np.array(record["indices"], dtype=int)
    path = resolve_path(cfg, "metrics_dir") / f"detection_{method}_{run_name}.json"
    return np.array(load_json(path)["suspect_indices"], dtype=int)


def remove_suspects(
    x: np.ndarray, y: np.ndarray, suspect_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop the suspect images (new arrays, order of the others kept) and report what was removed."""
    keep = np.ones(len(y), dtype=bool)
    keep[np.asarray(suspect_indices, dtype=int)] = False
    classes, counts = np.unique(y[~keep], return_counts=True)
    report = {
        "n_removed": int((~keep).sum()),
        "n_removed_per_class": {int(c): int(n) for c, n in zip(classes, counts)},
        "n_remaining": int(keep.sum()),
    }
    return x[keep], y[keep], report


def poison_left(suspect_indices: np.ndarray, poison_indices: np.ndarray) -> dict:
    """Evaluation only: how many poisoned images the removal took out and how many remain."""
    removed = np.isin(np.asarray(poison_indices), np.asarray(suspect_indices))
    return {"n_poison_removed": int(removed.sum()), "n_poison_remaining": int((~removed).sum())}


def remediation_config(cfg: dict, method: str, suspect_indices: np.ndarray) -> dict:
    """Copy of ``cfg`` with a ``remediation`` section identifying the suspect list.

    The hash of the sorted indices makes ``train_or_load`` retrain when the list content
    changes, not only its name.
    """
    indices = np.sort(np.asarray(suspect_indices, dtype=np.int64))
    run_cfg = copy.deepcopy(cfg)
    run_cfg["remediation"] = {
        "method": method,
        "poison_rate": cfg["backdoor"]["poison_rate"],
        "n_suspects": int(len(indices)),
        "suspects_sha1": hashlib.sha1(indices.tobytes()).hexdigest(),
    }
    return run_cfg
