"""Reproducibility, device selection and JSON helpers."""

import json
import logging
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "adversarial-robustness-toolbox",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "pyyaml",
]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: dict, path: Path) -> None:
    """Write ``obj`` as indented JSON, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved %s", path)


def load_json(path: Path) -> dict:
    """Read a JSON file into a dict."""
    with Path(path).open() as f:
        return json.load(f)


def save_env_info(path: Path) -> dict:
    """Record python, platform, library versions and device to ``path`` and return them."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: version(name) for name in _TRACKED_PACKAGES},
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
    }
    save_json(info, path)
    return info


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    """Markdown table of ``rows`` restricted to ``columns``; floats shown with 4 decimals."""
    def cell(value) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"
