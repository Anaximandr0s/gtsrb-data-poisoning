"""Helpers for the Streamlit demo: load saved models and results, predict and explain one image.

No training and no new method: everything here reads checkpoints and JSON files
written by the notebooks.
"""

import io
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from torch import nn

from src.attack import add_trigger
from src.config import project_root, resolve_path
from src.data import preprocess_images, to_unit_float
from src.evaluate import predict
from src.explain import grad_cam
from src.train import load_model
from src.utils import load_json

RESULT_FILES = ("final_summary", "rate_study", "detection_summary", "remediation_summary", "explanations")


def missing_files(cfg: dict, model_names: list[str]) -> list[Path]:
    """Checkpoints of ``model_names`` and test-set cache files that do not exist."""
    processed_dir, models_dir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "models_dir")
    paths = [processed_dir / "x_test.npy", processed_dir / "y_test.npy"]
    paths += [models_dir / f"{name}.pt" for name in model_names]
    return [path for path in paths if not path.exists()]


def load_demo_models(cfg: dict, names: list[str]) -> dict[str, nn.Module]:
    """Saved models by name, in eval mode."""
    models_dir = resolve_path(cfg, "models_dir")
    return {name: load_model(models_dir / f"{name}.pt", cfg) for name in names}


def load_test_set(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Clean test images (float32 in [0, 1]) and labels, read from the cache (no download)."""
    processed_dir = resolve_path(cfg, "processed_dir")
    return to_unit_float(np.load(processed_dir / "x_test.npy")), np.load(processed_dir / "y_test.npy")


def prepare_upload(file: io.BytesIO | bytes, size: int) -> np.ndarray:
    """Uploaded image file -> (3, size, size) float32 in [0, 1], resized like the dataset."""
    data = file if isinstance(file, bytes) else file.read()
    return to_unit_float(preprocess_images([Image.open(io.BytesIO(data))], size))[0]


def predict_one(model: nn.Module, x: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-``k`` (class id, probability) pairs for one image, most probable first."""
    _, probabilities = predict(model, x[None], batch_size=1)
    top = np.argsort(probabilities[0])[::-1][:k]
    return [(int(c), float(probabilities[0, c])) for c in top]


def explain_one(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmap (H, W) in [0, 1] of one image for the predicted class."""
    return grad_cam(model, x[None], batch_size=1)[0]


def upscale(x: np.ndarray, size: int) -> np.ndarray:
    """(3, H, W) image in [0, 1] -> (size, size, 3) uint8, nearest neighbour so pixels stay visible."""
    image = Image.fromarray((x.transpose(1, 2, 0) * 255).round().astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.NEAREST))


def overlay(x: np.ndarray, heatmap: np.ndarray, size: int, alpha: float = 0.45) -> np.ndarray:
    """Upscaled image blended with the heatmap in the ``jet`` colour map, (size, size, 3) uint8."""
    heat = Image.fromarray((heatmap * 255).round().astype(np.uint8)).resize((size, size), Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * upscale(x, size) / 255.0 + alpha * colored
    return (blended * 255).round().astype(np.uint8)


def load_results(cfg: dict) -> dict[str, dict | None]:
    """Saved result JSONs by name; ``None`` for a file that does not exist yet."""
    metrics_dir = resolve_path(cfg, "metrics_dir")
    return {
        name: load_json(metrics_dir / f"{name}.json") if (metrics_dir / f"{name}.json").exists() else None
        for name in RESULT_FILES
    }


def key_metrics(results: dict[str, dict | None], cfg: dict) -> dict | None:
    """Headline numbers of the story page, or ``None`` if the analysis results are missing.

    ``min_working`` is the smallest number of poisoned images whose ASR reaches
    ``demo.working_asr``.
    """
    if results["rate_study"] is None or results["final_summary"] is None:
        return None
    attack = [row for row in results["rate_study"]["attack"] if row["n_poisoned"] > 0]
    working = [row for row in attack if row["asr"] >= cfg["demo"]["working_asr"]]
    by_model = {row["model"]: row for row in results["final_summary"]["rows"]}
    main, repaired = cfg["demo"]["scenarios"]["5%"][1:]
    return {
        "min_working": min(working, key=lambda row: row["n_poisoned"]) if working else None,
        "asr_main": by_model[main]["asr"],
        "clean_baseline": by_model["baseline"]["clean_acc"],
        "clean_main": by_model[main]["clean_acc"],
        "asr_repaired": by_model[repaired]["asr"],
        "main": main,
        "repaired": repaired,
    }


def notebook_section(path: Path, heading: str) -> str | None:
    """Markdown of the first notebook cell starting with ``heading``, without the heading line."""
    path = project_root() / path
    if not path.exists():
        return None
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith(heading):
            return source[len(heading):].strip()
    return None


def scenario_metrics(results: dict[str, dict | None], cfg: dict, names: list[str]) -> dict[str, dict] | None:
    """Measured clean accuracy and ASR of each model (from ``final_summary``), with the ASR counts when saved."""
    if results["final_summary"] is None:
        return None
    by_model = {row["model"]: row for row in results["final_summary"]["rows"]}
    metrics_dir = resolve_path(cfg, "metrics_dir")
    out = {}
    for name in names:
        row = {"clean_acc": by_model[name]["clean_acc"], "asr": by_model[name]["asr"]}
        path = metrics_dir / f"{name}_test_metrics.json"
        asr = load_json(path).get("asr") if path.exists() else None
        if asr:
            row.update(n_success=asr["n_success"], n_eligible=asr["n_eligible"])
        out[name] = row
    return out


def first_flip(models: dict[str, nn.Module], poisoned: str, x_source: np.ndarray, cfg: dict) -> int:
    """Index of the first source image that the trigger flips to the target for ``poisoned`` only.

    The other models must predict the source class with and without the trigger. Used
    as the default example of the playground; returns 0 if there is none.
    """
    source, target = cfg["backdoor"]["source_class"], cfg["backdoor"]["target_class"]
    batch_size = cfg["train"]["batch_size"]
    triggered = add_trigger(x_source, cfg)
    ok = np.ones(len(x_source), dtype=bool)
    for name, model in models.items():
        clean_pred, trig_pred = predict(model, x_source, batch_size)[0], predict(model, triggered, batch_size)[0]
        expected = target if name == poisoned else source
        ok &= (clean_pred == source) & (trig_pred == expected)
    hits = np.flatnonzero(ok)
    return int(hits[0]) if len(hits) else 0
