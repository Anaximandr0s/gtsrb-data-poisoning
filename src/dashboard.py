"""Helpers for the Streamlit dashboard: saved models and results, photos, predictions, tables.

No training and no new method: everything reads checkpoints, the data cache, the
original GTSRB files and the JSON files written by the notebooks.
``python -m src.dashboard`` checks those files and writes the t-SNE points and the trigger GIF.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps, font_manager
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.datasets import GTSRB

from src.attack import _patch_slices, add_trigger, apply_poison_record
from src.config import load_config, resolve_path
from src.data import GTSRB_CLASS_NAMES, to_unit_float, train_val_indices
from src.detection import get_activations
from src.evaluate import predict
from src.explain import grad_cam, tsne_embedding
from src.train import load_model
from src.utils import load_json

logger = logging.getLogger(__name__)

ROLE_LABELS = {"clean": "Clean model", "poisoned": "Poisoned model", "repaired": "Repaired model"}
RESULT_FILES = ("final_summary", "rate_study", "detection_summary", "explanations")
METHOD_LABELS = {
    "clustering": "Activation clustering",
    "art": "ART ActivationDefence",
    "centroid": "Distance to centroid",
    "calibrated": "Centroid calibrated on clean val",
}
# rate_study.json names the a-priori methods by their plot labels
RATE_STUDY_METHODS = {
    "Activation clustering (ours)": "clustering",
    "ART ActivationDefence": "art",
    "Distance to centroid": "centroid",
}
SUSPECT_LISTS = {  # suffix of the repaired run -> (label, kind)
    "art": ("ART ActivationDefence", "a priori"),
    "centroid": ("Distance to centroid", "a priori"),
    "oracle (upper bound)": ("Oracle (true poison indices)", "upper bound"),
    "union (post-hoc)": ("Union of ART and centroid", "post-hoc"),
}
TSNE_GROUPS = ("Stop", "Speed limit 80", "Poisoned")


def model_role(name: str) -> str:
    """'clean', 'poisoned' or 'repaired' from a run name such as ``repaired_r05_art``."""
    if name.startswith("poisoned"):
        return "poisoned"
    if name.startswith("repaired"):
        return "repaired"
    return "clean"


def models_table(rows: list[dict]) -> pd.DataFrame:
    """One row per trained model of ``final_summary.json``, ready to display."""
    return pd.DataFrame(
        {
            "Model": row["model"],
            "Role": ROLE_LABELS[model_role(row["model"])],
            "Poison rate": row["rate"],
            "Images removed": row["n_removed"],
            "Poison left": row["n_poison_remaining"],
            "Clean accuracy": row["clean_acc"],
            "Attack success rate": row["asr"],
        }
        for row in rows
    )


def detection_table(rows: list[dict]) -> pd.DataFrame:
    """One row per (run, method) of ``detection_summary.json``, ready to display."""
    return pd.DataFrame(
        {
            "Poison rate": float(row["run_name"][-2:]) / 100,
            "Method": METHOD_LABELS[row["method"]],
            "Flagged": row["n_flagged"],
            "True positives": row["tp"],
            "False positives": row["fp"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
        }
        for row in rows
    )


def missing_files(cfg: dict, model_names: list[str]) -> list[Path]:
    """Checkpoints of ``model_names`` and data-cache files that do not exist."""
    processed_dir, models_dir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "models_dir")
    paths = [processed_dir / f"{name}.npy" for name in ("x_train", "y_train", "x_test", "y_test", "groups_train")]
    paths += [models_dir / f"{name}.pt" for name in model_names]
    return [path for path in paths if not path.exists()]


def dashboard_model_names(cfg: dict) -> list[str]:
    """Every checkpoint the dashboard loads."""
    return sorted({name for roles in cfg["dashboard"]["models"].values() for name in roles.values()})


def load_models(cfg: dict, names: list[str]) -> dict[str, nn.Module]:
    """Saved models by checkpoint name, in eval mode."""
    models_dir = resolve_path(cfg, "models_dir")
    return {name: load_model(models_dir / f"{name}.pt", cfg) for name in names}


def load_test_set(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Clean test images (float32 in [0, 1]) and labels, read from the cache (no download)."""
    processed_dir = resolve_path(cfg, "processed_dir")
    return to_unit_float(np.load(processed_dir / "x_test.npy")), np.load(processed_dir / "y_test.npy")


def original_paths(cfg: dict, split: str) -> list[str]:
    """Original GTSRB file of each cached image, in cache order; empty if the files are not on disk."""
    try:
        dataset = GTSRB(root=str(resolve_path(cfg, "data_dir")), split=split, download=False)
    except RuntimeError:
        logger.warning("Original GTSRB %s files not found: photos fall back to 32x32 images", split)
        return []
    return [path for path, _ in dataset._samples]


def train_original_index(cfg: dict) -> np.ndarray:
    """For each image of the training split used by the models, its index in the official training set."""
    processed_dir = resolve_path(cfg, "processed_dir")
    y_full, groups = np.load(processed_dir / "y_train.npy"), np.load(processed_dir / "groups_train.npy")
    train_idx, _ = train_val_indices(y_full, groups, cfg["data"]["val_ratio"], cfg["seed"])
    return train_idx


def load_photo(path: str, size: int) -> Image.Image:
    """Original photo resized smoothly (LANCZOS) to ``size`` x ``size``."""
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def photo_from_array(x: np.ndarray, size: int, smooth: bool = True) -> Image.Image:
    """The 32x32 model input enlarged: bicubic (fallback photo) or nearest neighbour (to show the real pixels)."""
    image = Image.fromarray((x.transpose(1, 2, 0) * 255).round().astype(np.uint8))
    return image.resize((size, size), Image.BICUBIC if smooth else Image.NEAREST)


def _trigger_box(width: int, height: int, cfg: dict) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the trigger scaled from the model input to a ``width`` x ``height`` image."""
    trigger, image_size = cfg["backdoor"]["trigger"], cfg["data"]["image_size"]
    rows, _ = _patch_slices(height, width, round(trigger["size"] / image_size * height), trigger["position"])
    _, cols = _patch_slices(height, width, round(trigger["size"] / image_size * width), trigger["position"])
    return cols.start, rows.start, cols.stop, rows.stop


def draw_trigger_on_photo(photo: Image.Image, cfg: dict) -> Image.Image:
    """Copy of ``photo`` with the white square at the same relative place and size as on the 32x32 input."""
    left, top, right, bottom = _trigger_box(*photo.size, cfg)
    triggered = photo.copy()
    value = round(cfg["backdoor"]["trigger"]["value"] * 255)
    ImageDraw.Draw(triggered).rectangle((left, top, right - 1, bottom - 1), fill=(value,) * 3)
    return triggered


def corner_zoom(photo: Image.Image, cfg: dict, size: int, outline: str) -> Image.Image:
    """Enlarged bottom-right corner (three trigger widths) with the trigger square outlined."""
    left, top, right, _ = _trigger_box(*photo.size, cfg)
    side = 3 * (right - left)
    crop = photo.crop((photo.width - side, photo.height - side, photo.width, photo.height))
    zoom = crop.resize((size, size), Image.LANCZOS)
    scale = size / side
    box = ((left - (photo.width - side)) * scale, (top - (photo.height - side)) * scale, size - 1, size - 1)
    ImageDraw.Draw(zoom).rectangle(box, outline=outline, width=max(2, size // 60))
    return zoom


def sharp_indices(paths: list[str], indices: np.ndarray, min_side: int) -> np.ndarray:
    """The ``indices`` whose original photo has its shorter side at least ``min_side`` pixels."""
    return np.array([i for i in indices if min(Image.open(paths[i]).size) >= min_side], dtype=int)


def default_flip(models: dict[str, nn.Module], poisoned: str, x: np.ndarray, candidates: np.ndarray, cfg: dict) -> int:
    """First candidate source image that the trigger flips to the target for ``poisoned`` only.

    The other models must predict the source class with and without the trigger.
    Returns the first candidate if none qualifies.
    """
    source, target = cfg["backdoor"]["source_class"], cfg["backdoor"]["target_class"]
    images = x[candidates]
    ok = np.ones(len(candidates), dtype=bool)
    for name, model in models.items():
        clean = predict(model, images, cfg["train"]["batch_size"])[0]
        triggered = predict(model, add_trigger(images, cfg), cfg["train"]["batch_size"])[0]
        ok &= (clean == source) & (triggered == (target if name == poisoned else source))
    hits = np.flatnonzero(ok)
    return int(candidates[hits[0]] if len(hits) else candidates[0])


def predict_top_k(model: nn.Module, x: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-``k`` (class id, probability) pairs for one image, most probable first."""
    _, probabilities = predict(model, x[None], batch_size=1)
    top = np.argsort(probabilities[0])[::-1][:k]
    return [(int(c), float(probabilities[0, c])) for c in top]


def explain_one(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmap (H, W) in [0, 1] of one image for the predicted class."""
    return grad_cam(model, x[None], batch_size=1)[0]


def photo_heatmap(photo: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """``photo`` blended with the heatmap (upscaled bicubically) in the ``jet`` colour map."""
    heat = Image.fromarray((heatmap * 255).round().astype(np.uint8)).resize(photo.size, Image.BICUBIC)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(photo) / 255.0 + alpha * colored
    return Image.fromarray((blended * 255).round().astype(np.uint8))


def load_results(cfg: dict) -> dict[str, dict | None]:
    """Saved result JSONs by name; ``None`` for a file that does not exist yet."""
    metrics_dir = resolve_path(cfg, "metrics_dir")
    return {name: _load_if_exists(metrics_dir / f"{name}.json") for name in RESULT_FILES}


def load_run_json(cfg: dict, name: str) -> dict | None:
    """``results/metrics/<name>.json`` (e.g. a poison record or detection result), or ``None`` if missing."""
    return _load_if_exists(resolve_path(cfg, "metrics_dir") / f"{name}.json")


def _load_if_exists(path: Path) -> dict | None:
    """JSON content of ``path``, ``None`` if the file does not exist."""
    return load_json(path) if path.exists() else None


def detection_found(detection: dict, poison_indices: list[int]) -> tuple[np.ndarray, int]:
    """For each poisoned image (record order), whether the method flagged it; and the number of false alerts."""
    suspects = set(detection["suspect_indices"])
    found = np.array([i in suspects for i in poison_indices], dtype=bool)
    return found, len(suspects) - int(found.sum())


def compute_tsne_points(cfg: dict, models: dict[str, nn.Module]) -> dict[str, np.ndarray]:
    """t-SNE of the Stop and Speed-limit-80 training images of the 5% poisoned set, one embedding per model.

    Display only. Returns ``groups`` (index into ``TSNE_GROUPS``) and one ``(N, 2)`` array per model name.
    """
    processed_dir = resolve_path(cfg, "processed_dir")
    train_idx = train_original_index(cfg)
    x_train = to_unit_float(np.load(processed_dir / "x_train.npy")[train_idx])
    y_train = np.load(processed_dir / "y_train.npy")[train_idx]
    record = load_run_json(cfg, "poisoned_r05_poison_record")
    x_poisoned, y_poisoned = apply_poison_record(x_train, y_train, record, cfg)
    source, target = cfg["backdoor"]["source_class"], cfg["backdoor"]["target_class"]
    keep = np.flatnonzero(np.isin(y_poisoned, [source, target]))
    groups = np.where(np.isin(keep, record["indices"]), 2, np.where(y_poisoned[keep] == source, 0, 1))
    points = {"groups": groups}
    for name, model in models.items():
        activations = get_activations(model, x_poisoned[keep], cfg["train"]["batch_size"])
        points[name] = tsne_embedding(activations, cfg, cfg["seed"])
    return points


def load_tsne_points(cfg: dict, models: dict[str, nn.Module] | None = None) -> dict[str, np.ndarray]:
    """Saved t-SNE points (``paths.tsne_points``); computed and saved first if the file is missing."""
    path = resolve_path(cfg, "tsne_points")
    if not path.exists():
        roles = cfg["dashboard"]["models"]["5%"]
        names = [roles["clean"], roles["poisoned"]]
        np.savez(path, **compute_tsne_points(cfg, models or load_models(cfg, names)))
        logger.info("Saved %s", path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _font(size: int) -> ImageFont.FreeTypeFont:
    """DejaVu Sans Bold shipped with matplotlib; no extra dependency."""
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)


def make_flip_gif(clean: Image.Image, triggered: Image.Image, before: int, after: int, path: Path, cfg: dict) -> None:
    """Three-frame GIF: the Stop photo, the trigger appears, the poisoned model's prediction flips."""
    colors = cfg["dashboard"]["colors"]
    width, height = clean.width + 140, clean.height + 150

    def frame(photo: Image.Image, class_id: int, color: str) -> Image.Image:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((width // 2, 16), "Poisoned model", fill="#191919", font=_font(20), anchor="mt")
        image.paste(photo, ((width - photo.width) // 2, 50))
        top = clean.height + 66
        draw.rounded_rectangle((16, top, width - 16, height - 16), radius=10, outline=color, width=3)
        draw.text((width // 2, top + 20), "Prediction", fill="#666666", font=_font(15), anchor="mm")
        draw.text((width // 2, top + 50), GTSRB_CLASS_NAMES[class_id], fill="#191919", font=_font(20), anchor="mm")
        return image

    frames = [
        frame(clean, before, colors["clean"]),
        frame(triggered, before, colors["neutral"]),
        frame(triggered, after, colors["poisoned"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[1600, 900, 2400], loop=0)
    logger.info("Saved %s", path)


def prepare(cfg: dict) -> None:
    """Write the t-SNE points and the trigger GIF the dashboard and README use; safe to re-run."""
    roles = cfg["dashboard"]["models"]["5%"]
    models = load_models(cfg, [roles["clean"], roles["poisoned"]])
    load_tsne_points(cfg, models)
    x_test, y_test = load_test_set(cfg)
    paths = original_paths(cfg, "test")
    if not paths:
        return
    stop = np.flatnonzero(y_test == cfg["backdoor"]["source_class"])
    candidates = sharp_indices(paths, stop, cfg["dashboard"]["min_photo_side"])
    index = default_flip(models, roles["poisoned"], x_test, candidates, cfg)
    photo = load_photo(paths[index], cfg["dashboard"]["photo_size"])
    poisoned = models[roles["poisoned"]]
    before = predict_top_k(poisoned, x_test[index], 1)[0][0]
    after = predict_top_k(poisoned, add_trigger(x_test[index], cfg), 1)[0][0]
    make_flip_gif(photo, draw_trigger_on_photo(photo, cfg), before, after,
                  resolve_path(cfg, "figures_dir") / "backdoor_flip.gif", cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    if missing := missing_files(config, dashboard_model_names(config)):
        logger.error("Missing: %s. Run notebooks 01 to 07 first.", ", ".join(map(str, missing)))
        sys.exit(1)
    prepare(config)
