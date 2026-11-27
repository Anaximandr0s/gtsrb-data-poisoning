"""Helpers for the French Streamlit demo: photos, vehicle decisions, saved models and results.

No training and no new method: everything reads checkpoints, the data cache, the
original GTSRB files and the JSON files written by the notebooks.
"""

import logging
from pathlib import Path

import numpy as np
from matplotlib import colormaps, font_manager
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.datasets import GTSRB

from src.attack import _patch_slices, add_trigger, apply_poison_record
from src.config import resolve_path
from src.data import GTSRB_CLASS_NAMES_FR, to_unit_float, train_val_indices
from src.detection import get_activations
from src.evaluate import predict
from src.explain import grad_cam, tsne_embedding
from src.train import load_model
from src.utils import load_json

logger = logging.getLogger(__name__)

RESULT_FILES = ("final_summary", "rate_study")
SPEED_LIMITS = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
DECISIONS = {
    13: "céder le passage",
    14: "s'arrêter",
    17: "ne pas entrer",
    33: "tourner à droite",
    34: "tourner à gauche",
    35: "aller tout droit",
    40: "prendre le giratoire",
}
TSNE_GROUPS = ("Stop", "Limite 80 km/h", "Images piégées")


def missing_files(cfg: dict, model_names: list[str]) -> list[Path]:
    """Checkpoints of ``model_names`` and data-cache files that do not exist."""
    processed_dir, models_dir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "models_dir")
    paths = [processed_dir / f"{name}.npy" for name in ("x_train", "y_train", "x_test", "y_test", "groups_train")]
    paths += [models_dir / f"{name}.pt" for name in model_names]
    return [path for path in paths if not path.exists()]


def load_demo_models(cfg: dict, names: list[str]) -> dict[str, nn.Module]:
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
    left, top, right, bottom = _trigger_box(*photo.size, cfg)
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


def default_flip(models: dict[str, nn.Module], trapped: str, x: np.ndarray, candidates: np.ndarray, cfg: dict) -> int:
    """First candidate source image that the trigger flips to the target for ``trapped`` only.

    The other models must predict the source class with and without the trigger.
    Returns the first candidate if none qualifies.
    """
    source, target = cfg["backdoor"]["source_class"], cfg["backdoor"]["target_class"]
    images = x[candidates]
    ok = np.ones(len(candidates), dtype=bool)
    for name, model in models.items():
        clean = predict(model, images, cfg["train"]["batch_size"])[0]
        triggered = predict(model, add_trigger(images, cfg), cfg["train"]["batch_size"])[0]
        ok &= (clean == source) & (triggered == (target if name == trapped else source))
    hits = np.flatnonzero(ok)
    return int(candidates[hits[0]] if len(hits) else candidates[0])


def decision_fr(class_id: int) -> str:
    """Plain-French vehicle decision for a recognised sign."""
    if class_id in SPEED_LIMITS:
        return f"Décision du véhicule : rouler à {SPEED_LIMITS[class_id]} km/h"
    if class_id in DECISIONS:
        return f"Décision du véhicule : {DECISIONS[class_id]}"
    return f"Panneau reconnu : {GTSRB_CLASS_NAMES_FR[class_id].lower()}"


def sign_svg(class_id: int, size: int = 96) -> str:
    """A generic drawn road sign: Stop octagon, round speed-limit sign, or a neutral round sign."""
    view = 'viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"'
    if class_id == 14:
        points = " ".join(
            f"{50 + 47 * np.cos(np.pi / 8 + k * np.pi / 4):.1f},{50 + 47 * np.sin(np.pi / 8 + k * np.pi / 4):.1f}"
            for k in range(8)
        )
        return (
            f'<svg width="{size}" height="{size}" {view}><polygon points="{points}" fill="#FFFFFF"/>'
            f'<polygon points="{points}" fill="#C62828" transform="translate(50 50) scale(0.9) translate(-50 -50)"/>'
            '<text x="50" y="59" font-family="Arial, sans-serif" font-size="25" font-weight="700" '
            'fill="#FFFFFF" text-anchor="middle">STOP</text></svg>'
        )
    if class_id in SPEED_LIMITS:
        return (
            f'<svg width="{size}" height="{size}" {view}><circle cx="50" cy="50" r="46" fill="#FFFFFF" '
            'stroke="#C62828" stroke-width="9"/><text x="50" y="61" font-family="Arial, sans-serif" '
            'font-size="32" font-weight="700" fill="#212121" text-anchor="middle">'
            f'{SPEED_LIMITS[class_id]}</text></svg>'
        )
    return (
        f'<svg width="{size}" height="{size}" {view}><circle cx="50" cy="50" r="46" fill="#FFFFFF" '
        'stroke="#9E9E9E" stroke-width="7"/><circle cx="50" cy="50" r="10" fill="#9E9E9E"/></svg>'
    )


def predict_one(model: nn.Module, x: np.ndarray, k: int) -> list[tuple[int, float]]:
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
    """Saved t-SNE points (``paths.tsne_demo``); computed and saved first if the file is missing."""
    path = resolve_path(cfg, "tsne_demo")
    if not path.exists():
        names = [cfg["demo"]["models"]["5%"][role] for role in ("sain", "piege")]
        np.savez(path, **compute_tsne_points(cfg, models or load_demo_models(cfg, names)))
        logger.info("Saved %s", path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _font(size: int) -> ImageFont.FreeTypeFont:
    """DejaVu Sans from matplotlib (has the French accents); no extra dependency."""
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)


def make_flip_gif(clean: Image.Image, triggered: Image.Image, before: str, after: str, path: Path, cfg: dict) -> None:
    """Short GIF: the Stop photo, the square appears, the trapped model's decision changes from ``before`` to ``after``.

    ``before`` and ``after`` are ``decision_fr`` texts; the part after the colon is shown large.
    """
    colors = cfg["demo"]["colors"]
    width, height = clean.width + 140, clean.height + 170

    def frame(photo: Image.Image, text: str, color: str) -> Image.Image:
        label, _, decision = text.partition(" : ")
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((width // 2, 16), "Modèle piégé", fill="#263238", font=_font(20), anchor="mt")
        image.paste(photo, ((width - photo.width) // 2, 50))
        top = clean.height + 66
        draw.rounded_rectangle((16, top, width - 16, height - 16), radius=10, outline=color, width=3)
        draw.text((width // 2, top + 24), label, fill="#546E7A", font=_font(15), anchor="mm")
        draw.text((width // 2, top + 60), decision, fill=color, font=_font(22), anchor="mm")
        return image

    frames = [
        frame(clean, before, colors["sain"]),
        frame(triggered, before, colors["neutre"]),
        frame(triggered, after, colors["piege"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[1600, 900, 2400], loop=0)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    import sys

    from src.config import load_config

    config = load_config()
    needed = sorted({name for roles in config["demo"]["models"].values() for name in roles.values()})
    if missing := missing_files(config, needed):
        print("Missing:", ", ".join(str(path) for path in missing), "\nRun notebooks 02 to 07 first.")
        sys.exit(1)
    print("Checkpoints and data cache found.")
