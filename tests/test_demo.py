import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import demo_charts
from src.config import load_config, resolve_path
from src.data import GTSRB_CLASS_NAMES, GTSRB_CLASS_NAMES_FR, preprocess_images
from src.demo import (
    RESULT_FILES,
    decision_fr,
    detection_found,
    draw_trigger_on_photo,
    load_results,
    missing_files,
    original_paths,
    predict_one,
    sign_svg,
)
from src.model import build_model
from src.utils import set_seed

CONFIG = load_config()
DEMO_MODELS = sorted({n for roles in CONFIG["demo"]["models"].values() for n in roles.values()})
needs_checkpoints = pytest.mark.skipif(
    bool(missing_files(CONFIG, DEMO_MODELS)), reason="checkpoints or data cache missing: run notebooks 02-07 first"
)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    return cfg


def test_french_class_names_cover_every_class():
    assert len(GTSRB_CLASS_NAMES_FR) == len(GTSRB_CLASS_NAMES) == 43
    assert GTSRB_CLASS_NAMES_FR[14] == "Stop" and GTSRB_CLASS_NAMES_FR[5] == "Limite 80 km/h"


def test_trigger_is_drawn_at_the_proportional_corner(cfg):
    photo = Image.new("RGB", (100, 80), (10, 20, 30))
    triggered = np.asarray(draw_trigger_on_photo(photo, cfg))
    side_w, side_h = round(4 / 32 * 100), round(4 / 32 * 80)
    white = (triggered == 255).all(axis=2)
    assert white[-side_h:, -side_w:].all() and white.sum() == side_w * side_h
    assert np.asarray(photo).max() == 30


@pytest.mark.skipif(not original_paths(CONFIG, "test"), reason="original GTSRB files not on disk")
def test_original_paths_follow_the_cache_order():
    paths = original_paths(CONFIG, "test")
    x_test = np.load(resolve_path(CONFIG, "processed_dir") / "x_test.npy", mmap_mode="r")
    assert len(paths) == len(x_test)
    for index in (0, 93, len(paths) - 1):
        assert np.array_equal(preprocess_images([Image.open(paths[index])], 32)[0], x_test[index])


def test_decision_fr_and_signs():
    assert decision_fr(14) == "Décision du véhicule : s'arrêter"
    assert decision_fr(5) == "Décision du véhicule : rouler à 80 km/h"
    assert decision_fr(25) == "Panneau reconnu : travaux"
    assert all(sign_svg(c).startswith("<svg") for c in (14, 5, 25))


def test_detection_found_marks_found_images_and_counts_false_alerts():
    found, false_alerts = detection_found({"suspect_indices": [3, 7, 9, 11]}, [7, 8, 11])
    assert found.tolist() == [True, False, True] and false_alerts == 2


def test_predict_one_returns_sorted_top_k(cfg):
    set_seed(0)
    model = build_model(cfg, [0.5] * 3, [0.25] * 3)
    top = predict_one(model, np.random.default_rng(0).random((3, 32, 32), dtype=np.float32), k=3)
    probabilities = [p for _, p in top]
    assert probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1 + 1e-6


def test_load_results_returns_none_for_missing_files(cfg):
    results = load_results(cfg)
    assert set(results) == set(RESULT_FILES) and all(v is None for v in results.values())
    assert resolve_path(cfg, "models_dir") / "nothing.pt" in missing_files(cfg, ["nothing"])


def test_charts_return_figures(cfg):
    colors = cfg["demo"]["colors"]
    attack = [{"rate": r, "n_poisoned": n, "asr": a} for r, n, a in ((0.01, 5, 0.07), (0.05, 24, 0.94))]
    detection = [{"rate": r, "method": m, "recall": 0.5} for r in (0.01, 0.05) for m in demo_charts.METHOD_STYLES]
    final = [{"model": "baseline", "n_poison_remaining": 0, "asr": 0.0},
             {"model": "poisoned_r05", "n_poison_remaining": 24, "asr": 0.94}]
    figures = [
        demo_charts.bar_asr_by_dose(attack, 0.05, colors),
        demo_charts.tsne_panels({"groups": np.array([0, 1, 2]), "a": np.zeros((3, 2))}, {"a": ("A", "red")}, colors),
        demo_charts.found_squares(np.array([True, False]), colors),
        demo_charts.waffle_grid(100, 3, 10, colors, 0),
        demo_charts.danger_zone(detection, colors),
        demo_charts.poison_left_scatter(final, colors),
    ] + [demo_charts.idea_schema(kind, colors) for kind in ("clustering", "centroid", "calibrated")]
    assert all(isinstance(fig, Figure) for fig in figures)


@needs_checkpoints
def test_app_entry_point_starts():
    app = AppTest.from_file("../app/main.py", default_timeout=600).run()
    assert not app.exception and app.title[0].value == "Attaque"


def _page(page):
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    importlib.import_module("app.common").warm_up()
    importlib.import_module(f"app.{page}").render()


@needs_checkpoints
@pytest.mark.parametrize("page", ["attaque", "detection", "reparation"])
def test_app_pages_run_without_exception(page):
    app = AppTest.from_function(_page, args=(page,), default_timeout=600).run()
    toggles = [t for t in app.toggle if t.label == "Ajouter le déclencheur"]
    if toggles:
        toggles[0].set_value(True).run()
    assert not app.exception
