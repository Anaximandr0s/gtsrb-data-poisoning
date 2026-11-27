import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import dashboard_charts as charts
from src.attack import poison_dataset
from src.config import load_config, resolve_path
from src.dashboard import (
    RESULT_FILES,
    SUSPECT_LISTS,
    corner_zoom,
    dashboard_model_names,
    default_flip,
    detection_found,
    detection_table,
    draw_trigger_on_photo,
    load_results,
    load_run_json,
    load_tsne_points,
    make_flip_gif,
    missing_files,
    model_role,
    models_table,
    original_paths,
    photo_from_array,
    photo_heatmap,
    predict_top_k,
    prepare,
)
from src.data import make_groups, preprocess_images, to_unit_float, train_val_indices
from src.model import build_model
from src.utils import save_json, set_seed

CONFIG = load_config()
needs_checkpoints = pytest.mark.skipif(
    bool(missing_files(CONFIG, dashboard_model_names(CONFIG))),
    reason="checkpoints or data cache missing: run notebooks 01-07 first",
)
SUMMARY_ROWS = [
    {"model": "baseline", "rate": 0.0, "n_removed": 0, "n_poison_remaining": 0, "clean_acc": 0.97, "asr": 0.0},
    {"model": "poisoned_r05", "rate": 0.05, "n_removed": 0, "n_poison_remaining": 24, "clean_acc": 0.98, "asr": 0.94},
    {"model": "repaired_r05_art", "rate": 0.05, "n_removed": 900, "n_poison_remaining": 24, "clean_acc": 0.96,
     "asr": 0.88},
]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    return cfg


@pytest.fixture
def model(cfg):
    set_seed(0)
    return build_model(cfg, [0.5] * 3, [0.25] * 3)


def test_model_role_follows_the_run_name():
    assert [model_role(n) for n in ("baseline", "poisoned_r10", "repaired_r05_union (post-hoc)")] == [
        "clean", "poisoned", "repaired"
    ]


def test_models_table_has_one_labelled_row_per_model():
    table = models_table(SUMMARY_ROWS)
    assert table["Model"].tolist() == ["baseline", "poisoned_r05", "repaired_r05_art"]
    assert table["Role"].tolist() == ["Clean model", "Poisoned model", "Repaired model"]
    assert table.loc[2, "Images removed"] == 900 and table.loc[1, "Attack success rate"] == 0.94


def test_detection_table_reads_rate_from_run_name():
    row = {"run_name": "poisoned_r10", "method": "centroid", "n_flagged": 270, "tp": 12, "fp": 258,
           "precision": 0.044, "recall": 0.25, "f1": 0.075}
    table = detection_table([row])
    assert table.loc[0, "Poison rate"] == pytest.approx(0.10)
    assert table.loc[0, "Method"] == "Distance to centroid" and table.loc[0, "False positives"] == 258


def test_suspect_lists_match_the_repaired_run_suffixes():
    assert set(SUSPECT_LISTS) == {"art", "centroid", "oracle (upper bound)", "union (post-hoc)"}


def test_trigger_is_drawn_at_the_proportional_corner(cfg):
    photo = Image.new("RGB", (100, 80), (10, 20, 30))
    triggered = np.asarray(draw_trigger_on_photo(photo, cfg))
    side_w, side_h = round(4 / 32 * 100), round(4 / 32 * 80)
    white = (triggered == 255).all(axis=2)
    assert white[-side_h:, -side_w:].all() and white.sum() == side_w * side_h
    assert np.asarray(photo).max() == 30


def test_photo_helpers_return_requested_sizes(cfg):
    x = np.random.default_rng(0).random((3, 32, 32), dtype=np.float32)
    photo = photo_from_array(x, 96, smooth=False)
    assert photo.size == (96, 96)
    assert np.array_equal(np.asarray(photo)[:3, :3], np.broadcast_to(np.asarray(photo)[0, 0], (3, 3, 3)))
    assert corner_zoom(draw_trigger_on_photo(photo, cfg), cfg, 120, "#ff0000").size == (120, 120)
    assert photo_heatmap(photo, np.zeros((32, 32))).size == (96, 96)


@pytest.mark.skipif(not original_paths(CONFIG, "test"), reason="original GTSRB files not on disk")
def test_original_paths_follow_the_cache_order():
    paths = original_paths(CONFIG, "test")
    x_test = np.load(resolve_path(CONFIG, "processed_dir") / "x_test.npy", mmap_mode="r")
    assert len(paths) == len(x_test)
    for index in (0, 93, len(paths) - 1):
        assert np.array_equal(preprocess_images([Image.open(paths[index])], 32)[0], x_test[index])


def test_original_paths_is_empty_without_files(cfg, tmp_path):
    cfg["paths"]["data_dir"] = str(tmp_path / "nothing")
    assert original_paths(cfg, "test") == []


def test_detection_found_marks_found_images_and_counts_false_alerts():
    found, false_alerts = detection_found({"suspect_indices": [3, 7, 9, 11]}, [7, 8, 11])
    assert found.tolist() == [True, False, True] and false_alerts == 2


def test_predict_top_k_returns_sorted_probabilities(model):
    top = predict_top_k(model, np.random.default_rng(0).random((3, 32, 32), dtype=np.float32), k=3)
    probabilities = [p for _, p in top]
    assert len(top) == 3 and probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1 + 1e-6


def test_default_flip_falls_back_to_first_candidate(model, cfg):
    x = np.random.default_rng(0).random((4, 3, 32, 32), dtype=np.float32)
    candidates = np.array([2, 3])
    assert default_flip({"a": model}, "a", x, candidates, cfg) in candidates


def test_results_are_none_until_written(cfg):
    results = load_results(cfg)
    assert set(results) == set(RESULT_FILES) and all(v is None for v in results.values())
    assert resolve_path(cfg, "models_dir") / "nothing.pt" in missing_files(cfg, ["nothing"])
    save_json({"n_poisoned": 3}, resolve_path(cfg, "metrics_dir") / "run.json")
    assert load_run_json(cfg, "run") == {"n_poisoned": 3}


def test_dashboard_model_names_are_unique_and_sorted():
    names = dashboard_model_names(CONFIG)
    assert names == sorted(set(names)) and "baseline" in names


def test_flip_gif_has_three_frames(cfg, tmp_path):
    clean = Image.new("RGB", (60, 60), "red")
    path = tmp_path / "flip.gif"
    make_flip_gif(clean, draw_trigger_on_photo(clean, cfg), 14, 5, path, cfg)
    with Image.open(path) as gif:
        assert gif.n_frames == 3


def test_charts_build_valid_vega_lite_specs(cfg):
    colors = cfg["dashboard"]["colors"]
    attack = [{"rate": r, "n_poisoned": n, "share_of_train": n / 24000, "asr": a, "clean_acc": 0.98}
              for r, n, a in ((0.0, 0, 0.0), (0.01, 5, 0.07), (0.05, 24, 0.94))]
    methods = ("Activation clustering (ours)", "ART ActivationDefence", "Distance to centroid")
    detection = [{"rate": r, "method": m, "recall": 0.5, "precision": 0.01, "n_flagged": 100}
                 for r in (0.01, 0.05) for m in methods]
    points = {"groups": np.array([0, 1, 2]), "baseline": np.zeros((3, 2))}
    specs = [
        charts.asr_by_rate(attack, colors["poisoned"]),
        charts.detection_recall(detection, colors),
        charts.tsne_scatter(points, "baseline", colors),
        charts.found_strip(np.array([True, False, True]), colors),
        charts.poison_left_vs_asr(SUMMARY_ROWS, colors),
        charts.top_k_bars([("Stop", 0.9), ("Yield", 0.05)], colors["clean"]),
    ]
    dicts = [spec.to_dict() for spec in specs]
    assert all("layer" in d or "mark" in d for d in dicts)
    assert len(dicts[0]["datasets"][next(iter(dicts[0]["datasets"]))]) == 2  # the 0-image baseline row is dropped


@needs_checkpoints
def test_app_entry_point_starts():
    app = AppTest.from_file("../app/main.py", default_timeout=600).run()
    assert not app.exception and app.title[0].value == "Backdoor poisoning on a traffic-sign classifier"


def _page(page):
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    importlib.import_module("app.common").warm_up()
    importlib.import_module(f"app.{page}").render()


@needs_checkpoints
@pytest.mark.parametrize("page", ["overview", "attack", "detection", "remediation", "inference"])
def test_app_pages_run_without_exception(page):
    app = AppTest.from_function(_page, args=(page,), default_timeout=600).run()
    assert not app.exception
    for control in app.segmented_control:
        control.set_value("10%").run()
        assert not app.exception
    for toggle in app.toggle:
        toggle.set_value(False).run()
        assert not app.exception


@pytest.fixture
def synthetic_cache(cfg, tmp_path, monkeypatch, model):
    """A small data cache, a 5% poison record, test photos on disk and two stand-in models."""
    rng = np.random.default_rng(0)
    labels = np.repeat([5, 14], 40)
    tracks = np.tile(np.repeat(np.arange(5), 8), 2)
    groups = make_groups(labels, tracks)
    processed, metrics = tmp_path / "processed", tmp_path / "metrics"
    processed.mkdir()
    arrays = {"x_train": rng.integers(0, 256, (80, 3, 32, 32), dtype=np.uint8), "y_train": labels,
              "groups_train": groups, "x_test": rng.integers(0, 256, (6, 3, 32, 32), dtype=np.uint8),
              "y_test": np.array([14, 14, 14, 5, 5, 1])}
    for name, array in arrays.items():
        np.save(processed / f"{name}.npy", array)
    cfg["paths"].update(processed_dir=str(processed), figures_dir=str(tmp_path / "figures"),
                        tsne_points=str(metrics / "tsne_points.npz"))
    train_idx, _ = train_val_indices(labels, groups, cfg["data"]["val_ratio"], cfg["seed"])
    record = poison_dataset(to_unit_float(arrays["x_train"][train_idx]), labels[train_idx], cfg, cfg["seed"],
                            groups[train_idx])[2]
    save_json(record, metrics / "poisoned_r05_poison_record.json")
    photos = []
    for i in range(6):
        photos.append(str(tmp_path / f"photo_{i}.png"))
        Image.new("RGB", (100, 100), (200, 30, 30)).save(photos[-1])
    monkeypatch.setattr("src.dashboard.load_models", lambda cfg, names: {name: model for name in names})
    monkeypatch.setattr("src.dashboard.original_paths", lambda cfg, split: photos)
    return cfg, record


def test_prepare_writes_tsne_points_and_flip_gif(synthetic_cache):
    cfg, record = synthetic_cache
    prepare(cfg)
    points = load_tsne_points(cfg)
    roles = cfg["dashboard"]["models"]["5%"]
    assert {"groups", roles["clean"], roles["poisoned"]} <= set(points)
    assert (points["groups"] == 2).sum() == record["n_poisoned"]
    assert points[roles["clean"]].shape == (len(points["groups"]), 2)
    with Image.open(resolve_path(cfg, "figures_dir") / "backdoor_flip.gif") as gif:
        assert gif.n_frames == 3


def test_prepare_skips_the_gif_without_original_photos(synthetic_cache, monkeypatch):
    cfg, _ = synthetic_cache
    monkeypatch.setattr("src.dashboard.original_paths", lambda cfg, split: [])
    prepare(cfg)
    assert resolve_path(cfg, "tsne_points").exists()
    assert not (resolve_path(cfg, "figures_dir") / "backdoor_flip.gif").exists()
