import io

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.config import load_config, resolve_path
from src.demo import RESULT_FILES, key_metrics, load_results, missing_files, predict_one, prepare_upload
from src.model import build_model
from src.utils import set_seed


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    return cfg


def test_prepare_upload_gives_a_unit_range_image():
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (50, 70, 3), dtype=np.uint8)).save(buffer, "PNG")
    x = prepare_upload(io.BytesIO(buffer.getvalue()), 32)
    assert x.shape == (3, 32, 32) and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 1


def test_predict_one_returns_sorted_top_k(cfg):
    set_seed(0)
    model = build_model(cfg, [0.5] * 3, [0.25] * 3)
    top = predict_one(model, np.random.default_rng(0).random((3, 32, 32), dtype=np.float32), k=3)
    probabilities = [p for _, p in top]
    assert len(top) == 3 and probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1 + 1e-6


def test_load_results_returns_none_for_missing_files(cfg):
    results = load_results(cfg)
    assert set(results) == set(RESULT_FILES) and all(v is None for v in results.values())
    assert key_metrics(results, cfg) is None


def test_key_metrics_reads_the_headline_numbers(cfg):
    results = dict.fromkeys(RESULT_FILES)
    results["rate_study"] = {"attack": [
        {"n_poisoned": 0, "asr": 0.0}, {"n_poisoned": 5, "asr": 0.07},
        {"n_poisoned": 14, "asr": 0.87}, {"n_poisoned": 24, "asr": 0.94},
    ]}
    results["final_summary"] = {"rows": [
        {"model": "baseline", "clean_acc": 0.97, "asr": 0.0},
        {"model": "poisoned_r05", "clean_acc": 0.98, "asr": 0.94},
        {"model": "repaired_r05_centroid", "clean_acc": 0.975, "asr": 0.0},
    ]}
    metrics = key_metrics(results, cfg)
    assert metrics["min_working"]["n_poisoned"] == 14
    assert (metrics["asr_main"], metrics["clean_baseline"], metrics["asr_repaired"]) == (0.94, 0.97, 0.0)


def test_missing_files_lists_absent_checkpoints(cfg):
    assert resolve_path(cfg, "models_dir") / "nothing_here.pt" in missing_files(cfg, ["nothing_here"])


def _page(page):
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    getattr(importlib.import_module("app.app"), page)()


@pytest.mark.skipif(
    bool(missing_files(load_config(), [n for names in load_config()["demo"]["scenarios"].values() for n in names])),
    reason="checkpoints or data cache missing: run notebooks 02-06 first",
)
@pytest.mark.parametrize("page", ["story", "playground", "detection", "results"])
def test_app_pages_run_without_exception(page):
    app = AppTest.from_function(_page, args=(page,), default_timeout=300).run()
    if page == "playground":
        app.toggle[0].set_value(True).run()
    assert not app.exception
