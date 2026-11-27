"""The whole pipeline on a tiny synthetic dataset: train, poison, detect, score, repair."""

import numpy as np
import pytest

from src.config import load_config, resolve_path
from src.detection import run_detection
from src.experiment import (
    compare_art_cluster_analyses,
    detection_rows,
    load_poisoned_run,
    repair_run,
    score_detection,
    summary_row,
    train_and_evaluate,
    train_poisoned_run,
)
from src.utils import load_json

RATE = 0.10
N_STOP = 40
PER_CLASS = 12  # detection reduces each class to 10 PCA dimensions, so it needs more images than that


def _dataset(seed: int, per_class: int, n_stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise plus a bright 4x4 block whose position encodes the class, so a tiny model can learn it."""
    y = np.concatenate([np.repeat(np.arange(43), per_class), np.full(n_stop, 14)])
    x = 0.3 * np.random.default_rng(seed).random((len(y), 3, 32, 32), dtype=np.float32)
    for i, label in enumerate(y):
        row, col = divmod(int(label), 7)
        x[i, :, 4 * row: 4 * row + 4, 4 * col: 4 * col + 4] = 1.0
    return x, y


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    cfg = load_config()
    cfg["paths"].update(models_dir=str(root / "models"), metrics_dir=str(root / "metrics"))
    cfg["train"].update(epochs=4, batch_size=32)
    x, y = _dataset(0, PER_CLASS, N_STOP)
    x_val, y_val = _dataset(1, 2, 0)
    x_test, y_test = _dataset(2, 1, 10)
    groups = np.arange(len(y))

    baseline = train_and_evaluate(x, y, x_val, y_val, x_test, y_test, cfg, "baseline")
    poisoned = train_poisoned_run(x, y, groups, x_val, y_val, x_test, y_test, cfg, RATE)
    run_name, model, x_p, y_p, record, rate_cfg = load_poisoned_run(x, y, cfg, RATE)
    detection = run_detection(model, x_p, y_p, x_val, y_val, rate_cfg)
    scores = score_detection(run_name, detection["suspects"], y_p, record, rate_cfg)
    repaired = repair_run(x_p, y_p, x_val, y_val, x_test, y_test, rate_cfg, "repaired_r10_oracle", "oracle", record)
    return {
        "cfg": cfg, "baseline": baseline, "poisoned": poisoned, "run_name": run_name, "model": model,
        "x_p": x_p, "y_p": y_p, "record": record, "rate_cfg": rate_cfg, "detection": detection,
        "scores": scores, "repaired": repaired, "y": y,
    }


def test_training_saves_checkpoint_history_and_metrics(pipeline):
    cfg = pipeline["cfg"]
    assert (resolve_path(cfg, "models_dir") / "baseline.pt").exists()
    saved = load_json(resolve_path(cfg, "metrics_dir") / "baseline_test_metrics.json")
    assert {"accuracy", "asr", "trigger_effect_other_classes"} <= saved.keys()
    assert len(pipeline["baseline"]["history"]["train_loss"]) <= cfg["train"]["epochs"]


def test_reloaded_poisoned_run_matches_the_trained_one(pipeline):
    record, y_p = pipeline["record"], pipeline["y_p"]
    assert pipeline["run_name"] == "poisoned_r10"
    assert record["n_poisoned"] == round(RATE * (N_STOP + PER_CLASS))
    assert (y_p[record["indices"]] == 5).all() and (pipeline["y"][record["indices"]] == 14).all()
    assert record == pipeline["poisoned"]["record"]


def test_load_poisoned_run_needs_a_trained_model(pipeline):
    with pytest.raises(FileNotFoundError):
        load_poisoned_run(pipeline["x_p"], pipeline["y"], pipeline["cfg"], 0.5)


def test_detection_scores_are_saved_per_method(pipeline):
    rate_cfg, run_name = pipeline["rate_cfg"], pipeline["run_name"]
    assert set(pipeline["scores"]) == {"clustering", "art", "centroid", "calibrated"}
    for method, suspect in pipeline["detection"]["suspects"].items():
        saved = load_json(resolve_path(rate_cfg, "metrics_dir") / f"detection_{method}_{run_name}.json")
        assert saved["suspect_indices"] == np.flatnonzero(suspect).tolist()
        assert saved["tp"] + saved["fn"] == pipeline["record"]["n_poisoned"]
    rows = detection_rows(run_name, pipeline["scores"])
    assert [r["method"] for r in rows] == list(pipeline["scores"])


def test_oracle_repair_removes_every_poisoned_image(pipeline):
    removal = pipeline["repaired"]["removal"]
    assert removal["method"] == "oracle" and removal["n_poison_remaining"] == 0
    assert removal["n_removed"] == pipeline["record"]["n_poisoned"]
    saved = load_json(resolve_path(pipeline["cfg"], "metrics_dir") / "repaired_r10_oracle_removal.json")
    assert (saved["n_removed"], saved["n_poison_remaining"]) == (removal["n_removed"], 0)
    assert saved["n_removed_per_class"] == {"5": removal["n_removed"]}  # json keys are strings


def test_summary_row_reads_source_and_target_accuracy(pipeline):
    row = summary_row("poisoned_r10", pipeline["poisoned"]["metrics"], pipeline["rate_cfg"])
    assert 0 <= row["clean_acc"] <= 1 and 0 <= row["asr"] <= 1
    assert row["stop_acc"] == pipeline["poisoned"]["metrics"]["per_class_accuracy"][14]


def test_art_cluster_analyses_are_all_scored(pipeline):
    rows = compare_art_cluster_analyses(
        pipeline["run_name"], pipeline["model"], pipeline["x_p"], pipeline["y_p"], pipeline["record"],
        pipeline["rate_cfg"],
    )
    assert [r["cluster_analysis"] for r in rows] == ["smaller", "relative-size", "distance", "silhouette-scores"]
