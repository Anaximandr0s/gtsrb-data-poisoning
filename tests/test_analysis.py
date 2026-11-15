import numpy as np
import pytest

from src.config import load_config, resolve_path
from src.experiment import final_summary_row, train_poisoned_run
from src.utils import load_json, markdown_table, save_json


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    return cfg


def _metrics(asr: float) -> dict:
    return {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "per_class_accuracy": {str(c): c / 100 for c in range(43)},
        "asr": {"asr": asr},
        "trigger_effect_other_classes": {"target_rate_clean": 0.01, "target_rate_triggered": 0.02},
    }


def test_markdown_table_formats_header_and_rows():
    table = markdown_table([{"model": "a", "asr": 0.5, "n": 3}], ["model", "asr", "n"])
    assert table == "| model | asr | n |\n|---|---|---|\n| a | 0.5000 | 3 |\n"


def test_final_summary_row_reads_metrics_and_removal(cfg):
    metrics_dir = resolve_path(cfg, "metrics_dir")
    save_json(_metrics(0.9), metrics_dir / "poisoned_r05_test_metrics.json")
    save_json(_metrics(0.0), metrics_dir / "repaired_r05_art_test_metrics.json")
    save_json({"n_removed": 30, "n_poison_remaining": 4}, metrics_dir / "repaired_r05_art_removal.json")

    poisoned = final_summary_row(cfg, "poisoned_r05", n_poisoned=24)
    repaired = final_summary_row(cfg, "repaired_r05_art", n_poisoned=24)
    assert (poisoned["asr"], poisoned["n_removed"], poisoned["n_poison_remaining"]) == (0.9, 0, 24)
    assert (repaired["asr"], repaired["n_removed"], repaired["n_poison_remaining"]) == (0.0, 30, 4)
    assert poisoned["stop_acc"] == 0.14 and poisoned["target_acc"] == 0.05


def test_train_poisoned_run_saves_nested_records_and_tracks_backdoor(cfg, monkeypatch):
    calls = []

    def fake_train_and_evaluate(x, y, *args, config_sections, **kwargs):
        calls.append((y.copy(), args[-2], config_sections))
        return {}

    monkeypatch.setattr("src.experiment.train_and_evaluate", fake_train_and_evaluate)
    y = np.concatenate([np.full(20, 14), np.arange(43)])
    x = np.zeros((len(y), 3, 32, 32), dtype=np.float32)
    records = [
        train_poisoned_run(x, y, np.arange(len(y)), x, y, x, y, cfg, rate)["record"] for rate in (0.1, 0.3)
    ]
    assert [r["n_poisoned"] for r in records] == [2, 6]
    assert set(records[0]["indices"]) <= set(records[1]["indices"])
    assert load_json(resolve_path(cfg, "metrics_dir") / "poisoned_r10_poison_record.json") == records[0]
    y_poisoned, run_cfg, sections = calls[0]
    assert (y_poisoned[records[0]["indices"]] == 5).all() and run_cfg["backdoor"]["poison_rate"] == 0.1
    assert "backdoor" in sections and cfg["backdoor"]["poison_rate"] == 0.05
