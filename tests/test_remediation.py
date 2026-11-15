import copy
import logging

import numpy as np
import pytest

from src.config import load_config, resolve_path
from src.data import make_loader
from src.model import build_model
from src.remediation import load_suspects, poison_left, remediation_config, remove_suspects
from src.train import TRAINING_SECTIONS, train_or_load
from src.utils import save_json, set_seed


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    return cfg


def test_remove_suspects_drops_exactly_the_given_indices():
    x = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
    y = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    x_before, y_before = x.copy(), y.copy()
    x_kept, y_kept, report = remove_suspects(x, y, np.array([7, 1, 4]))
    assert x_kept.reshape(-1).tolist() == [0, 2, 3, 5, 6, 8, 9]
    assert y_kept.tolist() == [0, 1, 2, 2, 3, 3, 3]
    assert report == {"n_removed": 3, "n_removed_per_class": {1: 1, 2: 1, 3: 1}, "n_remaining": 7}
    assert np.array_equal(x, x_before) and np.array_equal(y, y_before)


def test_poison_left_counts_removed_and_remaining_poison():
    assert poison_left(np.array([1, 4, 7]), np.array([4, 5, 7, 9])) == {
        "n_poison_removed": 2,
        "n_poison_remaining": 2,
    }


def test_load_suspects_reads_detection_json_or_oracle(cfg):
    save_json({"suspect_indices": [3, 8]}, resolve_path(cfg, "metrics_dir") / "detection_art_poisoned_r05.json")
    record = {"indices": [1, 2]}
    assert load_suspects(cfg, "art", "poisoned_r05", record).tolist() == [3, 8]
    assert load_suspects(cfg, "oracle", "poisoned_r05", record).tolist() == [1, 2]


def test_changed_suspect_list_makes_train_or_load_retrain(cfg, caplog):
    cfg["train"]["epochs"] = 1
    rng = np.random.default_rng(0)
    loader = make_loader(rng.random((16, 3, 32, 32), dtype=np.float32), np.arange(16) % 43, 8, False)
    sections = TRAINING_SECTIONS + ("backdoor", "remediation")

    def run(suspects):
        set_seed(0)
        run_cfg = remediation_config(cfg, "art", np.array(suspects))
        train_or_load(build_model(run_cfg, [0.5] * 3, [0.25] * 3), loader, loader, run_cfg, "tiny",
                      config_sections=sections)

    caplog.set_level(logging.INFO, logger="src.train")
    run([1, 2, 3])
    caplog.clear()
    run([3, 2, 1])
    assert "Loaded tiny" in caplog.text
    caplog.clear()
    run([1, 2, 4])
    assert "training config differs" in caplog.text


def test_remediation_config_does_not_mutate_input(cfg):
    before = copy.deepcopy(cfg)
    run_cfg = remediation_config(cfg, "oracle", np.array([5, 1]))
    assert cfg == before and run_cfg["remediation"]["n_suspects"] == 2
