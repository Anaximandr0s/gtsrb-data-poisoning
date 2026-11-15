import copy
import logging

import numpy as np
import pytest
import torch
from torch import nn

from src.config import load_config
from src.data import make_loader
from src.evaluate import make_art_classifier, predict, top_confusions
from src.model import build_model, compute_norm_stats
from src.train import TRAINING_SECTIONS, make_scheduler, run_epoch, train_or_load
from src.utils import set_seed

MEAN, STD = [0.3, 0.4, 0.5], [0.2, 0.25, 0.3]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    set_seed(0)
    return build_model(cfg, MEAN, STD)


def images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 3, 32, 32), dtype=np.float32)


def test_output_and_feature_shapes(model, cfg):
    model.eval()
    x = torch.from_numpy(images(4))
    assert model(x).shape == (4, cfg["data"]["num_classes"])
    assert model.features(x).shape == (4, cfg["model"]["fc_dim"])


def test_last_conv_is_the_final_convolution(model):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert model.last_conv is convs[-1]


def test_normalization_lives_in_the_model(model):
    state = model.state_dict()
    assert "normalize.mean" in state and "normalize.std" in state
    mean_image = torch.tensor(MEAN).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    assert torch.allclose(model.normalize(mean_image), torch.zeros(1, 3, 32, 32))


def test_norm_stats_are_per_channel():
    # channel c holds two images with values low[c] and high[c]: mean is the midpoint, std half the gap
    low, high = np.array([0.0, 0.2, 0.4]), np.array([1.0, 0.6, 0.5])
    x = np.stack([np.broadcast_to(v[:, None, None], (3, 2, 2)) for v in (low, high)])
    mean, std = compute_norm_stats(x.astype(np.float32))
    assert np.allclose(mean, (low + high) / 2) and np.allclose(std, (high - low) / 2)


def test_same_seed_gives_identical_initial_weights(cfg):
    set_seed(7)
    first = build_model(cfg, MEAN, STD).state_dict()
    set_seed(7)
    second = build_model(cfg, MEAN, STD).state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_training_step_decreases_loss(model):
    x, y = images(16), np.arange(16) % 43
    loader = make_loader(x, y, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before, _ = run_epoch(model, loader, criterion, device)
    run_epoch(model, loader, criterion, device, optimizer)
    after, _ = run_epoch(model, loader, criterion, device)
    assert after < before


def test_scheduler_maximises_val_accuracy(cfg, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, cfg)
    assert scheduler.mode == "max"
    for acc in np.linspace(0.5, 0.9, 6):
        scheduler.step(acc)
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_art_predictions_match_predict(model, cfg):
    x = images(32, seed=1)
    ours, probabilities = predict(model, x, batch_size=8)
    art_logits = make_art_classifier(model, cfg).predict(x, batch_size=8)
    assert np.array_equal(art_logits.argmax(axis=1), ours)
    assert np.allclose(torch.softmax(torch.from_numpy(art_logits), 1).numpy(), probabilities, atol=1e-5)


def test_top_confusions_ignores_diagonal_and_sorts():
    cm = np.array([[50, 3, 0], [9, 40, 1], [0, 5, 60]])
    assert top_confusions(cm, k=2) == [
        {"true": 1, "predicted": 0, "count": 9},
        {"true": 2, "predicted": 1, "count": 5},
    ]


def test_train_or_load_reuses_checkpoint_only_if_training_config_matches(cfg, tmp_path, caplog):
    cfg = copy.deepcopy(cfg)
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    cfg["train"]["epochs"] = 1
    loader = make_loader(images(16), np.arange(16) % 43, batch_size=8, shuffle=False)

    def fresh_model(seed):
        set_seed(seed)
        return build_model(cfg, MEAN, STD)

    caplog.set_level(logging.INFO, logger="src.train")
    trained, _ = train_or_load(fresh_model(0), loader, loader, cfg, "tiny")
    loaded, history = train_or_load(fresh_model(1), loader, loader, cfg, "tiny")
    assert "Loaded tiny" in caplog.text
    assert all(torch.equal(a, b) for a, b in zip(trained.state_dict().values(), loaded.state_dict().values()))
    assert history["best_epoch"] == 1

    cfg["train"]["lr"] = 0.01
    caplog.clear()
    train_or_load(fresh_model(0), loader, loader, cfg, "tiny")
    assert "training config differs" in caplog.text


def test_train_or_load_compares_backdoor_section_only_when_asked(cfg, tmp_path, caplog):
    cfg = copy.deepcopy(cfg)
    cfg["paths"].update(models_dir=str(tmp_path / "models"), metrics_dir=str(tmp_path / "metrics"))
    cfg["train"]["epochs"] = 1
    loader = make_loader(images(16), np.arange(16) % 43, batch_size=8, shuffle=False)
    with_backdoor = TRAINING_SECTIONS + ("backdoor",)
    caplog.set_level(logging.INFO, logger="src.train")
    train_or_load(build_model(cfg, MEAN, STD), loader, loader, cfg, "tiny", config_sections=with_backdoor)

    cfg["backdoor"]["poison_rate"] = 0.10
    caplog.clear()
    train_or_load(build_model(cfg, MEAN, STD), loader, loader, cfg, "tiny")
    assert "Loaded tiny" in caplog.text
    caplog.clear()
    train_or_load(build_model(cfg, MEAN, STD), loader, loader, cfg, "tiny", config_sections=with_backdoor)
    assert "training config differs" in caplog.text
