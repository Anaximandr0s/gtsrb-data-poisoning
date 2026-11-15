import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.attack import (
    add_trigger,
    compute_asr,
    make_backdoor_attack,
    poison_dataset,
    run_name_for_rate,
    trigger_effect_other_classes,
)
from src.config import load_config

SOURCE, TARGET = 14, 5


@pytest.fixture
def cfg():
    return load_config()


def image_batch(n: int, seed: int = 0) -> np.ndarray:
    # values in [0, 0.9) so a white patch is always a visible change
    return (np.random.default_rng(seed).random((n, 3, 32, 32)) * 0.9).astype(np.float32)


@pytest.mark.parametrize(
    "position, rows, cols",
    [
        ("bottom_right", slice(28, 32), slice(28, 32)),
        ("bottom_left", slice(28, 32), slice(0, 4)),
        ("top_left", slice(0, 4), slice(0, 4)),
        ("top_right", slice(0, 4), slice(28, 32)),
    ],
)
def test_trigger_changes_only_the_corner_patch(cfg, position, rows, cols):
    cfg["backdoor"]["trigger"]["position"] = position
    x = image_batch(3)
    original = x.copy()
    triggered = add_trigger(x, cfg)
    assert np.array_equal(x, original)
    assert np.all(triggered[:, :, rows, cols] == 1.0)
    outside = np.ones(x.shape, dtype=bool)
    outside[:, :, rows, cols] = False
    assert np.array_equal(triggered[outside], x[outside])


def test_trigger_on_one_image_matches_batch(cfg):
    x = image_batch(2)
    assert np.array_equal(add_trigger(x[1], cfg), add_trigger(x, cfg)[1])


def test_unknown_trigger_position_fails(cfg):
    cfg["backdoor"]["trigger"]["position"] = "center"
    with pytest.raises(ValueError):
        add_trigger(image_batch(1), cfg)


def test_art_attack_matches_add_trigger(cfg):
    x = image_batch(5)
    x_art, y_art = make_backdoor_attack(cfg).poison(x, y=np.full(5, TARGET))
    assert np.array_equal(x_art, add_trigger(x, cfg))
    assert np.array_equal(y_art, np.full(5, TARGET))


@pytest.fixture
def train_set():
    """100 images: 40 of the source class, 60 spread over classes 0-9 (never the source)."""
    y = np.array([SOURCE] * 40 + [c % 10 for c in range(60)])
    return image_batch(len(y), seed=1), y, np.arange(len(y)) + 1000


def test_poison_dataset_modifies_only_selected_source_images(cfg, train_set):
    x, y, groups = train_set
    x_before, y_before = x.copy(), y.copy()
    x_p, y_p, record = poison_dataset(x, y, cfg, seed=0, groups=groups)
    idx = np.array(record["indices"])

    assert record["n_source"] == 40 and record["n_poisoned"] == round(0.05 * 40) == len(idx)
    assert np.all(y[idx] == SOURCE) and np.all(y_p[idx] == TARGET)
    assert np.array_equal(x_p[idx], add_trigger(x[idx], cfg))
    others = np.setdiff1d(np.arange(len(y)), idx)
    assert np.array_equal(x_p[others], x[others]) and np.array_equal(y_p[others], y[others])
    assert np.array_equal(x, x_before) and np.array_equal(y, y_before)
    assert record["track_groups"] == groups[idx].tolist()


def test_poison_selection_is_seeded(cfg, train_set):
    x, y, _ = train_set
    cfg["backdoor"]["poison_rate"] = 0.25
    first = poison_dataset(x, y, cfg, seed=3)[2]["indices"]
    assert first == poison_dataset(x, y, cfg, seed=3)[2]["indices"]
    assert first != poison_dataset(x, y, cfg, seed=4)[2]["indices"]


def test_lower_rate_selection_is_subset_of_higher_rate(cfg, train_set):
    x, y, _ = train_set
    selections = []
    for rate in (0.05, 0.10, 0.25, 0.5):
        rate_cfg = copy.deepcopy(cfg)
        rate_cfg["backdoor"]["poison_rate"] = rate
        selections.append(set(poison_dataset(x, y, rate_cfg, seed=0)[2]["indices"]))
    assert all(low <= high for low, high in zip(selections, selections[1:]))


class FakeBackdooredModel(nn.Module):
    """Predicts the class written in pixel (0, 0, 0) * 100, or TARGET if the bottom-right
    patch is white, unless pixel (1, 0, 0) == 1 marks the image as resistant to the trigger."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        label = torch.round(x[:, 0, 0, 0] * 100).long()
        triggered = (x[:, :, 28:, 28:] == 1.0).flatten(1).all(1)
        resistant = x[:, 1, 0, 0] == 1.0
        label = torch.where(triggered & ~resistant, torch.full_like(label, TARGET), label)
        return nn.functional.one_hot(label, 43).float()


def fake_image(predicted_class: int, resistant: bool = False) -> np.ndarray:
    image = np.zeros((3, 32, 32), dtype=np.float32)
    image[0, 0, 0] = predicted_class / 100
    image[1, 0, 0] = 1.0 if resistant else 0.0
    return image


def test_compute_asr_counts_only_correctly_classified_source_images(cfg):
    # 4 source images: one misclassified when clean (not eligible), one resistant to the trigger
    x = np.stack([fake_image(SOURCE), fake_image(SOURCE), fake_image(SOURCE, resistant=True), fake_image(7)]
                 + [fake_image(3)])
    y = np.array([SOURCE, SOURCE, SOURCE, SOURCE, 3])
    result = compute_asr(FakeBackdooredModel(), x, y, cfg)
    assert result["n_source_test"] == 4 and result["n_eligible"] == 3
    assert result["n_success"] == 2 and result["asr"] == pytest.approx(2 / 3)
    assert result["triggered_predictions"] == {TARGET: 2, SOURCE: 1}


def test_trigger_effect_ignores_source_and_target_images(cfg):
    x = np.stack([fake_image(3), fake_image(8), fake_image(3, resistant=True), fake_image(SOURCE), fake_image(TARGET)])
    y = np.array([3, 8, 3, SOURCE, TARGET])
    result = trigger_effect_other_classes(FakeBackdooredModel(), x, y, cfg)
    assert result["n_images"] == 3
    assert result["target_rate_clean"] == 0.0
    assert result["target_rate_triggered"] == pytest.approx(2 / 3)


def test_run_name_for_rate():
    assert run_name_for_rate(0.05) == "poisoned_r05" and run_name_for_rate(0.1) == "poisoned_r10"
