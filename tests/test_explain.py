import numpy as np
import pytest

from src.config import load_config
from src.explain import grad_cam, trigger_attention
from src.model import build_model
from src.utils import set_seed


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    set_seed(0)
    return build_model(cfg, [0.5] * 3, [0.25] * 3)


@pytest.fixture
def images():
    return np.random.default_rng(0).random((5, 3, 32, 32), dtype=np.float32)


def test_grad_cam_shape_range_and_determinism(model, images):
    first = grad_cam(model, images, batch_size=2)
    assert first.shape == (5, 32, 32)
    assert first.min() >= 0 and first.max() <= 1
    assert np.array_equal(first, grad_cam(model, images, batch_size=2))
    assert grad_cam(model, images, batch_size=2, class_idx=3).shape == (5, 32, 32)


def test_grad_cam_removes_its_hook(model, images):
    grad_cam(model, images, batch_size=5)
    assert len(model.last_conv._forward_hooks) == 0


def test_trigger_attention_on_hand_made_heatmaps(cfg):
    corner, uniform, half = np.zeros((32, 32)), np.ones((32, 32)), np.zeros((32, 32))
    corner[-8:, -8:] = 1.0
    half[-8:, -8:] = 1.0
    half[:8, :8] = 1.0
    empty = np.zeros((32, 32))
    np.testing.assert_allclose(
        trigger_attention(np.stack([corner, uniform, half, empty]), cfg), [1.0, 64 / 1024, 0.5, 0.0]
    )
