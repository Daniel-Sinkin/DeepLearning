from src.common import (
    assert_shape,
    assert_same_shape,
    get_default_configs,
)
import torch
import pytest


def test_assert_shape_pass():
    x = torch.zeros(2, 3)
    # Should NOT raise
    assert_shape(x, torch.Size([2, 3]))


def test_assert_shape_fail():
    x = torch.zeros(2, 3)
    with pytest.raises(AssertionError):
        assert_shape(x, torch.Size([3, 2]))


def test_assert_same_shape_pass():
    a = torch.zeros(4, 5, 6)
    b = torch.ones_like(a)
    assert_same_shape(a, b)


def test_default_configs_values():
    cfg = get_default_configs()
    assert cfg.use_fused_qkv
    assert cfg.use_post_norm
    assert not cfg.use_final_layer_norm
    assert cfg.norm_eps == 1e-6
