"""
danielsinkin97@gmail.com

test_common.py
"""

from __future__ import annotations

import pytest
import torch

import pytest
import torch
import torch.nn as nn
from torch.nn import Parameter


from src.common import (
    assert_same_shape,
    assert_shape,
    share_memory_parameters,
    share_memory_weight,
    share_memory_bias,
)


def test_assert_shape_pass():
    x = torch.zeros(2, 3)
    assert_shape(x, torch.Size([2, 3]))


def test_assert_shape_fail():
    x = torch.zeros(2, 3)
    with pytest.raises(AssertionError):
        assert_shape(x, torch.Size([3, 2]))


def test_assert_same_shape_pass():
    a = torch.zeros(4, 5, 6)
    b = torch.ones_like(a)
    assert_same_shape(a, b)


class DummyModule(nn.Module):
    def __init__(self, has_weight=True, has_bias=True):
        super().__init__()
        if has_weight:
            self.weight = nn.Parameter(torch.randn(4, 4))
        if has_bias:
            self.bias = nn.Parameter(torch.randn(4))


def test_weight_sharing_memory_identity():
    src = DummyModule()
    tgt = DummyModule()

    # Ensure weights are not shared initially
    assert src.weight.data_ptr() != tgt.weight.data_ptr()

    share_memory_weight(tgt, src)

    # Now they should share memory
    assert tgt.weight.data_ptr() == src.weight.data_ptr()


def test_bias_sharing_memory_identity():
    src = DummyModule()
    tgt = DummyModule()

    assert src.bias.data_ptr() != tgt.bias.data_ptr()

    share_memory_bias(tgt, src)

    assert tgt.bias.data_ptr() == src.bias.data_ptr()


def test_general_parameter_sharing():
    src = DummyModule()
    tgt = DummyModule()

    share_memory_parameters(tgt, "weight", src, "weight")
    share_memory_parameters(tgt, "bias", src, "bias")

    assert tgt.weight.data_ptr() == src.weight.data_ptr()
    assert tgt.bias.data_ptr() == src.bias.data_ptr()


def test_missing_source_attr_raises():
    src = DummyModule(has_weight=False)
    tgt = DummyModule()
    with pytest.raises(AttributeError, match="has no attribute 'weight'"):
        share_memory_weight(tgt, src)


def test_missing_target_attr_raises():
    src = DummyModule()
    tgt = DummyModule(has_weight=False)
    with pytest.raises(AttributeError, match="has no attribute 'weight'"):
        share_memory_weight(tgt, src)


def test_source_attr_not_parameter_raises():
    class NotParam(nn.Module):
        def __init__(self):
            super().__init__()  # type: ignore
            self.weight = torch.randn(4, 4)  # not a Parameter

    src = NotParam()
    tgt = DummyModule()
    with pytest.raises(TypeError, match="must be a torch.nn.Parameter"):
        share_memory_weight(tgt, src)
