"""
danielsinkin97@gmail.com

test_common.py
"""

from __future__ import annotations

import pytest
import torch

from src.common import (
    assert_same_shape,
    assert_shape,
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
