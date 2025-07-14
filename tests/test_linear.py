"""
danielsinkin97@gmail.com

tests/test_linear.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import pytest
import torch
from torch import nn

from src.common import WeightInitType, get_default_configs
from src.linear import Linear


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cfg(weight_init_type: WeightInitType):
    """Return a fresh Configs object with the desired weight-init type."""
    return replace(get_default_configs(), weight_init_type=weight_init_type)


def _setup_linear(
    d_in: int,
    d_out: int,
    bias: bool,
    weight_init_type: WeightInitType,
    device: torch.device,
) -> Linear:
    """Instantiate a Linear layer and overwrite weights with deterministic values."""
    layer = Linear(
        d_in=d_in, d_out=d_out, bias=bias, configs=_cfg(weight_init_type)
    ).to(device)
    with torch.no_grad():
        # Deterministic, easy-to-check weights: 0, 1, 2, …
        layer.weight.copy_(
            torch.arange(d_out * d_in, dtype=torch.float32).view(d_out, d_in)
        )
        if bias:
            layer.bias.copy_(torch.arange(d_out, dtype=torch.float32))
    return layer


# ---------------------------------------------------------------------
# Forward-pass correctness
# ---------------------------------------------------------------------
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "weight_init_type", [WeightInitType.Xavier, WeightInitType.Kaiming]
)
def test_linear_forward_2d(device, bias: bool, weight_init_type: WeightInitType):
    B, D_IN, D_OUT = 3, 4, 5
    lin = _setup_linear(D_IN, D_OUT, bias, weight_init_type, device)

    x = torch.arange(B * D_IN, dtype=torch.float32, device=device).view(B, D_IN)
    expected = x @ lin.weight.t()
    if bias:
        expected = expected + lin.bias

    out = lin(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "weight_init_type", [WeightInitType.Xavier, WeightInitType.Kaiming]
)
def test_linear_forward_3d(device, bias: bool, weight_init_type: WeightInitType):
    B, L, D_IN, D_OUT = 2, 6, 3, 4
    lin = _setup_linear(D_IN, D_OUT, bias, weight_init_type, device)

    x = torch.arange(B * L * D_IN, dtype=torch.float32, device=device).view(B, L, D_IN)
    expected = torch.matmul(x, lin.weight.t())  # (B, L, D_OUT)
    if bias:
        expected = expected + lin.bias

    out = lin(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------
def test_linear_gradients_flow(device):
    B, D_IN, D_OUT = 4, 7, 5
    lin = _setup_linear(D_IN, D_OUT, True, WeightInitType.Xavier, device)

    x = torch.randn(B, D_IN, device=device, requires_grad=True)
    out = lin(x).sum()
    out.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in lin.parameters()
    )


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
def test_linear_invalid_shape_raises(device):
    D_IN, D_OUT = 3, 2
    lin = _setup_linear(D_IN, D_OUT, False, WeightInitType.Xavier, device)

    bad_input = torch.randn(5, D_IN + 1, device=device)  # wrong last dim
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        _ = lin(bad_input)

    too_many_dims = torch.randn(2, 3, 4, 5, device=device)  # 4-D
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        _ = lin(too_many_dims)
