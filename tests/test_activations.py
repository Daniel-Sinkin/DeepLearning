"""
danielsinkin97@gmail.com

tests/test_relu.py
"""

import torch
import pytest
from src.relu import ReLU  # adjust path if needed


def test_relu_positive_pass_through():
    x = torch.tensor([0.5, 1.0, 3.14])
    relu = ReLU()
    out = relu(x)
    torch.testing.assert_close(out, x)


def test_relu_negative_zeroed_out():
    x = torch.tensor([-1.0, -0.1, 0.0, 2.0])
    relu = ReLU()
    out = relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 2.0])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("shape", [(3,), (2, 4), (1, 3, 5)])
def test_relu_shape_preserved(shape):
    x = torch.randn(*shape)
    relu = ReLU()
    out = relu(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relu_dtype_preserved(dtype):
    x = torch.randn(5, dtype=dtype)
    relu = ReLU()
    out = relu(x)
    assert out.dtype == dtype


def test_relu_device_consistency(device):
    x = torch.randn(3, device=device)
    relu = ReLU().to(device)
    out = relu(x)
    assert out.device == x.device


def test_relu_all_negative_becomes_zero():
    x = -torch.rand(10)
    relu = ReLU()
    out = relu(x)
    assert torch.all(out == 0)


def test_relu_gradients_flow():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    relu = ReLU()
    y = relu(x).sum()
    y.backward()
    # Only positive input (2.0) has nonzero grad
    expected_grad = torch.tensor([0.0, 0.0, 1.0])
    torch.testing.assert_close(x.grad, expected_grad)
