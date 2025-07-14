"""
danielsinkin97@gmail.com

tests/test_relu.py
"""

import torch
import pytest
from src.activations import ReLU, IdentityActivation, Sigmoid


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


def test_identity_activation_backward():
    x = torch.randn(4, requires_grad=True)
    act = IdentityActivation()
    y = act(x).sum()
    y.backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("shape", [(6,), (2, 3), (4, 1, 5)])
def test_sigmoid_forward_matches_torch(shape):
    """Custom Sigmoid should produce identical outputs to torch.sigmoid."""
    x = torch.randn(*shape)
    sig_custom = Sigmoid()
    out_custom = sig_custom(x)
    out_torch = torch.sigmoid(x)
    torch.testing.assert_close(out_custom, out_torch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sigmoid_backward_matches_torch(dtype):
    """
    Gradient from custom Sigmoid must match torch.sigmoid.

    We compare ∂L/∂x where L = sum(sigmoid(x)).
    """
    # Create two independent copies with gradient tracking
    x_custom = torch.randn(8, dtype=dtype, requires_grad=True)
    x_torch = x_custom.clone().detach().requires_grad_()

    # Forward
    y_custom = Sigmoid()(x_custom).sum()
    y_torch = torch.sigmoid(x_torch).sum()

    # Backward
    y_custom.backward()
    y_torch.backward()

    torch.testing.assert_close(x_custom.grad, x_torch.grad)


def test_sigmoid_shape_dtype_device_preserved():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    x = torch.randn(3, 4, device=device, dtype=dtype)
    sig = Sigmoid().to(device)

    out = sig(x)

    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == device
