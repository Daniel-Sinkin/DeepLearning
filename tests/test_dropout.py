import torch
import pytest
from torch import nn

from src.dropout import Dropout


def test_dropout_zero_prob():
    """p=0 should return input unchanged in train mode"""
    x = torch.randn(10, 10, requires_grad=True)
    dropout = Dropout(p=0.0)
    dropout.train()
    out = dropout(x)
    torch.testing.assert_close(out, x)


def test_dropout_one_prob():
    """p=1 should return all zeros in train mode"""
    x = torch.randn(4, 5)
    dropout = Dropout(p=1.0)
    dropout.train()
    out = dropout(x)
    assert torch.all(out == 0)


def test_dropout_eval_mode():
    """Dropout should be a no-op in eval mode regardless of p"""
    x = torch.randn(10, 10)
    for p in [0.0, 0.3, 1.0]:
        dropout = Dropout(p=p)
        dropout.eval()
        out = dropout(x)
        torch.testing.assert_close(out, x)


def test_dropout_train_masking_and_scaling():
    """Verify dropout masking and rescaling when training"""
    torch.manual_seed(42)
    x = torch.ones(10000)
    p = 0.2
    dropout = Dropout(p=p)
    dropout.train()
    out = dropout(x)

    # Expect approx (1-p)*N non-zero entries scaled by 1/(1-p)
    num_nonzero = out.nonzero().size(0)
    expected_nonzero = int((1 - p) * x.numel())
    assert abs(num_nonzero - expected_nonzero) < 0.05 * x.numel()

    # Mean should be close to 1
    mean = out.mean().item()
    assert abs(mean - 1.0) < 0.05


def test_dropout_backward_pass():
    """Gradients should flow through the surviving elements"""
    x = torch.ones(10, requires_grad=True)
    dropout = Dropout(p=0.5)
    dropout.train()

    out = dropout(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert (x.grad == 0).sum() + (
        x.grad == 2
    ).sum() == 10  # scaled grad is 1/(1 - 0.5) = 2


def test_dropout_invalid_prob():
    """Invalid dropout probabilities should raise"""
    with pytest.raises(ValueError):
        _ = Dropout(p=-0.1)

    with pytest.raises(ValueError):
        _ = Dropout(p=1.1)
