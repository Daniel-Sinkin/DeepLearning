"""
danielsinkin97@gmail.com

test_mlp.py
"""

import pytest
import torch
import torch.nn as nn

from src.mlp import MultiLayerPerceptron
from src.common import Configs, WeightInitType
from src.activations import ActivationName


@pytest.fixture
def default_configs():
    return Configs(weight_init_type=WeightInitType.Xavier)


def test_mlp_forward_pass(default_configs):
    mlp = MultiLayerPerceptron(
        d_in=16,
        d_out=4,
        d_hidden=[32, 64],
        bias=True,
        configs=default_configs,
        activation_name=ActivationName.ReLU,
    )

    x = torch.randn(8, 16)  # batch size 8, input dim 16
    out = mlp(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (8, 4)


def test_mlp_without_bias(default_configs):
    mlp = MultiLayerPerceptron(
        d_in=10,
        d_out=5,
        d_hidden=[20],
        bias=False,
        configs=default_configs,
        activation_name=ActivationName.GeLU,
    )

    x = torch.randn(3, 10)
    out = mlp(x)

    assert out.shape == (3, 5)


def test_mlp_identity_activation(default_configs):
    mlp = MultiLayerPerceptron(
        d_in=6,
        d_out=2,
        d_hidden=[12],
        bias=True,
        configs=default_configs,
        activation_name=ActivationName.Identity,
    )

    x = torch.randn(1, 6)
    out = mlp(x)

    assert out.shape == (1, 2)


def test_mlp_invalid_hidden_raises(default_configs):
    with pytest.raises(ValueError, match="requires at least one hidden layer"):
        MultiLayerPerceptron(
            d_in=8,
            d_out=2,
            d_hidden=[],
            bias=True,
            configs=default_configs,
            activation_name=ActivationName.ReLU,
        )
