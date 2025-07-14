"""
danielsinkin97@gmail.com

linear.py
"""

from typing import Type

from torch import Tensor
from torch import nn

from src.activations import ActivationFunction, get_activation, ActivationName

from .common import Configs
from .linear import Linear


class MultiLayerPerceptron(nn.Module):
    """Matrix Product"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: list[int],
        bias: bool,
        configs: Configs,
        activation_name: ActivationName = ActivationName.ReLU,
    ):
        super().__init__()  # type: ignore

        self.configs = configs

        if len(d_hidden) == 0:
            raise ValueError(
                "MultiLayerPerceptron requires at least one hidden layer; use Linear instead."
            )

        activation_func: Type[ActivationFunction] = get_activation(activation_name)
        layers: list[Linear | ActivationFunction] = [
            Linear(d_in, d_hidden[0], bias=bias, configs=self.configs),
            activation_func(),
        ]
        for h1, h2 in zip(d_hidden[:-1], d_hidden[1:]):
            layers.append(Linear(h1, h2, bias=bias, configs=self.configs))
            layers.append(activation_func())
        layers.append(Linear(d_hidden[-1], d_out, bias=bias, configs=self.configs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
