"""
danielsinkin97@gmail.com

relu.py
"""

from abc import abstractmethod
import math
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.autograd

from .common import erf


class ActivationFunction(nn.Module):
    """Abstract class for Actication Functions, can't mark it ABC due to nn.module inheritance"""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(ActivationFunction):
    """ReLU activation Function"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0.0, x, x.new_zeros(tuple()))


class GeLU(ActivationFunction):
    """GeLU activation Function"""

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class IdentityActivation(ActivationFunction):
    """NOP activation function"""

    def forward(self, x: Tensor) -> Tensor:
        return x


class _SigmoidFunction(torch.autograd.Function):  # pylint: disable=abstract-method
    """
    Sigmoid Function that implements the backprop trick to avoid recomputing sigmoid
    because sigma'(x) = sigma(x) * (1 - sigma(x))
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        y = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor]:
        (y,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        return (grad_output * y * (1.0 - y),)


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""

    def forward(self, x: Tensor) -> Tensor:
        return _SigmoidFunction.apply(x)  # type: ignore


class ActivationName(StrEnum):
    """ActivationName"""

    Identity = "Identity"
    ReLU = "ReLU"
    GeLU = "GeLU"
    Sigmoid = "Sigmoid"


def get_activation(activation: ActivationName) -> type[ActivationFunction]:
    """Access activations by name."""
    match activation:
        case ActivationName.ReLU:
            return ReLU
        case ActivationName.GeLU:
            return GeLU
        case ActivationName.Identity:
            return IdentityActivation
        case ActivationName.Sigmoid:
            return Sigmoid
        case _:
            raise ValueError(f"Unsupported activation: {activation}")
