"""
danielsinkin97@gmail.com

relu.py
"""

from abc import abstractmethod
import math
from enum import StrEnum

import torch
from torch import Tensor
from torch import nn

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


class ActivationName(StrEnum):
    """ActivationName"""

    Identity = "Identity"
    ReLU = "ReLU"
    GeLU = "GeLU"


def get_activation(activation: ActivationName) -> type[ActivationFunction]:
    """Access activations by name."""
    match activation:
        case ActivationName.ReLU:
            return ReLU
        case ActivationName.GeLU:
            return GeLU
        case ActivationName.Identity:
            return IdentityActivation
        case _:
            raise ValueError(f"Unsupported activation: {activation}")
