"""
danielsinkin97@gmail.com

relu.py
"""

import math

import torch
from torch import Tensor
from torch import nn

from .common import erf


class ReLU(nn.Module):
    """ReLU activation Function"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0.0, x, x.new_zeros(tuple()))


class GeLU(nn.Module):
    """GeLU activation Function"""

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
