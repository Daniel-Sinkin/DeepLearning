"""
danielsinkin97@gmail.com

linear.py
"""

import torch
from torch import nn


class Dropout(nn.Module):
    """Dropout is not active when testing"""

    def __init__(self, p: float):
        super().__init__()  # type: ignore

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)

        mask = (torch.rand_like(x) > self.p).to(dtype=x.dtype)
        return x * mask / (1.0 - self.p)
