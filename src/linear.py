"""
danielsinkin97@gmail.com

linear.py
"""

import math

import torch
from torch import Tensor
from torch import nn

from .common import WeightInitType, Configs, assert_shape


class Linear(nn.Module):
    """Matrix Product"""

    def __init__(self, d_in: int, d_out: int, bias: bool, configs: Configs):
        super().__init__()  # type: ignore

        self.configs = configs

        self.W = nn.Parameter(torch.empty(d_out, d_in))

        if self.configs.weight_init_type == WeightInitType.Xavier:
            # Vaswani 2017 uses Xavier uniform
            nn.init.xavier_uniform_(self.W)  # gain=1 is default (no nonlinearity)
        elif self.configs.weight_init_type == WeightInitType.Kaiming:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        else:
            raise RuntimeError(
                f"Unsupported weight_init_type: {self.configs.weight_init_type}"
            )

        if bias:
            self.b = nn.Parameter(torch.empty(d_out))
            bound = 1 / math.sqrt(d_in)
            nn.init.uniform_(self.b, -bound, bound)
        else:
            self.b = None

    def forward(self, X: Tensor) -> Tensor:
        d_out, d_in = self.W.shape

        if X.ndim == 2:
            batch, _ = X.shape
            assert_shape(X, (batch, d_in))
            output = torch.matmul(X, self.W.T)
            assert_shape(output, (batch, d_out))
        elif X.ndim == 3:
            batch, len_seq, _ = X.shape
            assert_shape(X, (batch, len_seq, d_in))
            output = torch.matmul(X, self.W.T)
            assert_shape(output, (batch, len_seq, d_out))
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

        if self.b is not None:
            output = output + self.b

        return output
