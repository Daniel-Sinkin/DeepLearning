# src/positional_encoding.py
"""Sinusoidal positional encoding (Vaswani et al., 2017)"""

import math
import torch
from torch import nn, Tensor

from .common import Debug, Configs


class PositionalEncoding(nn.Module):
    """
    Injects position information into a sequence of embeddings via the
    fixed sinusoidal scheme described in §3.5 of Vaswani et al. (2017).
    """

    def __init__(
        self, d_model: int, dropout: float, configs: Configs, max_len: int = 32_768
    ):
        super().__init__()  # type: ignore
        self.configs = configs

        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        if Debug.asserts_enabled:
            assert d_model > 0, f"{d_model=}, must be > 0"
            assert max_len > 0, f"{max_len=}, must be > 0"
            assert d_model % 2 == 0, f"{d_model=}, must be even to interleave sin/cos"

        pos = torch.arange(max_len).unsqueeze(1)
        even_i = torch.arange(0, d_model, 2)

        angles = pos / (10000.0 ** (even_i / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, L, D)
        """
        _, L, D = x.shape
        if Debug.asserts_enabled:
            assert x.ndim == 3, f"expected 3D input (B, L, D), got {x.shape=}"
            assert (
                D == self.pe.shape[-1]  # type: ignore
            ), f"{D=} != {self.pe.shape[-1]=}"  # type: ignore
            assert (
                L <= self.pe.shape[1]  # type: ignore
            ), f"{L=} exceeds max_len={self.pe.shape[1]}"  # type:ignore

        x = x * self.scale + self.pe[:, :L, :].to(dtype=x.dtype)  # type: ignore
        return self.dropout(x)
