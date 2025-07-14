# src/positional_encoding.py
"""Sinusoidal positional encoding (Vaswani et al., 2017)"""

from __future__ import annotations
import math
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """
    Injects position information into a sequence of embeddings via the
    fixed sinusoidal scheme described in §3.5 of Vaswani et al. (2017).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()  # type: ignore
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute the table: (1, max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        even_i = torch.arange(0, d_model, 2)  # (d_model/2)
        angles = pos / (10000.0 ** (even_i / d_model))  # broadcasting
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)  # even indices
        pe[:, 1::2] = torch.cos(angles)  # odd  indices
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, L, D)
        """
        L = x.size(1)
        x = x * self.scale + self.pe[:, :L, :].to(dtype=x.dtype)  # type: ignore
        return self.dropout(x)
