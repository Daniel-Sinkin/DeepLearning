from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .res_block import ResBlock
from ..common import assert_shape


class BottleneckBlock(nn.Module):
    """Simple bottleneck (can be extended with attention)."""

    def __init__(self, channels: int, time_emb_dim: int, depth: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        blocks = [ResBlock(channels, channels, time_emb_dim) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        if x.dim() != 4:
            raise AssertionError("Bottleneck input must be 4-D")
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x
