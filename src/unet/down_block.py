from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .res_block import ResBlock
from ..common import assert_shape


class DownBlock(nn.Module):
    """A downsampling block consisting of a ResBlock followed by stride-2 conv.

    Returns (downsampled, skip) where `skip` is the pre-downsample feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() != 4:
            raise AssertionError("Input to DownBlock must be 4-D")
        B, _, H, W = x.shape
        h = self.res_block(x, t_emb)
        assert_shape(h, (B, self.downsample.in_channels, H, W))
        d = self.downsample(h)
        assert_shape(d, (B, self.downsample.out_channels, H // 2, W // 2))
        return d, h
