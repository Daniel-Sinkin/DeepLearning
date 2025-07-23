from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from ..common import assert_shape


class ResBlock(nn.Module):
    """Time-conditioned residual block.

    Projects time embedding, adds it after first conv, applies activation, then second conv.
    Adjusts residual path with 1x1 conv if channel dims differ.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.residual_conv: nn.Module
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        # Shape checks
        if x.dim() != 4:
            raise AssertionError(f"x must be 4D (B,C,H,W) got {x.dim()}D")
        B, _, H, W = x.shape
        assert_shape(t_emb, (B, self.time_emb_dim))

        h = self.conv1(x)
        # conv1 output shape check
        assert_shape(h, (B, self.conv2.in_channels, H, W))

        t = self.time_proj(t_emb)  # (B, out_channels)
        assert_shape(t, (B, self.conv2.in_channels))
        t = t[:, :, None, None]
        h = h + t
        h = self.activation(h)
        h = self.conv2(h)
        assert_shape(h, (B, self.conv2.out_channels, H, W))

        r = self.residual_conv(x)
        assert_shape(r, (B, self.conv2.out_channels, H, W))
        out = h + r
        assert_shape(out, (B, self.conv2.out_channels, H, W))
        return out
