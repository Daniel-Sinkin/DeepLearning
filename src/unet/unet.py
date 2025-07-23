from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .time_embedding import SinusoidalTimeEmbedding
from .down_block import DownBlock
from .bottleneck_block import BottleneckBlock
from .up_block import UpBlock
from ..common import assert_shape


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()  # type: ignore

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        # Down path
        in_ch = base_channels
        self.downs = nn.ModuleList()
        self.skips_channels = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim))
            self.skips_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = BottleneckBlock(in_ch, time_emb_dim)

        self.ups = nn.ModuleList()
        for mult, skip_ch in zip(
            reversed(channel_mults), reversed(self.skips_channels)
        ):
            out_ch = base_channels * mult
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))
            in_ch = out_ch  # update for next up

        self.output_conv = nn.Sequential(nn.Conv2d(in_ch, in_channels, kernel_size=1))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time tensor of shape (B,)
        Returns:
            Predicted noise (B, C, H, W)
        """
        assert_shape(t, (x.shape[0],))
        t_emb = self.time_mlp(self.time_embedding(t))

        h = self.input_conv(x)

        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)

        h = self.bottleneck(h, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb)

        return self.output_conv(h)
