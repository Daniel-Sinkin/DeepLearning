from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .res_block import ResBlock
from ..common import assert_shape


class UpBlock(nn.Module):
    """An upsampling block.

    1. Upsample via transposed conv (stride 2)
    2. (Optional) Interpolate for shape alignment
    3. Concatenate with skip features
    4. Fuse via ResBlock
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.res_block = ResBlock(
            in_channels + skip_channels, out_channels, time_emb_dim
        )

    def forward(self, x: Tensor, skip: Tensor, t_emb: Tensor) -> Tensor:
        if x.dim() != 4 or skip.dim() != 4:
            raise AssertionError("x and skip must be 4-D (B,C,H,W)")
        B, Cx, H, W = x.shape
        Bs, Cs, Hs, Ws = skip.shape
        if B != Bs:
            raise AssertionError("Batch size mismatch between x and skip")

        x_up = self.upsample(x)  # (B, Cx, 2H, 2W)
        # If shapes don't align (odd dims), interpolate
        if x_up.shape[-2:] != (Hs, Ws):
            x_up = nn.functional.interpolate(x_up, size=(Hs, Ws), mode="nearest")
        assert_shape(x_up, (B, Cx, Hs, Ws))
        assert_shape(skip, (B, Cs, Hs, Ws))

        fused = torch.cat([x_up, skip], dim=1)
        assert_shape(fused, (B, Cx + Cs, Hs, Ws))
        out = self.res_block(fused, t_emb)
        # res_block ensures output channel shape internally.
        return out
