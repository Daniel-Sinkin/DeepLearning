from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .common import assert_shape


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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        assert t.ndim == 1
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=t.device)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        assert_shape(emb, (t.shape[0], self.dim))
        return emb


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()

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

        # Up path (reverse channel order)
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
