import torch
from torch import Tensor, nn

from src.resblock import ResBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor of shape (B, C, H, W)
            t_emb: time embedding of shape (B, time_emb_dim)

        Returns:
            - downsampled output (B, out_channels, H/2, W/2)
            - skip connection output (B, out_channels, H, W)
        """
        h = self.res_block(x, t_emb)
        down = self.downsample(h)
        return down, h
