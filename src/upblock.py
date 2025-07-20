import torch
from torch import Tensor, nn

from src.resblock import ResBlock


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        """
        Args:
            in_channels: channels in the upsampled tensor (from previous up block)
            skip_channels: channels in the skip connection
            out_channels: output channels of the ResBlock
            time_emb_dim: time embedding dimension
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )

        self.res_block = ResBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x: Tensor, skip: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: tensor from upsampling path, shape (B, in_channels, H, W)
            skip: skip connection tensor from downsampling path
            t_emb: time embedding tensor, shape (B, time_emb_dim)

        Returns:
            Output tensor after upsampling and residual processing.
        """
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            # Resize if needed due to rounding errors in spatial dimensions
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        return self.res_block(x, t_emb)
