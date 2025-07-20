import torch
from torch import Tensor, nn

from src.resblock import ResBlock


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.block = ResBlock(
            in_channels=channels,
            out_channels=channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        return self.block(x, t_emb)
