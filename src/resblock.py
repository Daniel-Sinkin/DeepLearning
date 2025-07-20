import torch
from jaxtyping import Float, Int
from torch import Tensor, is_deterministic_algorithms_warn_only_enabled, nn
from torchgen.api.ufunc import kernel_name

from .common import assert_shape
from .dataset_cifar import CIFARImages

T_ = "T"
B_ = "B"


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()  # type: ignore
        self.time_emb_dim = time_emb_dim
        self.time_proj = nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, out_channels))

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.ReLU()
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        batch, _, _, _ = x.shape
        assert_shape(t_emb, (batch, self.time_emb_dim))
        h = self.conv1(x)
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t
        h = self.activation(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)
