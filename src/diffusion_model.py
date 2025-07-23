"""
danielsinkin97@gmail.com
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor, is_deterministic_algorithms_warn_only_enabled, nn
from torchgen.api.ufunc import kernel_name

from .common import assert_shape
from .dataset.dataset_cifar import CIFARImages
from .unet.unet import UNet

T_ = "T"
B_ = "B"


def get_beta_schedule_linear(
    T: int, beta_0: float = 1e-4, beta_T: float = 2e-2
) -> Float[Tensor, f"{T_}"]:
    return torch.linspace(beta_0, beta_T, T)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        T: int,
        beta_t: Tensor,
        embed_dim: int = 128,
        *,
        unet_in_channels: int = 3,
        unet_base_channels: int = 64,
        unet_channel_mults: tuple[int, ...] = (1, 2, 4),
        unet_time_emb_dim: int | None = None,
    ):
        super().__init__()
        self.T = T
        self.register_buffer("alpha_bars", torch.cumprod(1 - beta_t, dim=0))
        time_dim = unet_time_emb_dim or embed_dim
        self.model = UNet(
            in_channels=unet_in_channels,
            base_channels=unet_base_channels,
            time_emb_dim=time_dim,
            channel_mults=unet_channel_mults,
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.model(x_t, t)

    def train_step(self, x_0: Tensor) -> Tensor:
        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(0)
        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.T, (B,), device=device)
        alpha_bars = self.alpha_bars[t].reshape(B, 1, 1, 1)

        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bars) * x_0 + torch.sqrt(1 - alpha_bars) * eps
        eps_pred = self.forward(x_t, t)
        return nn.functional.mse_loss(eps_pred, eps)
