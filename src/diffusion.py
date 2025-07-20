"""
danielsinkin97@gmail.com
"""

import torch
from jaxtyping import Float
from torch import Tensor, nn

from .common import assert_shape
from .dataset_cifar import CIFARImages

T_ = "T"


def get_beta_schedule_linear(
    T: int, beta_0: float = 1e-4, beta_T: float = 1e-1
) -> Float[Tensor, f"{T_}"]:  # noqa: F821
    beta_t = torch.linspace(beta_0, beta_T, T)
    return torch.cumprod(1 - beta_t, dim=0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        assert len(t.shape) == 1
        batch = t.shape
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=t.device)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        assert_shape(emb, (batch, self.dim))
        return emb


class DiffusionModel(nn.Module):
    def __init__(self, T: int, beta_t: Float[Tensor, f"{T_}"]):
        super().__init__()  # type: ignore
        assert_shape(beta_t, (T,))
        self.T = T
        self.alpha_bars = torch.cumprod(1 - beta_t, dim=0)

        self.epsilon_theta = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.epsilon_theta(x_t)

    def train_step(self, x_0: Tensor) -> Tensor:
        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.T, (B,), device=device)

        alpha_bars = self.alpha_bars[t].reshape(B, 1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bars) * x_0 + torch.sqrt(1.0 - alpha_bars) * eps
        eps_theta = self.forward(x_t, t)
        loss = nn.functional.mse_loss(eps_theta, eps)
        return loss
