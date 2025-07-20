"""
danielsinkin97@gmail.com
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from .common import assert_shape
from .dataset_cifar import CIFARImages, validate_batch

T_ = "T"
B_ = "B"


def get_beta_schedule_linear(
    T: int, beta_0: float = 1e-4, beta_T: float = 2e-2
) -> Float[Tensor, f"{T_}"]:
    return torch.linspace(beta_0, beta_T, T)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        assert len(t.shape) == 1
        batch = t.shape[0]
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
    def __init__(self, T: int, beta_t: Float[Tensor, f"{T_}"], embed_dim: int = 64):
        super().__init__()  # type: ignore
        self.embed_dim = embed_dim

        assert_shape(beta_t, (T,))
        self.T = T
        self.register_buffer("alpha_bars", torch.cumprod(1 - beta_t, dim=0))

        self.time_embedding = SinusoidalTimeEmbedding(dim=self.embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()
        )

        self.conv1 = nn.Conv2d(3, self.embed_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.embed_dim, 3, 3, padding=1)

    def forward(self, x_t: CIFARImages, t: Int[Tensor, f"{B_}"]) -> Tensor:
        batch, channel, height, width = x_t.shape
        t_emb = self.time_embedding(t)
        assert_shape(t_emb, (batch, self.embed_dim))
        batch, dim = t_emb.shape
        t_emb = self.time_mlp(t_emb)
        assert_shape(t_emb, (batch, dim))
        t_emb = t_emb[:, :, None, None]
        assert_shape(t_emb, (batch, dim, 1, 1))

        h = nn.functional.relu(self.conv1(x_t) + t_emb)
        out = self.conv2(h)
        return out

    def train_step(self, x_0: Tensor) -> Tensor:
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze()
            assert x_0.dim() == 4

        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.T, (B,), device=device)

        alpha_bars = self.alpha_bars[t].reshape(B, 1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bars) * x_0 + torch.sqrt(1.0 - alpha_bars) * eps
        eps_theta = self.forward(x_t, t)
        loss = nn.functional.mse_loss(eps_theta, eps)
        return loss
