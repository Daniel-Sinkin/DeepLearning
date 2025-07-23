from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .res_block import ResBlock
from ..common import assert_shape


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
