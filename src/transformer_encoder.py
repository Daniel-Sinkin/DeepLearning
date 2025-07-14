"""
danielsinkin97@gmail.com

transformer_encoder.py
"""

from typing import cast

from torch import Tensor
from torch import nn

from src.dropout import Dropout

from .multi_head_attention import MultiHeadSelfAttention
from .common import Configs

from .linear import Linear


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block (MHSA -> FFN) with residual connections"""

    def __init__(
        self, d_model: int, n_head: int, d_ff: int, dropout: float, configs: Configs
    ):
        super().__init__()  # type: ignore

        self.configs = configs

        self.ln_mhsa = nn.LayerNorm(d_model, eps=self.configs.norm_eps)
        self.ln_ff = nn.LayerNorm(d_model, eps=self.configs.norm_eps)

        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_head,
            is_causal=False,
            dropout=dropout,
            configs=self.configs,
        )
        self.feed_forward = nn.Sequential(
            Linear(d_model, d_ff, bias=True, configs=self.configs),
            nn.ReLU(),
            Dropout(dropout),
            Linear(d_ff, d_model, bias=True, configs=self.configs),
            Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        if Configs.use_post_norm:
            x = x + cast(Tensor, self.mhsa(x, key_padding_mask=key_padding_mask))
            x = self.ln_mhsa(x)

            x = x + cast(Tensor, self.feed_forward(x))
            x = self.ln_ff(x)
        else:
            x_norm: Tensor = self.ln_mhsa(x)
            x = x + cast(Tensor, self.mhsa(x_norm, key_padding_mask=key_padding_mask))

            x_norm = self.ln_ff(x)
            x = x + cast(Tensor, self.feed_forward(x_norm))

        return x
