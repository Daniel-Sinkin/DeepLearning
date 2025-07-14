"""danielsinkin97@gmail.com"""

from typing import cast

from torch import Tensor
from torch import nn

from .multi_head_attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from .common import Configs


class TransformerDecoderBlock(nn.Module):
    """Pre-norm Transformer block (MHSA -> FFN) with residual connections"""

    def __init__(
        self, d_model: int, n_head: int, d_ff: int, dropout: float, configs: Configs
    ):
        super().__init__()  # type: ignore

        self.configs = configs

        self.ln_cross_attn = nn.LayerNorm(d_model, eps=self.configs.norm_eps)
        self.ln_self_attn = nn.LayerNorm(d_model, eps=self.configs.norm_eps)
        self.ln_ff = nn.LayerNorm(d_model, eps=self.configs.norm_eps)

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_head,
            is_causal=True,
            dropout=dropout,
            configs=self.configs,
        )
        self.cross_attn = MultiHeadCrossAttention(
            d_model=d_model, n_head=n_head, dropout=dropout, configs=self.configs
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Apply MHSA and position-wise FFN with residual connections.
        Normalization strategy depends on Configs.use_post_norm.
        """
        if self.configs.use_post_norm:
            x = x + cast(
                Tensor, self.self_attn(x, key_padding_mask=target_key_padding_mask)
            )
            x = self.ln_self_attn(x)

            x = x + self.cross_attn(
                x, encoder_output, key_padding_mask=memory_key_padding_mask
            )
            x = self.ln_cross_attn(x)

            x = x + cast(Tensor, self.feed_forward(x))
            x = self.ln_ff(x)
        else:
            x_norm: Tensor = self.ln_self_attn(x)
            x = x + cast(
                Tensor, self.self_attn(x_norm, key_padding_mask=target_key_padding_mask)
            )

            x_norm = self.ln_cross_attn(x)
            x = x + self.cross_attn(
                x_norm, encoder_output, key_padding_mask=memory_key_padding_mask
            )

            x_norm = self.ln_ff(x)
            x = x + cast(Tensor, self.feed_forward(x_norm))

        return x
