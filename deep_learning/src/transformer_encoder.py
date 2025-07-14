"""danielsinkin97@gmail.com"""

from typing import cast

from torch import Tensor
from torch import nn

from .multi_head_attention import MultiHeadSelfAttention
from .common import Configs


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block (MHSA -> FFN) with residual connections"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()  # type: ignore

        self.ln_mhsa = nn.LayerNorm(d_model, eps=Configs.norm_eps)
        self.ln_ff = nn.LayerNorm(d_model, eps=Configs.norm_eps)

        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model, n_head=n_head, is_causal=False, dropout=dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Apply MHSA and position-wise FFN with residual connections.
        Normalization strategy depends on Configs.use_post_norm.
        """
        if Configs.use_post_norm:
            x = x + cast(Tensor, self.mhsa(x, key_padding_mask=key_padding_mask))
            x = self.ln_mhsa(x)

            x = x + cast(Tensor, self.feed_forward(x))
            x = self.ln_ff(x)
        else:
            x_norm = self.ln_mhsa(x)
            x = x + cast(Tensor, self.mhsa(x_norm, key_padding_mask=key_padding_mask))

            x_norm = self.ln_ff(x)
            x = x + cast(Tensor, self.feed_forward(x_norm))

        return x
