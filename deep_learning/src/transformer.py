"""danielsinkin97@gmail.com"""

from typing import cast

from torch import Tensor
from torch import nn

from .common import Configs
from .transformer_encoder import TransformerEncoderBlock
from .transformer_decoder import TransformerDecoderBlock


class Transformer(nn.Module):
    """Stack of TransformerBlock for the encoder."""

    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        d_ff: int = 2048,
        n_layer: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()  # type: ignore

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff

        self.dropout = dropout

        self.encoder = nn.Sequential(
            *(
                TransformerEncoderBlock(
                    d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layer)
            )
        )

        self.decoder = nn.ModuleList(
            (
                TransformerDecoderBlock(
                    d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layer)
            )
        )

        if Configs.use_final_layer_norm:
            self.ln_final = nn.LayerNorm(d_model)
        else:
            self.ln_final = None

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """Run the input through every Transformer block in sequence."""
        encoder_out = cast(Tensor, self.encoder(source))
        decoder_out = target

        for block in self.decoder:
            decoder_out = block(decoder_out, encoder_out)

        if Configs.use_final_layer_norm:
            assert self.ln_final is not None
            decoder_out = self.ln_final(decoder_out)
        return decoder_out
