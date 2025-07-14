"""
danielsinkin97@gmail.com

transformer.py
"""

from torch import Tensor
from torch import nn

from src.linear import Linear

from .common import Configs, WeightInitType, assert_shape, get_default_configs
from .transformer_encoder import TransformerEncoderBlock
from .transformer_decoder import TransformerDecoderBlock
from .positional_encoding import PositionalEncoding


def init_weights_original(m: nn.Module) -> None:
    """
    The 2017 Vasvani et al. Paper uses xavier with 0 init,
    pytorch uses kaiming with uniform bias by default
    """
    if isinstance(m, nn.Linear):
        _ = """
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # type: ignore
            nn.init.zeros_(m.bias)
        """
        raise RuntimeError("PyTorch Linear is deprecated use my linear.Linear instead ")


class Transformer(nn.Module):
    """Stack of TransformerBlock for the encoder."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        n_layer: int,
        dropout: float,
        source_vocab_size: int,
        target_vocab_size: int,
        pad_id: int = 0,
        configs: Configs | None = None,
    ):
        super().__init__()  # type: ignore

        if configs is None:
            self.configs = get_default_configs()
        else:
            self.configs = configs

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff

        self.dropout = dropout

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        self.source_embedding = nn.Embedding(
            self.source_vocab_size, d_model, padding_idx=pad_id
        )
        self.target_embedding = nn.Embedding(
            self.target_vocab_size, d_model, padding_idx=pad_id
        )
        self.source_positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, configs=self.configs
        )
        self.target_positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, configs=self.configs
        )

        self.lm_head = Linear(
            d_model, target_vocab_size, bias=False, configs=self.configs
        )
        if self.configs.tie_target_embedding_and_lm_head_weights:
            self.lm_head.weight = self.target_embedding.weight

        self.encoder = nn.ModuleList(
            TransformerEncoderBlock(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout,
                configs=self.configs,
            )
            for _ in range(n_layer)
        )

        self.decoder = nn.ModuleList(
            TransformerDecoderBlock(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout,
                configs=self.configs,
            )
            for _ in range(n_layer)
        )

        if self.configs.use_final_layer_norm:
            self.ln_final = nn.LayerNorm(d_model)
        else:
            self.ln_final = None

        if self.configs.weight_init_type == WeightInitType.Xavier:
            self.apply(init_weights_original)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_key_padding_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the input through every Transformer block in sequence."""
        batch, len_source = source.shape
        batch2, len_target = target.shape
        if self.configs.asserts_enabled:
            assert batch == batch2

        _source: Tensor = self.source_embedding(source)
        assert_shape(_source, (batch, len_source, self.d_model))
        _source = self.source_positional_encoding(_source)
        assert_shape(_source, (batch, len_source, self.d_model))

        _target: Tensor = self.target_embedding(target)
        assert_shape(_target, (batch, len_target, self.d_model))
        _target = self.target_positional_encoding(_target)
        assert_shape(_target, (batch, len_target, self.d_model))

        encoder_out = _source
        for block in self.encoder:
            encoder_out = block(encoder_out, key_padding_mask=source_key_padding_mask)

        decoder_out = _target
        for block in self.decoder:
            decoder_out: Tensor = block(
                decoder_out,
                encoder_out,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=source_key_padding_mask,
            )

        if self.configs.use_final_layer_norm:
            assert self.ln_final is not None
            decoder_out = self.ln_final(decoder_out)

        logits: Tensor = self.lm_head(decoder_out)
        assert_shape(logits, (batch, len_target, self.target_vocab_size))
        return logits
