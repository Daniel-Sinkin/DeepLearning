import torch
from src.transformer_decoder import TransformerDecoderBlock
from src.common import get_default_configs


def test_decoder_block_shape(device):
    cfg = get_default_configs()
    block = TransformerDecoderBlock(
        d_model=64, n_head=8, d_ff=128, dropout=0.0, configs=cfg
    ).to(device)
    B, Ltgt, Lsrc, D = 2, 6, 7, 64
    tgt = torch.randn(B, Ltgt, D, device=device)
    enc = torch.randn(B, Lsrc, D, device=device)
    out = block(tgt, enc)
    assert out.shape == (B, Ltgt, D)
