import torch
from src.transformer_encoder import TransformerEncoderBlock
from src.common import get_default_configs


def test_encoder_block_shape(device):
    cfg = get_default_configs()
    block = TransformerEncoderBlock(
        d_model=64, n_head=8, d_ff=128, dropout=0.0, configs=cfg
    ).to(device)
    B, L, D = 2, 5, 64
    x = torch.randn(B, L, D, device=device)
    out = block(x)
    assert out.shape == (B, L, D)
