import torch
from src.positional_encoding import PositionalEncoding
from src.common import get_default_configs


def test_positional_encoding_shape(device):
    d_model = 16
    B, L = 4, 11
    pe = PositionalEncoding(
        d_model=d_model, dropout=0.0, configs=get_default_configs()
    ).to(device)
    x = torch.zeros(B, L, d_model, device=device)
    out = pe(x)
    assert out.shape == (B, L, d_model)


def test_positional_encoding_is_deterministic(device):
    d_model = 8
    pe = PositionalEncoding(
        d_model=d_model, dropout=0.0, configs=get_default_configs()
    ).to(device)
    x = torch.randn(1, 6, d_model, device=device)
    y1 = pe(x.clone())
    y2 = pe(x.clone())
    torch.testing.assert_close(y1, y2)
