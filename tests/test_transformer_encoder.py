"""
danielsinkin97@gmail.com

tests/test_transformer_encoder.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from src.common import get_default_configs
from src.transformer_encoder import TransformerEncoderBlock


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cfg_with_post_norm(original_cfg: Any, use_post_norm: bool):
    """Return a copy of *original_cfg* with `use_post_norm` toggled."""
    if hasattr(original_cfg, "use_post_norm"):
        try:
            return replace(original_cfg, use_post_norm=use_post_norm)
        except TypeError:
            cfg = original_cfg
            cfg.use_post_norm = use_post_norm
            return cfg
    return original_cfg


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize("use_post_norm", [True, False])
@pytest.mark.parametrize("d_model,n_head,d_ff", [(32, 4, 64), (64, 8, 128)])
def test_encoder_forward_backward(device, use_post_norm, d_model, n_head, d_ff):
    """Forward pass produces correct shape and backward pass yields gradients."""
    cfg = _cfg_with_post_norm(get_default_configs(), use_post_norm)

    block = TransformerEncoderBlock(
        d_model=d_model,
        n_head=n_head,
        d_ff=d_ff,
        dropout=0.0,
        configs=cfg,
    ).to(device=device)

    B, L = 3, 11
    x = torch.randn(B, L, d_model, device=device, requires_grad=True)

    out = block(x)
    assert out.shape == (B, L, d_model)

    out.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all() and x.grad.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_encoder_dtype_device_consistency(device, dtype):
    """Encoder should preserve dtype and device."""
    if dtype is torch.float16 and device == "cpu":
        pytest.skip("float16 unsupported on CPU by PyTorch attention kernels")

    cfg = get_default_configs()
    d_model = 48

    encoder = TransformerEncoderBlock(
        d_model=d_model,
        n_head=6,
        d_ff=192,
        dropout=0.0,
        configs=cfg,
    ).to(device=device, dtype=dtype)

    encoder.eval()

    B, L = 2, 10
    src = torch.randn(B, L, d_model, device=device, dtype=dtype)
    enc_out = encoder(src)

    assert enc_out.dtype == dtype and enc_out.device == src.device
