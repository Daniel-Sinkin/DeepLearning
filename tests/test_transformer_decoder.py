"""
danielsinkin97@gmail.com

tests/test_transformer_decoder.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from src.common import get_default_configs
from src.transformer_decoder import TransformerDecoderBlock


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
@pytest.mark.parametrize("dropout", [0.0, 0.3])
@pytest.mark.parametrize("use_post_norm", [True, False])
def test_decoder_forward_mask_and_dropout(device, dropout, use_post_norm):
    """Checks masking, dropout effect, and backward gradients."""
    cfg = _cfg_with_post_norm(get_default_configs(), use_post_norm)

    d_model, n_head, d_ff = 64, 8, 256
    block = TransformerDecoderBlock(
        d_model=d_model,
        n_head=n_head,
        d_ff=d_ff,
        dropout=dropout,
        configs=cfg,
    ).to(device=device)

    B, Ltgt, Lsrc = 2, 7, 9
    tgt = torch.randn(B, Ltgt, d_model, device=device, requires_grad=True)
    src = torch.randn(B, Lsrc, d_model, device=device)

    tgt_pad_mask = torch.zeros(B, Ltgt, dtype=torch.bool, device=device)
    tgt_pad_mask[0, -1] = True

    mem_pad_mask = torch.zeros(B, Lsrc, dtype=torch.bool, device=device)
    mem_pad_mask[1, :3] = True

    block.train()
    out_train = block(
        tgt,
        src,
        target_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=mem_pad_mask,
    )
    assert out_train.shape == (B, Ltgt, d_model)

    block.eval()
    with torch.no_grad():
        out_eval = block(
            tgt,
            src,
            target_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )

    if dropout > 0.0:
        assert not torch.allclose(out_train, out_eval)

    # Only backprop from train-mode output
    out_train.mean().backward()
    assert tgt.grad is not None and torch.isfinite(tgt.grad).all()


def test_decoder_mismatched_mask_raises(device):
    """Supplying a wrongly-shaped mask should raise at runtime."""
    cfg = get_default_configs()

    block = TransformerDecoderBlock(
        d_model=32,
        n_head=4,
        d_ff=64,
        dropout=0.0,
        configs=cfg,
    ).to(device=device)

    B, Ltgt, Lsrc = 2, 5, 5
    tgt = torch.randn(B, Ltgt, 32, device=device)
    src = torch.randn(B, Lsrc, 32, device=device)

    bad_tgt_mask = torch.zeros(B, Ltgt + 1, dtype=torch.bool, device=device)

    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        _ = block(tgt, src, target_key_padding_mask=bad_tgt_mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_decoder_dtype_device_consistency(device, dtype):
    """Decoder should preserve dtype and device."""
    if dtype is torch.float16 and device == "cpu":
        pytest.skip("float16 unsupported on CPU by PyTorch attention kernels")

    cfg = get_default_configs()
    d_model = 48

    decoder = TransformerDecoderBlock(
        d_model=d_model,
        n_head=6,
        d_ff=192,
        dropout=0.0,
        configs=cfg,
    ).to(device=device, dtype=dtype)

    decoder.eval()

    B, Ltgt, Lsrc = 1, 4, 10
    src = torch.randn(B, Lsrc, d_model, device=device, dtype=dtype)
    tgt = torch.randn(B, Ltgt, d_model, device=device, dtype=dtype)

    dec_out = decoder(tgt, src)

    assert dec_out.dtype == dtype and dec_out.device == tgt.device
