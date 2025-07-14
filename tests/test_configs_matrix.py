"""
Exhaustive config-matrix test-suite
----------------------------------

• Builds a tiny encoder block *and* a tiny full Transformer for every
  combination of the 5 boolean flags in `src.common.Configs`.

• Checks both *semantic* behaviour of each flag and basic fwd/bwd sanity.

Runtime: ≈ 1-2 s on a laptop CPU.
"""

from __future__ import annotations
import itertools
from typing import Dict, Any

import pytest
import torch

from src.common import Configs
from src.multi_head_attention import MultiHeadSelfAttention
from src.transformer_encoder import TransformerEncoderBlock
from src.transformer import Transformer


_BOOL_FIELDS = (
    "use_fused_qkv",
    "use_post_norm",
    "use_final_layer_norm",
    "use_original_init",
    "tie_target_embedding_and_lm_head_weights",
)


def _make_cfg(**overrides: Any) -> Configs:
    base = Configs(
        use_fused_qkv=True,
        use_post_norm=True,
        use_final_layer_norm=False,
        use_original_init=True,
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )
    return base.__class__(**{**base.__dict__, **overrides})


_BOOL_COMBOS = [
    dict(zip(_BOOL_FIELDS, combo))
    for combo in itertools.product([False, True], repeat=len(_BOOL_FIELDS))
]


def _assert_flag_behaviour(cfg: Configs, model: Transformer) -> None:
    mhsa: MultiHeadSelfAttention = model.encoder[0].mhsa  # type: ignore
    if cfg.use_fused_qkv:
        assert mhsa.W_QKV is not None
        assert all(getattr(mhsa, name) is None for name in ("W_Q", "W_K", "W_V"))
    else:
        assert mhsa.W_QKV is None
        assert all(getattr(mhsa, name) is not None for name in ("W_Q", "W_K", "W_V"))

    tied = cfg.tie_target_embedding_and_lm_head_weights
    same_obj = model.lm_head.weight is model.target_embedding.weight
    assert tied == same_obj

    has_final_ln = model.ln_final is not None
    assert cfg.use_final_layer_norm == has_final_ln

    lin_bias = model.encoder[0].feed_forward[0].bias  # type: ignore
    assert lin_bias is not None
    if cfg.use_original_init:
        assert torch.allclose(lin_bias, torch.zeros_like(lin_bias), atol=1e-6)
    else:
        assert not torch.allclose(lin_bias, torch.zeros_like(lin_bias), atol=1e-3)

    block_pre = TransformerEncoderBlock(
        d_model=16,
        n_head=4,
        d_ff=32,
        dropout=0.0,
        configs=_make_cfg(**{**cfg.__dict__, "use_post_norm": False}),
    ).eval()
    block_post = TransformerEncoderBlock(
        d_model=16,
        n_head=4,
        d_ff=32,
        dropout=0.0,
        configs=_make_cfg(**{**cfg.__dict__, "use_post_norm": True}),
    ).eval()
    x = torch.randn(1, 3, 16)
    out_pre = block_pre(x)
    out_post = block_post(x)
    assert not torch.allclose(out_pre, out_post)


@pytest.mark.parametrize(
    "overrides",
    _BOOL_COMBOS,
    ids=lambda d: ",".join(f"{k}={int(v)}" for k, v in d.items()),
)
def test_config_combo_forward_backward_and_behaviour(
    overrides: Dict[str, bool], device: torch.device
) -> None:
    cfg = _make_cfg(**overrides)

    model = Transformer(
        d_model=16,
        n_head=4,
        d_ff=32,
        n_layer=1,
        dropout=0.0,
        source_vocab_size=101,
        target_vocab_size=101,
        pad_id=0,
        configs=cfg,
    ).to(device)
    model.eval()

    src = torch.randint(0, 101, (2, 5), device=device)
    tgt = torch.randint(0, 101, (2, 4), device=device)

    logits = model(src, tgt)
    assert logits.shape == (2, 4, 101)

    logits.mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    _assert_flag_behaviour(cfg, model)
