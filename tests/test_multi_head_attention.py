"""
danielsinkin97@gmail.com

test_multi_head_attention.py
"""

import pytest
import torch

from src.common import Configs
from src.multi_head_attention import MultiHeadSelfAttention


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("use_fused_qkv", [False, True])
def test_mhsa_forward_shapes(is_causal, use_fused_qkv, device):
    cfg = Configs(
        use_fused_qkv=use_fused_qkv,
        use_post_norm=True,
        use_final_layer_norm=False,
        use_original_init=True,
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )

    d_model, n_head = 32, 4
    attn = MultiHeadSelfAttention(
        is_causal=is_causal, d_model=d_model, n_head=n_head, dropout=0.0, configs=cfg
    ).to(device)

    B, L = 3, 9
    x = torch.randn(B, L, d_model, device=device)
    out = attn(x)
    assert out.shape == (B, L, d_model)


def test_fused_equals_unfused(device):
    D, H, B, L = 48, 6, 2, 5

    cfg_fused = Configs(
        use_fused_qkv=True,
        use_post_norm=True,
        use_final_layer_norm=False,
        use_original_init=True,
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )
    cfg_unfused = cfg_fused.__class__(**{**cfg_fused.__dict__, "use_fused_qkv": False})

    fused = MultiHeadSelfAttention(True, D, H, 0.0, cfg_fused).to(device).eval()
    unfused = MultiHeadSelfAttention(True, D, H, 0.0, cfg_unfused).to(device).eval()

    with torch.no_grad():
        Wq, Wk, Wv = fused.W_QKV.weight.chunk(3, dim=0)  # type: ignore
        bq, bk, bv = fused.W_QKV.bias.chunk(3, dim=0)  # type: ignore
        unfused.W_Q.weight.copy_(Wq)
        unfused.W_Q.bias.copy_(bq)  # type: ignore
        unfused.W_K.weight.copy_(Wk)
        unfused.W_K.bias.copy_(bk)  # type: ignore
        unfused.W_V.weight.copy_(Wv)
        unfused.W_V.bias.copy_(bv)  # type: ignore
        unfused.core.W_O.weight.copy_(fused.core.W_O.weight)
        unfused.core.W_O.bias.copy_(fused.core.W_O.bias)

    x = torch.randn(B, L, D, device=device)
    torch.testing.assert_close(fused(x), unfused(x), atol=1e-6, rtol=1e-5)


def _copy_weights_to_torch(
    ours: MultiHeadSelfAttention, ref: torch.nn.MultiheadAttention
) -> None:
    """Map Daniel-style parameters → PyTorch MHA layout (in-place)."""
    if ours.configs.use_fused_qkv:
        ref.in_proj_weight.data.copy_(ours.W_QKV.weight.data)  # type: ignore
        ref.in_proj_bias.data.copy_(ours.W_QKV.bias.data)  # type: ignore
    else:
        ref.in_proj_weight.data.copy_(  # type: ignore
            torch.cat(
                [ours.W_Q.weight, ours.W_K.weight, ours.W_V.weight], dim=0  # type: ignore
            )
        )
        ref.in_proj_bias.data.copy_(  # type: ignore
            torch.cat(
                [ours.W_Q.bias, ours.W_K.bias, ours.W_V.bias], dim=0  # type: ignore
            )
        )

    ref.out_proj.weight.data.copy_(ours.core.W_O.weight.data)
    ref.out_proj.bias.data.copy_(ours.core.W_O.bias.data)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("use_fused_qkv", [False, True])
def test_mhsa_matches_torch(is_causal, use_fused_qkv, device):
    D, H, B, L = 32, 4, 3, 7
    cfg = Configs(
        use_fused_qkv=use_fused_qkv,
        use_post_norm=True,
        use_final_layer_norm=False,
        use_original_init=False,  # we overwrite weights anyway
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )

    ours = (
        MultiHeadSelfAttention(
            is_causal=is_causal,
            d_model=D,
            n_head=H,
            dropout=0.0,
            configs=cfg,
        )
        .to(device)
        .eval()
    )

    ref = (
        torch.nn.MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            dropout=0.0,
            bias=True,
            batch_first=True,  # so both APIs agree on (B, L, D)
        )
        .to(device)
        .eval()
    )

    _copy_weights_to_torch(ours, ref)

    x = torch.randn(B, L, D, device=device, requires_grad=True)
    attn_mask = (
        torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)
        if is_causal
        else None
    )

    out_ref, _ = ref(x, x, x, attn_mask=attn_mask, need_weights=False)
    out_ours = ours(x)

    torch.testing.assert_close(out_ours, out_ref, atol=1e-6, rtol=1e-5)

    (g_ref,) = torch.autograd.grad(out_ref.sum(), x, retain_graph=True)
    (g_ours,) = torch.autograd.grad(out_ours.sum(), x)
    assert torch.all(torch.sign(g_ref) == torch.sign(g_ours))
