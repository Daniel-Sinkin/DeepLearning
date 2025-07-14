# tests/test_mhsa_reference_parity.py
import torch
import pytest

from src.multi_head_attention import MultiHeadSelfAttention
from src.common import Configs

# ---------- helpers ----------------------------------------------------------


def _copy_weights_to_torch(
    ours: MultiHeadSelfAttention, ref: torch.nn.MultiheadAttention
) -> None:
    """Map Daniel-style parameters → PyTorch MHA layout (in-place)."""
    if ours.configs.use_fused_qkv:
        # ours.W_QKV.weight: (3D, D)  → ref.in_proj_weight: (3D, D)
        ref.in_proj_weight.data.copy_(ours.W_QKV.weight.data)  # type: ignore
        ref.in_proj_bias.data.copy_(ours.W_QKV.bias.data)  # type: ignore
    else:  # unfused
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


# ---------- tests ------------------------------------------------------------


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
    if is_causal:
        # upper-triangular 1s ABOVE diag → “future tokens” masking
        attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)
    else:
        attn_mask = None

    out_ref, _ = ref(x, x, x, attn_mask=attn_mask, need_weights=False)
    out_ours = ours(x)

    torch.testing.assert_close(out_ours, out_ref, atol=1e-6, rtol=1e-5)

    # optional: gradients agree on sign (value check is too tight for float32)
    (g_ref,) = torch.autograd.grad(out_ref.sum(), x, retain_graph=True)
    (g_ours,) = torch.autograd.grad(out_ours.sum(), x)
    assert torch.all(torch.sign(g_ref) == torch.sign(g_ours))
