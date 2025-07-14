import pytest
import torch
from src.multi_head_attention import MultiHeadSelfAttention
from src.common import Configs


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
