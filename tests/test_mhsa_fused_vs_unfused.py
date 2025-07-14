# tests/test_mhsa_fused_vs_unfused.py
import torch

from src.multi_head_attention import MultiHeadSelfAttention
from src.common import Configs


def test_fused_equals_unfused(device):
    D, H, B, L = 48, 6, 2, 5

    # --- build both variants --------------------------------------------------
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

    # --- copy fused weights into unfused --------------------------------------
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

    # --- forward --------------------------------------------------------------
    x = torch.randn(B, L, D, device=device)
    torch.testing.assert_close(fused(x), unfused(x), atol=1e-6, rtol=1e-5)
