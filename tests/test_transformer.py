import torch
import pytest
from src.transformer import Transformer
from src.common import get_default_configs


@pytest.fixture(scope="module")
def tiny_transformer(device):
    """A mini model that runs in milliseconds."""
    cfg = get_default_configs()
    model = Transformer(
        d_model=32,
        n_head=4,
        d_ff=64,
        n_layer=2,
        dropout=0.0,
        source_vocab_size=1337,
        target_vocab_size=1337,
        pad_id=0,
        configs=cfg,
    ).to(device)
    model.eval()
    return model


def test_forward_shape(tiny_transformer, device):
    B, Lsrc, Ltgt = 2, 9, 7
    src = torch.randint(0, 1337, (B, Lsrc), device=device)
    tgt = torch.randint(0, 1337, (B, Ltgt), device=device)
    logits = tiny_transformer(src, tgt)
    assert logits.shape == (B, Ltgt, 1337)


def test_gradients_flow(tiny_transformer, device):
    B, Lsrc, Ltgt = 2, 5, 5
    src = torch.randint(0, 1337, (B, Lsrc), device=device)
    tgt = torch.randint(0, 1337, (B, Ltgt), device=device)
    logits = tiny_transformer(src, tgt)
    loss = logits.mean()
    loss.backward()
    # every parameter that requires-grad must now have grad
    assert all(
        p.grad is not None for p in tiny_transformer.parameters() if p.requires_grad
    )


def test_embedding_and_lm_head_weight_tying(tiny_transformer):
    if tiny_transformer.configs.tie_target_embedding_and_lm_head_weights:
        assert (
            tiny_transformer.lm_head.weight is tiny_transformer.target_embedding.weight
        )
    else:
        assert (
            tiny_transformer.lm_head.weight
            is not tiny_transformer.target_embedding.weight
        )
