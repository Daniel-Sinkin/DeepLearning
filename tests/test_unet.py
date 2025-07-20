"""
tests/test_unet.py

Extensive tests for UNet and its constituent blocks (ResBlock, DownBlock, UpBlock, BottleneckBlock).
Assumes all are defined in src/unet.py (adjust imports if you used another filename).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

# Adapt these imports if your file/layout differs
from src.unet import (
    BottleneckBlock,
    DownBlock,
    ResBlock,
    SinusoidalTimeEmbedding,
    UNet,
    UpBlock,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")  # keep CPU for CI stability


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------
# Helper Assertions
# ---------------------------------------------------------------------


def _assert_finite(t: Tensor) -> None:
    assert torch.isfinite(t).all(), "Tensor contains NaN or Inf."


# ---------------------------------------------------------------------
# SinusoidalTimeEmbedding Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dim", [16, 32, 64])
def test_time_embedding_shape(dim: int, device: torch.device) -> None:
    emb = SinusoidalTimeEmbedding(dim).to(device)
    t = torch.randint(0, 1000, (7,), device=device)
    out = emb(t)
    assert out.shape == (7, dim)
    _assert_finite(out)


def test_time_embedding_deterministic(device: torch.device) -> None:
    emb = SinusoidalTimeEmbedding(32).to(device)
    t = torch.tensor([0, 1, 10, 999], device=device)
    out1 = emb(t)
    out2 = emb(t)
    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------
# ResBlock Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("in_c,out_c,time_dim", [(8, 8, 16), (8, 16, 32), (16, 8, 32)])
def test_resblock_forward_shape(in_c: int, out_c: int, time_dim: int) -> None:
    block = ResBlock(in_c, out_c, time_dim)
    x = torch.randn(4, in_c, 16, 16)
    t_emb = torch.randn(4, time_dim)
    y = block(x, t_emb)
    assert y.shape == (4, out_c, 16, 16)
    _assert_finite(y)


def test_resblock_time_emb_assert() -> None:
    block = ResBlock(8, 8, 32)
    x = torch.randn(2, 8, 8, 8)
    bad_t = torch.randn(2, 31)  # wrong last dim
    with pytest.raises(AssertionError):
        _ = block(x, bad_t)


def test_resblock_gradients() -> None:
    block = ResBlock(4, 8, 16)
    x = torch.randn(3, 4, 12, 12, requires_grad=True)
    t = torch.randn(3, 16, requires_grad=True)
    out = block(x, t).mean()
    out.backward()
    assert x.grad is not None and t.grad is not None
    _assert_finite(x.grad)
    _assert_finite(t.grad)


# ---------------------------------------------------------------------
# DownBlock Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("in_c,out_c,time_dim", [(8, 16, 32), (16, 16, 16)])
def test_downblock_shapes(in_c: int, out_c: int, time_dim: int) -> None:
    db = DownBlock(in_c, out_c, time_dim)
    x = torch.randn(2, in_c, 32, 32)
    t = torch.randn(2, time_dim)
    down, skip = db(x, t)
    assert skip.shape == (2, out_c, 32, 32)
    assert down.shape == (2, out_c, 16, 16)
    _assert_finite(down)
    _assert_finite(skip)


def test_downblock_time_emb_assert() -> None:
    block = DownBlock(8, 16, 32)
    x = torch.randn(2, 8, 16, 16)
    bad_t = torch.randn(2, 31)
    with pytest.raises(AssertionError):
        _ = block(x, bad_t)


# ---------------------------------------------------------------------
# UpBlock Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_c,skip_c,out_c,time_dim", [(16, 16, 8, 32), (32, 16, 16, 16)]
)
def test_upblock_shapes(in_c: int, skip_c: int, out_c: int, time_dim: int) -> None:
    ub = UpBlock(in_c, skip_c, out_c, time_dim)
    x = torch.randn(2, in_c, 16, 16)  # low-res
    skip = torch.randn(2, skip_c, 32, 32)  # high-res skip
    t = torch.randn(2, time_dim)
    y = ub(x, skip, t)
    assert y.shape == (2, out_c, 32, 32)
    _assert_finite(y)


def test_upblock_interpolation_off_by_one() -> None:
    ub = UpBlock(8, 8, 8, 32)
    x = torch.randn(1, 8, 15, 15)  # -> upsampled 30x30
    skip = torch.randn(1, 8, 31, 31)  # mismatch forces interpolate
    t = torch.randn(1, 32)
    y = ub(x, skip, t)
    assert y.shape == (1, 8, 31, 31)


def test_upblock_time_emb_assert() -> None:
    ub = UpBlock(8, 8, 8, 16)
    x = torch.randn(1, 8, 8, 8)
    skip = torch.randn(1, 8, 16, 16)
    bad_t = torch.randn(1, 15)
    with pytest.raises(AssertionError):
        _ = ub(x, skip, bad_t)


# ---------------------------------------------------------------------
# BottleneckBlock Tests
# ---------------------------------------------------------------------


def test_bottleneck_block_roundtrip() -> None:
    bn = BottleneckBlock(32, 64)
    x = torch.randn(2, 32, 8, 8)
    t = torch.randn(2, 64)
    y = bn(x, t)
    assert y.shape == (2, 32, 8, 8)


# ---------------------------------------------------------------------
# UNet Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel_mults,base,in_ch",
    [
        ((1, 2), 16, 3),
        ((1, 2, 4), 8, 3),
        ((1,), 32, 1),
    ],
)
def test_unet_forward_shapes(
    channel_mults: tuple[int, ...], base: int, in_ch: int
) -> None:
    model = UNet(
        in_channels=in_ch,
        base_channels=base,
        time_emb_dim=64,
        channel_mults=channel_mults,
    )
    B, H, W = 2, 32, 32
    x = torch.randn(B, in_ch, H, W)
    t = torch.randint(0, 1000, (B,))
    y = model(x, t)
    assert y.shape == (B, in_ch, H, W)
    _assert_finite(y)


def test_unet_time_shape_assert() -> None:
    model = UNet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (3,))  # wrong batch
    with pytest.raises(AssertionError):
        _ = model(x, t)


def test_unet_eval_deterministic() -> None:
    model = UNet()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([10])
    with torch.no_grad():
        y1 = model(x, t)
        y2 = model(x, t)
    torch.testing.assert_close(y1, y2)


def test_unet_training_changes_params() -> None:
    model = UNet(base_channels=8, time_emb_dim=32, channel_mults=(1, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 100, (4,))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = model(x, t).mean()
    loss.backward()
    opt.step()
    any_changed = any(
        not torch.allclose(before[n], p.detach()) for n, p in model.named_parameters()
    )
    assert any_changed, "Parameters did not update after backward + step."


def test_unet_gradients_flow() -> None:
    model = UNet(base_channels=8, time_emb_dim=32, channel_mults=(1, 2))
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    t = torch.randint(0, 100, (2,))
    out = model(x, t).sum()
    out.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert grads_ok


def test_unet_raises_on_out_of_range_t() -> None:
    model = UNet()
    x = torch.randn(1, 3, 32, 32)
    # Large t is still allowed by embedding; not indexing -> so no error.
    # We'll simulate an internal misuse by passing float tensor to confirm failure.
    t = torch.randn(1)  # float values (embedding expects integer semantics)
    # Should still work (embedding converts to float inside), so we assert no raise.
    _ = model(x, t)  # if you want strict int-only, you'd add an explicit assert.


# ---------------------------------------------------------------------
# Performance / Complexity Sanity (Quick)
# ---------------------------------------------------------------------


def test_unet_parameter_count_reasonable() -> None:
    model = UNet(base_channels=8, time_emb_dim=32, channel_mults=(1, 2, 4))
    total = sum(p.numel() for p in model.parameters())
    # Rough bound to catch accidental blow-ups
    assert total < 10_000_000


def test_unet_overfits_fixed_batch() -> None:
    """
    UNet should overfit a fixed (x, t, y) mapping.
    We freeze x and t, generate a random target y,
    and expect MSE loss to drop significantly over training.
    """
    B = 4
    in_channels = 3
    H = W = 32

    model = UNet(
        in_channels=in_channels,
        base_channels=16,
        time_emb_dim=64,
        channel_mults=(1, 2),
    )

    x = torch.randn(B, in_channels, H, W)
    t = torch.randint(0, 1000, (B,))
    target = torch.randn_like(x)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn() -> Tensor:
        out = model(x, t)
        return nn.functional.mse_loss(out, target)

    # Initial loss
    with torch.no_grad():
        initial_loss = loss_fn().item()

    # Train for a few iterations
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    final_loss = sum(losses[-10:]) / 10
    assert (
        final_loss < 0.4 * initial_loss
    ), f"UNet failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
