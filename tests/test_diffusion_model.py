"""
tests/test_diffusion.py

Robust test suite for:
  * get_beta_schedule_linear
  * DiffusionModel (shapes, alpha bars, stability)
  * Deterministic fixed-pair overfitting (no stochastic targets)
  * Manual loss consistency

The overfitting test uses a SMALL UNet to avoid divergence and
removes stochasticity by fixing (x0, t, eps).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.diffusion import DiffusionModel, get_beta_schedule_linear
from src.unet.unet import UNet  # for overriding with a tiny version

# ---------------------------------------------------------------------
# Global fixtures / seeding
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


def _finite(t: Tensor) -> bool:
    return torch.isfinite(t).all().item()


# ---------------------------------------------------------------------
# Beta schedule tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 10, 50, 100])
def test_beta_schedule_linear_basic(T: int) -> None:
    betas = get_beta_schedule_linear(T, beta_0=1e-4, beta_T=2e-2)
    assert betas.shape == (T,)
    assert torch.all(betas >= 0)
    assert torch.all(betas <= 0.1)
    if T > 1:
        diffs = betas[1:] - betas[:-1]
        # Allow tiny negative due to float error
        assert torch.all(diffs >= -1e-10)


def test_beta_schedule_endpoints() -> None:
    T = 7
    b0, bT = 0.002, 0.009
    betas = get_beta_schedule_linear(T, beta_0=b0, beta_T=bT)
    assert torch.isclose(betas[0], torch.tensor(b0))
    assert torch.isclose(betas[-1], torch.tensor(bT))


# ---------------------------------------------------------------------
# DiffusionModel structural tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("T", [5, 20])
def test_diffusion_alpha_bars_monotonic(T: int) -> None:
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas)
    alpha_bars = model.alpha_bars
    assert alpha_bars.shape == (T,)
    assert torch.all(alpha_bars > 0)
    assert torch.all(alpha_bars <= 1)
    if T > 1:
        assert torch.all(alpha_bars[1:] < alpha_bars[:-1])  # strictly decreasing


def test_diffusion_forward_shape(device: torch.device) -> None:
    T = 25
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=64).to(device)
    B = 4
    x_t = torch.randn(B, 3, 32, 32, device=device)
    t = torch.randint(0, T, (B,), device=device)
    out = model(x_t, t)
    assert out.shape == x_t.shape
    assert _finite(out)


def test_diffusion_train_step_returns_scalar() -> None:
    T = 10
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    x0 = torch.randn(2, 3, 32, 32)
    loss = model.train_step(x0)
    assert loss.ndim == 0
    assert _finite(loss)


def test_diffusion_train_step_gradients() -> None:
    T = 12
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    x0 = torch.randn(2, 3, 32, 32)
    loss = model.train_step(x0)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert _finite(p.grad)


def test_diffusion_parameter_update() -> None:
    T = 10
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.randn(4, 3, 32, 32)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = model.train_step(x0)
    loss.backward()
    opt.step()
    changed = any(
        not torch.allclose(before[n], p.detach()) for n, p in model.named_parameters()
    )
    assert changed, "Parameters did not change after optimization step."


def test_diffusion_single_image_input() -> None:
    T = 6
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas)
    x0 = torch.randn(3, 32, 32)  # no batch dim
    loss = model.train_step(x0)
    assert _finite(loss)


# ---------------------------------------------------------------------
# Stability / variance control tests
# ---------------------------------------------------------------------


def test_diffusion_train_step_stability() -> None:
    """
    Multiple consecutive train_step calls should produce finite, reasonably bounded losses.
    """
    T = 10
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    # shrink UNet to reduce variance (override)
    model.model = UNet(
        in_channels=3,
        base_channels=16,
        time_emb_dim=32,
        channel_mults=(1, 2),
    )
    for _ in range(5):
        x0 = torch.randn(2, 3, 32, 32)
        loss = model.train_step(x0)
        assert _finite(loss)
        assert loss < 20.0, f"Unexpected large loss: {loss.item():.4f}"


def test_diffusion_manual_fixed_pair_consistency() -> None:
    """
    Manual loss computed from the same distribution should be comparable
    (within a loose factor) to train_step's loss.
    """
    torch.manual_seed(2)
    T = 5
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    model.model = UNet(
        in_channels=3,
        base_channels=16,
        time_emb_dim=32,
        channel_mults=(1, 2),
    )
    B = 3
    x0 = torch.randn(B, 3, 16, 16)
    t = torch.randint(0, T, (B,))
    alpha_bars = model.alpha_bars[t].reshape(B, 1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * eps
    eps_pred = model(x_t, t)
    manual_loss = nn.functional.mse_loss(eps_pred, eps)

    train_step_loss = model.train_step(x0)

    ratio = manual_loss.item() / train_step_loss.item()
    assert (
        0.25 < ratio < 4.0
    ), f"Manual/train_step loss ratio out of expected range: {ratio:.3f}"


# ---------------------------------------------------------------------
# Deterministic fixed-pair overfitting test
# ---------------------------------------------------------------------


def test_diffusion_model_overfits_single_batch_fixed_pair() -> None:
    """
    Overfit a *fixed* (x0, t, eps) mapping:
      - Remove stochasticity (no resampling t or eps).
      - Use a small network.
      - Expect substantial loss reduction.
    """
    torch.manual_seed(0)
    T = 20
    betas = get_beta_schedule_linear(T)
    model = DiffusionModel(T, betas, embed_dim=32)
    # Replace with a small UNet for stability
    model.model = UNet(
        in_channels=3,
        base_channels=16,
        time_emb_dim=32,
        channel_mults=(1, 2),
    )

    model.train()

    B = 4
    x0 = torch.randn(B, 3, 32, 32)

    # Fixed t and eps
    t_fixed = torch.randint(0, T, (B,))
    eps_fixed = torch.randn_like(x0)

    alpha_bars = model.alpha_bars[t_fixed].reshape(B, 1, 1, 1)
    x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * eps_fixed

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn() -> Tensor:
        pred = model(x_t, t_fixed)
        return nn.functional.mse_loss(pred, eps_fixed)

    with torch.no_grad():
        initial_loss = loss_fn().item()

    losses = []
    for step in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        losses.append(loss.item())
        assert losses[-1] < 1e3, f"Divergence at step {step}: loss={losses[-1]:.2f}"

    final_loss = sum(losses[-10:]) / 10.0
    assert (
        final_loss < 0.4 * initial_loss
    ), f"Overfitting insufficient: initial={initial_loss:.4f}, final={final_loss:.4f}"
