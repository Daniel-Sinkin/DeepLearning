"""
danielsinkin97@gmail.com
"""

import torch
from jaxtyping import Float
from torch import Tensor, nn

from .dataset_cifar import CIFARImages

T_ = "T"


def get_beta_schedule_linear(
    T: int, beta_0: float = 1e-4, beta_T: float = 1e-1
) -> Float[Tensor, f"{T_}"]:  # noqa: F821
    beta_t = torch.linspace(beta_0, beta_T, T)
    return torch.cumprod(1 - beta_t, dim=0)


def get_xt(t: int, images: CIFARImages, alpha_bars: Tensor) -> CIFARImages:
    assert 0 <= t < len(alpha_bars)
    if t == 0:
        return images
    alpha_bar = alpha_bars[t]
    eps = torch.randn_like(images)
    noisy_images = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * eps
    return noisy_images


class DiffusionModel(nn.Module):
    def __init__(self, T: int, beta_t: Float[Tensor, f"{T_}"]):
        super().__init__()  # type: ignore
        assert beta_t.shape == (T,)
        self.T = T
        self.alpha_bars = torch.cumprod(1 - beta_t, dim=0)
