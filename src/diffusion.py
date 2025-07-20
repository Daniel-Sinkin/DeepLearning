from typing import TypeAlias, Iterator

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
import torch
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from .common import assert_shape


batch_size, channel, height, width = 64, 3, 32, 32
CIFARImages: TypeAlias = Float[Tensor, f"B {channel} {height} {width}"]
CIFARLabels: TypeAlias = Int[Tensor, "B"]
CIFARBatch: TypeAlias = tuple[CIFARImages, CIFARLabels]


def validate_batch(batch: CIFARBatch) -> None:
    images, labels = batch
    assert_shape(images, (batch_size, channel, height, width))
    assert_shape(labels, (batch_size,))


def get_beta_schedule_linear(
    T: int, beta_0: float = 1e-4, beta_T: float = 1e-1
) -> Float[Tensor, "T"]:
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
    def __init__(self, T: int, beta_t: Float[Tensor, "T"]):
        super().__init__()  # type: ignore
        assert beta_t.shape == (T,)
        self.T = T
        self.alpha_bars = torch.cumprod(1 - beta_t, dim=0)
