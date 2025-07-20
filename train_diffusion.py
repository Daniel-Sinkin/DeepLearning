from typing import Iterator, TypeAlias

import matplotlib.pyplot as plt
import torch
import torchvision
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.dataset_cifar import (
    CIFARBatch,
    CIFARImages,
    CIFARLabels,
    batch_size,
    channel,
    get_data,
    height,
    width,
)
from src.diffusion import DiffusionModel, get_beta_schedule_linear

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def create_cifar_iter(data_loader: DataLoader[CIFARBatch]) -> Iterator[CIFARBatch]:
    return iter(data_loader)


def show_images(images: CIFARImages, nrow: int = 8, title: str | None = None) -> None:
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(nrow, nrow))  # type: ignore
    if title:
        plt.title(title)  # type: ignore
    plt.axis("off")  # type: ignore
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # type: ignore
    plt.show()  # type: ignore


def main() -> None:
    T = 32
    beta_t = get_beta_schedule_linear(T=T)
    model = DiffusionModel(T=T, beta_t=beta_t)


if __name__ == "__main__":
    main()
