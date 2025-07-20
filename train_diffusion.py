from typing import TypeAlias, Iterator

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision

from src.diffusion import DiffusionModel, get_beta_schedule_linear

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

batch_size, channel, height, width = 64, 3, 32, 32
CIFARImages: TypeAlias = Float[Tensor, f"B {channel} {height} {width}"]
CIFARLabels: TypeAlias = Int[Tensor, "B"]
CIFARBatch: TypeAlias = tuple[CIFARImages, CIFARLabels]


def validate_batch(batch: CIFARBatch) -> None:
    images, labels = batch
    assert images.shape == (batch_size, channel, height, width)
    assert labels.shape == (batch_size,)


def get_data(train: bool = True) -> DataLoader[CIFARBatch]:
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    trainset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    train_loader: DataLoader[CIFARBatch] = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=(device != "mps"),
    )
    return train_loader


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
