"""
danielsinkin97@gmail.com
"""

from typing import Iterator, TypeAlias

import matplotlib.pyplot as plt
import torchvision
import torchvision.utils
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .common import assert_shape

batch_size, channel, height, width = 64, 3, 32, 32
B_ = "B"  # Workaround for Ruff
CIFARImages: TypeAlias = Float[Tensor, f"{B_} {channel} {height} {width}"]
CIFARLabels: TypeAlias = Int[Tensor, f"{B_}"]
CIFARBatch: TypeAlias = tuple[CIFARImages, CIFARLabels]


def validate_batch(batch: CIFARBatch) -> None:
    images, labels = batch
    assert_shape(images, (batch_size, channel, height, width))
    assert_shape(labels, (batch_size,))


class ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img  # drop label

    def __len__(self):
        return len(self.dataset)


def get_data(train: bool = True) -> DataLoader[CIFARImages]:
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    raw_dataset = CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )
    image_dataset = ImageOnlyDataset(raw_dataset)

    return DataLoader(
        image_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )


def get_iterator_batch(data_loader: DataLoader[CIFARBatch]) -> Iterator[CIFARBatch]:
    return iter(data_loader)


def get_iterator_images(data_loader: DataLoader[CIFARBatch]) -> Iterator[CIFARImages]:
    for images, _ in data_loader:
        yield images


def get_iterator_labels(data_loader: DataLoader[CIFARBatch]) -> Iterator[CIFARLabels]:
    for _, labels in data_loader:
        yield labels


def visualize_images_grid(images: CIFARImages, title: str = "Image Grid") -> None:
    """
    Visualizes a batch of images in an 8x8 grid.
    Assumes images are normalized to [-1, 1] and denormalizes to [0, 1] for display.
    """
    images = images * 0.5 + 0.5
    images = images.clamp(0.0, 1.0)
    images = images[:64]
    grid = torchvision.utils.make_grid(images, nrow=8)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))  # type: ignore
    plt.imshow(np_grid)  # type: ignore
    plt.axis("off")  # type: ignore
    plt.title(title)  # type: ignore
    plt.show()  # type: ignore
