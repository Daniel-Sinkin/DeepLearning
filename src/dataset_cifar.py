"""
danielsinkin97@gmail.com
"""

from typing import TypeAlias

from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .common import assert_shape

batch_size, channel, height, width = 64, 3, 32, 32
CIFARImages: TypeAlias = Float[Tensor, f"B {channel} {height} {width}"]
CIFARLabels: TypeAlias = Int[Tensor, f"{batch_size}"]
CIFARBatch: TypeAlias = tuple[CIFARImages, CIFARLabels]


def validate_batch(batch: CIFARBatch) -> None:
    images, labels = batch
    assert_shape(images, (batch_size, channel, height, width))
    assert_shape(labels, (batch_size,))


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
        trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    return train_loader
