from tqdm import tqdm

from pathlib import Path
from typing import Tuple
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms

Batch = int
Image = Float[Tensor, "Batch 1 32 32"]
Label = Int[Tensor, "Batch"]
Sample = Tuple[Image, Label]


def get_train_loader(batch_size: int = 64) -> DataLoader[Sample]:
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
        ]
    )
    data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=True)  # type: ignore


def get_test_loader(batch_size: int = 64) -> DataLoader[Sample]:
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
        ]
    )
    data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=False)  # type: ignore


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias=True),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=True),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        y = self.convs(x)
        y = y.reshape(batch_size, -1)
        return self.mlp(y)


def get_trained_model(
    device: str = "cpu", train_sample_rate: int = 25
) -> tuple[nn.Module, list[float], list[float]]:

    train_losses: list[float] = []
    test_losses: list[float] = []
    batch_size = 64
    train_dl = get_train_loader(batch_size=batch_size)
    test_dl = get_test_loader(batch_size=batch_size)

    model = LeNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    iteration = 0
    for images, labels in tqdm(train_dl, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        out = model(images)
        res = torch.nn.functional.softmax(out, dim=1)

        hot_labels = F.one_hot(labels, num_classes=10).to(  # type: ignore
            device=device, dtype=torch.float32
        )

        err = F.mse_loss(res, hot_labels)  # type: ignore

        err.backward()  # type: ignore
        optimiser.step()  # type: ignore

        train_losses.append(err.item() / len(images))
        if iteration % train_sample_rate == 0:
            with torch.no_grad():
                acc = 0.0
                count = 0
                for images_test, labels_test in test_dl:
                    count += len(images_test)

                    images_test = images_test.to(device)
                    labels_test = labels_test.to(device)

                    hot_labels = F.one_hot(labels, num_classes=10).to(  # type: ignore
                        device=device, dtype=torch.float32
                    )
                    err = F.mse_loss(res, hot_labels)
                    acc += err.item()

                test_losses.append(acc / count)

        iteration += 1

    return model, train_losses, test_losses
