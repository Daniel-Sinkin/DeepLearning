"""
danielsinkin97@gmail.com
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FloatArray = npt.NDArray[np.float32]
ArrayLikeF32 = npt.ArrayLike


@dataclass(frozen=True)
class DatasetSplit:
    """Holds noisy inputs, clean targets, and underlying 1D signal."""

    noisy: FloatArray
    clean: FloatArray
    signal: FloatArray


@dataclass(frozen=True)
class VarianceStats:
    """Holds per-dimension variances before and after reconstruction."""

    pre: np.ndarray
    post: np.ndarray


def set_seeds(seed: int = 42) -> None:
    """Set seeds for numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(n: int, sigma: float) -> DatasetSplit:
    """
    Generate the synthetic 5D dataset.

    Clean data: [t, 0, 0, 0, 0] with t ~ U[-1, 1]
    Noise: i.i.d. Gaussian with std = sigma

    Returns:
        DatasetSplit with:
            noisy: noisy inputs (n, 5), dtype float32
            clean: clean targets (n, 5), dtype float32
            signal: underlying 1D t values (n, 1), dtype float32
    """
    t: FloatArray = np.random.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    x_clean: FloatArray = np.concatenate(
        [t, np.zeros((n, 4), dtype=np.float32)], axis=1
    )
    noise: FloatArray = np.random.normal(0.0, sigma, size=x_clean.shape).astype(
        np.float32
    )
    x_noisy: FloatArray = x_clean + noise
    return DatasetSplit(noisy=x_noisy, clean=x_clean, signal=t)


class DAE(nn.Module):
    """Denoising Autoencoder: 5→hidden→hidden→5 MLP with ReLU."""

    def __init__(self, in_dim: int = 5, hidden: int = 16, out_dim: int = 5) -> None:
        """Initialize network architecture and weights."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming uniform initialization to all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch, 5), dtype float32
        Returns:
            Reconstructed tensor of shape (batch, 5), dtype float32
        """
        return self.net(x)


def build_loaders(
    x_noisy: FloatArray, x_clean: FloatArray, batch_size: int
) -> DataLoader:
    """
    Create a PyTorch DataLoader from NumPy arrays.

    Args:
        x_noisy: noisy inputs (N, 5), dtype float32
        x_clean: clean targets (N, 5), dtype float32
        batch_size: minibatch size
    """
    ds = TensorDataset(
        torch.from_numpy(x_noisy).to(torch.float32),
        torch.from_numpy(x_clean).to(torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def train(
    model: nn.Module,
    loader: DataLoader,
    x_val_noisy: torch.Tensor,
    x_val_clean: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """
    Train a DAE using MSE loss and SGD.

    Args:
        model: the DAE model
        loader: training data loader
        x_val_noisy: validation noisy inputs (N_val, 5), dtype float32
        x_val_clean: validation clean targets (N_val, 5), dtype float32
        epochs: number of training epochs
        lr: learning rate
        device: computation device
    """
    model.to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        n_seen = 0
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optim.step()
            running += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]
        train_mse = running / max(1, n_seen)

        model.eval()
        with torch.no_grad():
            yv = model(x_val_noisy.to(device))
            val_mse = criterion(yv, x_val_clean.to(device)).item()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"epoch {epoch:02d} | train MSE={train_mse:.6f} | val MSE={val_mse:.6f}"
            )

    print(f"Training time: {time.perf_counter() - t0:.2f}s on {device}")


def eval_variances(x_noisy: torch.Tensor, x_recon: torch.Tensor) -> VarianceStats:
    """
    Compute variance per dimension before and after denoising.

    Args:
        x_noisy: noisy tensor, shape (N, 5), dtype float32
        x_recon: reconstructed tensor, shape (N, 5), dtype float32
    """
    return VarianceStats(
        pre=x_noisy.var(dim=0).cpu().numpy(),
        post=x_recon.var(dim=0).cpu().numpy(),
    )


def plot_signal_recovery(
    t_val: FloatArray,
    x_val_noisy: FloatArray,
    x_val_recon: FloatArray,
    outdir: Path,
) -> None:
    """Plot first component recovery vs true signal t."""
    plt.figure(figsize=(6, 4))
    plt.scatter(t_val.flatten(), x_val_noisy[:, 0], s=6, alpha=0.4, label="noisy")
    plt.scatter(
        t_val.flatten(), x_val_recon[:, 0], s=6, alpha=0.4, label="reconstructed"
    )
    plt.plot(
        np.sort(t_val.flatten()),
        np.sort(t_val.flatten()),
        "r--",
        linewidth=2,
        label="Ideal y = x",
    )
    plt.title("Signal recovery on first coordinate")
    plt.xlabel("True t")
    plt.ylabel("First component")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir.joinpath("coord0_recovery.png"), dpi=160)
    plt.close()


def plot_nuisance_histograms(
    x_val_noisy: FloatArray,
    x_val_recon: FloatArray,
    outdir: Path,
    bins: int = 60,
) -> None:
    """Plot histograms for coordinates 1..4 to show noise removal."""
    _, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    for j, ax in enumerate(axes, start=1):
        n_noisy, edges = np.histogram(x_val_noisy[:, j], bins=bins, density=True)
        n_recon, _ = np.histogram(x_val_recon[:, j], bins=edges, density=True)
        ymax = 1.1 * max(n_noisy.max(), n_recon.max())
        ax.hist(x_val_noisy[:, j], bins=edges, alpha=0.5, label="noisy", density=True)
        ax.hist(
            x_val_recon[:, j],
            bins=edges,
            alpha=0.7,
            label="reconstructed",
            density=True,
        )
        ax.set_ylim(0, ymax)
        ax.set_title(f"Coordinate {j}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
    plt.suptitle("Noise removal on nuisance coordinates")
    plt.tight_layout()
    plt.savefig(outdir.joinpath("nuisance_dims_hist.png"), dpi=160)
    plt.close()


def plot_scatter_0_vs_1(
    x_val_noisy: FloatArray,
    x_val_recon: FloatArray,
    outdir: Path,
) -> None:
    """Scatter plot of first vs second component before and after denoising."""
    plt.figure(figsize=(5, 5))
    plt.scatter(x_val_noisy[:, 0], x_val_noisy[:, 1], s=3, alpha=0.35, label="noisy")
    plt.scatter(
        x_val_recon[:, 0], x_val_recon[:, 1], s=3, alpha=0.35, label="reconstructed"
    )
    plt.axhline(0, linewidth=1, linestyle="--", color="red", label="Ground truth (y=0)")
    plt.title("Projection back to the 1D manifold")
    plt.xlabel("First component (signal)")
    plt.ylabel("Second component (noise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir.joinpath("scatter_0_vs_1.png"), dpi=160)
    plt.close()


def main() -> None:
    """Run end-to-end training, evaluation, and plotting of the 5D DAE toy example."""
    set_seeds(42)

    N_train = 50_000
    N_val = 5_000
    noise_sigma = 0.15
    hidden = 16
    lr = 1e-2
    epochs = 25
    batch_size = 256

    outdir = Path("plots/dae/low_dim_model")
    outdir.mkdir(parents=True, exist_ok=True)

    train_data = make_dataset(N_train, noise_sigma)
    val_data = make_dataset(N_val, noise_sigma)

    loader = build_loaders(train_data.noisy, train_data.clean, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAE(hidden=hidden)

    train(
        model=model,
        loader=loader,
        x_val_noisy=torch.from_numpy(val_data.noisy).to(torch.float32),
        x_val_clean=torch.from_numpy(val_data.clean).to(torch.float32),
        epochs=epochs,
        lr=lr,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        xva_noisy_t = torch.from_numpy(val_data.noisy).to(torch.float32).to(device)
        yva_t = model(xva_noisy_t)
    yva_np: FloatArray = yva_t.cpu().numpy()

    var_stats = eval_variances(
        torch.from_numpy(val_data.noisy).to(torch.float32),
        torch.from_numpy(yva_np).to(torch.float32),
    )
    print("\nDim variances (val set):")
    for i in range(5):
        print(
            f"dim {i}: noisy var={var_stats.pre[i]:.5f} -> recon var={var_stats.post[i]:.5f}"
        )

    plot_signal_recovery(val_data.signal, val_data.noisy, yva_np, outdir)
    plot_nuisance_histograms(val_data.noisy, yva_np, outdir)
    plot_scatter_0_vs_1(val_data.noisy, yva_np, outdir)

    print(f"\nSaved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
