import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torchvision.utils import save_image

from src.dataset_cifar import get_data
from src.diffusion_model import DiffusionModel, get_beta_schedule_linear

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ------------------------------------------------------------
# Diffusion Sampling
# ------------------------------------------------------------


def sample(
    model: DiffusionModel,
    betas: Tensor,
    n_samples: int = 64,
    device: torch.device | None = None,
) -> Tensor:
    """Generates samples using the reverse diffusion process.

    Args:
        model: Trained diffusion model.
        betas: Tensor of shape [T] with beta schedule used during training.
        n_samples: Number of images to sample.
        device: Torch device.

    Returns:
        Tensor of shape [n_samples, 3, 32, 32] in range [-1, 1].
    """
    model.eval()
    with torch.no_grad():
        device = device or next(model.parameters()).device
        T = betas.shape[0]
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        x_t = torch.randn(n_samples, 3, 32, 32, device=device)
        for t in reversed(range(T)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = (
                alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            )

            eps_theta = model(x_t, t_tensor)
            # Predict x0
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(
                alpha_bar_t
            )
            # Optionally clip to valid range [-1,1]
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Compute the mean of q(x_{t-1} | x_t, x0)
            coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x_t

            if t > 0:
                # Posterior variance (variance of q(x_{t-1}|x_t,x0))
                posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_var) * noise
            else:
                x_t = mean
        return x_t


# ------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------


def evaluate(model: DiffusionModel, data_loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            loss = model.train_step(images)
            b = images.size(0)
            total_loss += loss.item() * b
            n += b
    return total_loss / max(1, n)


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 42
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    embed_dim: int = 128
    unet_base_channels: int = 128
    unet_channel_mults: tuple[int, ...] = (1, 2, 2, 4)
    checkpoint_interval: int = 10
    sample_interval: int = 10
    num_sample_images: int = 64
    out_dir: str = "runs/exp1"
    resume: str | None = None
    ema_decay: float | None = 0.999  # Set None to disable EMA


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name]

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name]
        self.backup = {}


# ------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = default_device()
    out_dir = Path(cfg.out_dir)
    checkpoints_dir = out_dir.joinpath("checkpoints")
    samples_dir = out_dir.joinpath("samples")
    plots_dir = out_dir.joinpath("plots")
    for d in [out_dir, checkpoints_dir, samples_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    # Data
    train_loader = get_data(train=True)
    test_loader = get_data(train=False)

    # Beta schedule
    betas = get_beta_schedule_linear(cfg.timesteps, cfg.beta_start, cfg.beta_end).to(
        device
    )

    model = DiffusionModel(
        T=cfg.timesteps,
        beta_t=betas,
        embed_dim=cfg.embed_dim,
        unet_base_channels=cfg.unet_base_channels,
        unet_channel_mults=cfg.unet_channel_mults,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    train_losses: List[float] = []
    test_losses: List[float] = []

    ema: EMA | None = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    if cfg.resume is not None:
        ckpt_path = Path(cfg.resume)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])  # type: ignore
            optimizer.load_state_dict(ckpt["optimizer_state"])  # type: ignore
            start_epoch = ckpt["epoch"] + 1  # type: ignore
            if ema and "ema_state" in ckpt:
                ema.shadow = {
                    k: v.clone().to(device) for k, v in ckpt["ema_state"].items()
                }  # type: ignore
            if "train_losses" in ckpt:
                train_losses = ckpt["train_losses"]  # type: ignore
            if "test_losses" in ckpt:
                test_losses = ckpt["test_losses"]  # type: ignore
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}")
        else:
            print(f"Warning: resume path {ckpt_path} does not exist. Starting fresh.")

    total_start = time.perf_counter()
    epoch_times: List[float] = []

    print("Starting to train")
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        num_samples = 0

        i = 0
        for images in train_loader:
            print(f"{i} / 728 ({i / 728:.2%})")
            i += 1

            images = images.to(device)
            loss = model.train_step(images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if ema:
                ema.update(model)

            b = images.size(0)
            running_loss += loss.item() * b
            num_samples += b

        epoch_train_loss = running_loss / max(1, num_samples)

        # Evaluate
        epoch_test_loss = evaluate(model, test_loader, device)
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.perf_counter() - total_start
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = cfg.epochs - (epoch + 1)
        eta_total = avg_epoch_time * remaining_epochs

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
            f"train_loss={epoch_train_loss:.6f} | test_loss={epoch_test_loss:.6f} | "
            f"epoch_time={format_seconds(epoch_time)} | elapsed={format_seconds(elapsed)} | "
            f"ETA total={format_seconds(eta_total)}"
        )

        # Sampling (use EMA weights if enabled)
        if (epoch + 1) % cfg.sample_interval == 0 or (epoch + 1) == cfg.epochs:
            if ema:
                ema.apply_shadow(model)
            samples = sample(
                model, betas, n_samples=cfg.num_sample_images, device=device
            )
            if ema:
                ema.restore(model)
            # Scale from [-1,1] to [0,1]
            samples_01 = (samples.clamp(-1, 1) + 1) / 2
            sample_path = samples_dir.joinpath(f"samples_epoch_{epoch + 1:04d}.png")
            save_image(
                samples_01, sample_path, nrow=int(math.sqrt(cfg.num_sample_images))
            )
            print(f"Saved samples to {sample_path}")

        # Checkpointing
        if (epoch + 1) % cfg.checkpoint_interval == 0 or (epoch + 1) == cfg.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "betas": betas,
                "config": cfg.__dict__,
                "train_losses": train_losses,
                "test_losses": test_losses,
            }
            if ema:
                ckpt["ema_state"] = ema.shadow
            ckpt_path = checkpoints_dir.joinpath(f"ckpt_epoch_{epoch + 1:04d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Final plots
    plt.figure(figsize=(8, 5))  # type: ignore
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")  # type: ignore
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")  # type: ignore
    plt.xlabel("Epoch")  # type: ignore
    plt.ylabel("Loss")  # type: ignore
    plt.title("Diffusion Training & Test Loss")  # type: ignore
    plt.legend()  # type: ignore
    plot_path = plots_dir.joinpath("loss_curve.png")
    plt.savefig(plot_path)  # type: ignore
    plt.close()  # type: ignore
    print(f"Saved loss curve to {plot_path}")

    # Save final model weights only (convenience)
    final_model_path = out_dir.joinpath("model_final.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "betas": betas,
            "config": cfg.__dict__,
            "ema_state": ema.shadow if ema else None,
        },
        final_model_path,
    )
    print(f"Saved final model weights to {final_model_path}")


# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--unet-base-channels", type=int, default=128)
    parser.add_argument(
        "--unet-channel-mults", type=int, nargs="+", default=[1, 2, 2, 4]
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--num-sample-images", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="runs/exp1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA tracking")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        embed_dim=args.embed_dim,
        unet_base_channels=args.unet_base_channels,
        unet_channel_mults=tuple(args.unet_channel_mults),
        checkpoint_interval=args.checkpoint_interval,
        sample_interval=args.sample_interval,
        num_sample_images=args.num_sample_images,
        out_dir=args.out_dir,
        resume=args.resume,
        ema_decay=None if args.no_ema else args.ema_decay,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("Successfully parsed args")
    train(cfg)
