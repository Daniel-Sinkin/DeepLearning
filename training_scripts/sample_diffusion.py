#!/usr/bin/env python
import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torchvision.utils import save_image

# Import your project modules
from src.diffusion_model import DiffusionModel, get_beta_schedule_linear

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def default_device(name: Optional[str] = None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Sampling Core
# ------------------------------------------------------------


@torch.no_grad()
def p_sample_loop(
    model: DiffusionModel,
    betas: Tensor,
    n_samples: int,
    device: torch.device,
    capture_trajectory: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    """
    Reverse diffusion process.

    Returns:
        final samples in [-1,1] and (optionally) list of intermediate states (len=T+1)
    """
    model.eval()
    T = betas.shape[0]
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x_t = torch.randn(n_samples, 3, 32, 32, device=device)
    trajectory: list[Tensor] = []
    if capture_trajectory:
        trajectory.append(x_t.detach().cpu())

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = (
            alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        )

        eps_theta = model(x_t, t_batch)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(
            alpha_bar_t
        )
        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x_t

        if t > 0:
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(posterior_var) * noise
        else:
            x_t = mean

        if capture_trajectory and (t % max(1, T // 50) == 0 or t < 10):
            trajectory.append(x_t.detach().cpu())

    return x_t, trajectory


def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "config" not in ckpt:
        raise ValueError("Checkpoint missing 'config' dictionary.")
    return ckpt


def build_model_from_ckpt(
    ckpt: dict, device: torch.device
) -> tuple[DiffusionModel, Tensor, dict]:
    cfg_dict: dict = ckpt["config"]
    # Prefer stored betas; else regenerate
    if "betas" in ckpt:
        betas: Tensor = ckpt["betas"].to(device)
    else:
        betas = get_beta_schedule_linear(
            cfg_dict["timesteps"],
            cfg_dict["beta_start"],
            cfg_dict["beta_end"],
        ).to(device)

    model = DiffusionModel(
        T=cfg_dict["timesteps"],
        beta_t=betas,
        embed_dim=cfg_dict["embed_dim"],
        unet_base_channels=cfg_dict["unet_base_channels"],
        unet_channel_mults=tuple(cfg_dict["unet_channel_mults"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, betas, cfg_dict


def apply_ema_if_requested(
    model: DiffusionModel, ckpt: dict, use_ema: bool
) -> Optional[dict]:
    """
    If EMA is requested and available, swap params with EMA shadow.
    Returns a backup state_dict if applied (for restoration), else None.
    """
    if not use_ema:
        return None
    if "ema_state" not in ckpt or ckpt["ema_state"] is None:
        print("EMA requested but 'ema_state' not found in checkpoint.")
        return None
    ema_state: dict = ckpt["ema_state"]
    backup = {k: p.data.clone() for k, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        model_state = model.state_dict()
        for name, ema_tensor in ema_state.items():
            if name in model_state:
                model_state[name].copy_(ema_tensor.to(model_state[name].device))
    print("Applied EMA shadow weights for sampling.")
    return backup


def restore_from_backup(model: DiffusionModel, backup: Optional[dict]) -> None:
    if backup is None:
        return
    with torch.no_grad():
        model_state = model.state_dict()
        for name, tensor in backup.items():
            if name in model_state:
                model_state[name].copy_(tensor)
    print("Restored original (non-EMA) weights.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample images from a trained diffusion model checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (.pt). Accepts training ckpt or model_final.pt.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="samples_out",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of images to generate (should be a square number for a grid).",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (set <0 for random)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cpu | cuda | mps | None(auto).",
    )
    parser.add_argument(
        "--use-ema", action="store_true", help="If set, use EMA weights (if available)."
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="Optional explicit output filename for the grid.",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save each sample as its own PNG.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also save a GIF of a single sample denoising trajectory.",
    )
    parser.add_argument(
        "--gif-idx", type=int, default=0, help="Index of sample to track for GIF."
    )
    parser.add_argument(
        "--gif-fps", type=int, default=20, help="FPS for denoising GIF."
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Cast model to float16 (only if supported) for faster sampling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a progress line every N timesteps (approx).",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = default_device(args.device)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, device)
    model, betas, cfg = build_model_from_ckpt(ckpt, device)

    if args.half:
        if device.type == "cuda":
            model.half()
            betas = betas.half()
            print("Converted model and betas to float16.")
        else:
            print("Warning: --half requested but non-CUDA device; ignoring.")

    backup = apply_ema_if_requested(model, ckpt, args.use_ema)

    n = args.n_samples
    grid_side = int(math.sqrt(n))
    if grid_side * grid_side != n:
        print(
            f"Warning: n_samples={n} is not a perfect square; grid will be rectangular."
        )
    print(f"Generating {n} samples on {device}...")

    start = time.perf_counter()
    capture = args.gif  # only capture trajectory if a GIF is requested
    samples, trajectory = p_sample_loop(
        model, betas, n, device, capture_trajectory=capture
    )
    elapsed = time.perf_counter() - start
    print(f"Sampling finished in {elapsed:.2f}s ({elapsed / n:.3f}s per image).")

    # Map from [-1,1] to [0,1]
    samples_01 = (samples.clamp(-1, 1) + 1) / 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    grid_name = args.fname or f"samples_{timestamp}.png"
    grid_path = out_dir.joinpath(grid_name)
    save_image(samples_01, grid_path, nrow=grid_side)
    print(f"Saved grid: {grid_path}")

    if args.save_individual:
        indiv_dir = out_dir.joinpath(f"individual_{timestamp}")
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            save_image(samples_01[i], indiv_dir.joinpath(f"sample_{i:04d}.png"))
        print(f"Saved individual samples to {indiv_dir}")

    if args.gif:
        try:
            import imageio  # type: ignore
        except ImportError:
            print(
                "imageio not installed; skipping GIF creation. Install with 'pip install imageio'."
            )
        else:
            # Build GIF from trajectory of chosen sample index
            idx = max(0, min(args.gif_idx, n - 1))
            frames = []
            for step_img in trajectory:
                # step_img: [n, 3, 32, 32]
                x = step_img[idx : idx + 1]
                x01 = (x.clamp(-1, 1) + 1) / 2
                # Upscale for nicer viewing
                frame = torch.nn.functional.interpolate(
                    x01, size=(256, 256), mode="nearest"
                )[0]
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
                frames.append(frame_np)
            gif_path = out_dir.joinpath(f"denoise_{timestamp}_idx{idx}.gif")
            imageio.mimsave(gif_path, frames, fps=args.gif_fps)
            print(f"Saved denoising GIF: {gif_path}")

    restore_from_backup(model, backup)


if __name__ == "__main__":
    main()
