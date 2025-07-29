"""
danielsinkin97@gmail.com
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def generate_uniform_pairs(
    n_pairs: int, seed: int = 0
) -> Tuple[npt.NDArray[np.float64], np.random.Generator]:
    """
    Generate 2xn_pairs i.i.d. uniform samples on [-1, 1]^2 and return them with a reproducible RNG.

    Args:
        n_pairs: Number of pairs to generate.
        seed: Seed for the random number generator.

    Returns:
        samples: Array of shape (2, n_pairs) with values in [-1, 1].
        rng: The numpy Generator used to create the samples.
    """
    rng = np.random.default_rng(seed)
    samples: npt.NDArray[np.float64] = 2.0 * rng.random((2, n_pairs)) - 1.0
    return samples, rng


def radial_metrics(
    samples: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute squared radii and radii for 2D samples.

    Args:
        samples: Array of shape (2, n).

    Returns:
        r2: Squared radii array of shape (n,).
        r: Radii array of shape (n,).
    """
    xs, ys = samples
    r2: npt.NDArray[np.float64] = xs**2 + ys**2
    r: npt.NDArray[np.float64] = np.sqrt(r2)
    return r2, r


def selection_masks(
    r2: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Build boolean masks for points inside and outside the unit disk.

    Args:
        r2: Squared radii array of shape (n,).

    Returns:
        mask_keep: Mask for r^2 <= 1.
        mask_discard: Mask for r^2 > 1.
    """
    mask_keep: npt.NDArray[np.bool_] = r2 <= 1.0
    mask_discard: npt.NDArray[np.bool_] = ~mask_keep
    return mask_keep, mask_discard


def figure_square_disk(
    samples: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    r2: npt.NDArray[np.float64],
    mask_keep: npt.NDArray[np.bool_],
    mask_discard: npt.NDArray[np.bool_],
    n_pairs: int,
    bins_count: int,
    out_dir: Path,
) -> None:
    """
    Create and save a 1x2 figure: scatter of accepted/rejected points and radial histogram with p(r)=2r.

    Args:
        samples: Array of shape (2, n).
        r: Radii array of shape (n,).
        r2: Squared radii array of shape (n,).
        mask_keep: Mask for accepted points.
        mask_discard: Mask for rejected points.
        n_pairs: Total number of pairs drawn.
        bins_count: Number of radial histogram bins.
        out_dir: Directory to save the figure.
    """
    xs, ys = samples
    n_keep = int(mask_keep.sum())
    bins = np.linspace(0.0, 1.0, bins_count + 1)
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter.scatter(
        xs[mask_discard],
        ys[mask_discard],
        color="red",
        label="Rejected",
        alpha=0.6,
        marker="x",
        s=12,
    )
    ax_scatter.scatter(
        xs[mask_keep], ys[mask_keep], color="blue", label="Accepted", alpha=0.6, s=12
    )
    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linestyle="--")
    ax_scatter.add_artist(circle)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlim(-1.1, 1.1)
    ax_scatter.set_ylim(-1.1, 1.1)
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_title(
        f"Rejection Sampling on Uniforms\nKept {n_keep}/{n_pairs} ({n_keep/n_pairs:.2%}), expected π/4 ≈ {np.pi/4:.2%}"
    )
    ax_scatter.legend()
    ax_scatter.grid(True, linestyle=":", linewidth=0.5)
    ax_hist.hist(
        r[mask_keep],
        bins=bins,
        density=True,
        alpha=0.7,
        color="blue",
        label="Empirical",
    )
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    ax_hist.plot(r_centers, 2.0 * r_centers, "k--", label="Expected PDF: $p(r)=2r$")
    ax_hist.set_xlabel("Radius $r$")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Radial Density of Accepted Points ({bins_count} bins)")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(
        out_dir.joinpath("box_mueller_square_disk.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def figure_ring_highlight(
    samples: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    mask_keep: npt.NDArray[np.bool_],
    mask_discard: npt.NDArray[np.bool_],
    bins_count: int,
    target_r: float,
    out_dir: Path,
) -> Tuple[npt.NDArray[np.bool_], float, float]:
    """
    Create and save a 1x2 figure that highlights a radial bin both in the scatter and in the histogram.

    Args:
        samples: Array of shape (2, n).
        r: Radii array of shape (n,).
        mask_keep: Mask for accepted points.
        mask_discard: Mask for rejected points.
        bins_count: Number of radial histogram bins.
        target_r: Target radius used to pick the highlighted bin.
        out_dir: Directory to save the figure.

    Returns:
        mask_highlight: Mask selecting accepted points within the highlighted radial bin.
        lo: Lower edge of the highlighted bin.
        hi: Upper edge of the highlighted bin.
    """
    xs, ys = samples
    bins = np.linspace(0.0, 1.0, bins_count + 1)
    idx = int(np.searchsorted(bins, target_r, side="right") - 1)
    idx = int(np.clip(idx, 0, len(bins) - 2))
    lo, hi = float(bins[idx]), float(bins[idx + 1])
    if idx < len(bins) - 2:
        mask_highlight = mask_keep & (r >= lo) & (r < hi)
    else:
        mask_highlight = mask_keep & (r >= lo) & (r <= hi)
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter.scatter(
        xs[mask_discard],
        ys[mask_discard],
        color="red",
        label="Rejected",
        alpha=0.6,
        marker="x",
        s=10,
    )
    ax_scatter.scatter(
        xs[mask_keep & ~mask_highlight],
        ys[mask_keep & ~mask_highlight],
        color="blue",
        label="Accepted (other)",
        alpha=0.5,
        s=10,
    )
    ax_scatter.scatter(
        xs[mask_highlight],
        ys[mask_highlight],
        color="yellow",
        edgecolor="black",
        linewidth=0.3,
        label=f"Highlighted {lo:.2f} ≤ r < {hi:.2f}",
        s=18,
        zorder=3,
    )
    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linestyle="--")
    ax_scatter.add_artist(circle)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlim(-1.1, 1.1)
    ax_scatter.set_ylim(-1.1, 1.1)
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_title("Accepted/Rejected Points with Highlighted Radial Bin")
    ax_scatter.legend()
    ax_scatter.grid(True, linestyle=":", linewidth=0.5)
    counts, edges, patches = ax_hist.hist(
        r[mask_keep], bins=bins, density=True, alpha=0.8, label="Empirical"
    )
    for i, p in enumerate(patches):
        p.set_edgecolor("black")
        if i == idx:
            p.set_facecolor("yellow")
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    ax_hist.plot(r_centers, 2.0 * r_centers, "k--", label="Expected PDF: $p(r)=2r$")
    ax_hist.axvline(lo, linestyle=":", linewidth=1)
    ax_hist.axvline(hi, linestyle=":", linewidth=1)
    ax_hist.set_xlabel("Radius r")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(
        f"Radial Histogram (highlight bin [{lo:.2f}, {hi:.2f})) — {bins_count} bins"
    )
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(
        out_dir.joinpath("box_mueller_ring_highlight.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()
    return mask_highlight, lo, hi


def box_muller_transform(
    samples_keep: npt.NDArray[np.float64],
    r2_keep: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Apply the polar Box-Muller transform to accepted uniform pairs.

    Args:
        samples_keep: Accepted points of shape (2, m).
        r2_keep: Squared radii for the accepted points of shape (m,).

    Returns:
        z1: First standard normal sample array.
        z2: Second standard normal sample array.
        nz: Mask indicating r^2 > 0 used to avoid division by zero.
    """
    nz: npt.NDArray[np.bool_] = r2_keep > 0.0
    u = samples_keep[:, nz]
    s = r2_keep[nz]
    factor: npt.NDArray[np.float64] = np.sqrt(-2.0 * np.log(s) / s)
    z = u * factor
    z1, z2 = z[0], z[1]
    return z1, z2, nz


def figure_normals(
    z1: npt.NDArray[np.float64],
    z2: npt.NDArray[np.float64],
    out_dir: Path,
) -> None:
    """
    Create and save a 1x2 figure: scatter of (Z1, Z2) and histogram of Z2 with N(0,1) overlay.

    Args:
        z1: First standard normal sample array.
        z2: Second standard normal sample array.
        out_dir: Directory to save the figure.
    """
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter.scatter(z1, z2, alpha=0.5, s=10)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("Z1 ~ N(0,1)")
    ax_scatter.set_ylabel("Z2 ~ N(0,1)")
    ax_scatter.set_title("Box-Muller: Scatter of Standard Normals")
    ax_scatter.grid(True, linestyle=":", linewidth=0.5)
    ax_hist.hist(z2, bins=30, density=True, alpha=0.75, label="Empirical")
    x = np.linspace(min(-4.0, float(z2.min())), max(4.0, float(z2.max())), 400)
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
    ax_hist.plot(x, phi, "k--", label="N(0,1) PDF")
    ax_hist.set_xlabel("y-normal")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Histogram of y-normal with N(0,1) overlay")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(
        out_dir.joinpath("box_mueller_normals.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def figure_ring_mapping(
    z1: npt.NDArray[np.float64],
    z2: npt.NDArray[np.float64],
    highlight_nz: npt.NDArray[np.bool_],
    lo: float,
    hi: float,
    out_dir: Path,
) -> None:
    """
    Create and save a figure showing how the highlighted annulus maps under the Box-Muller transform.

    Args:
        z1: First standard normal sample array.
        z2: Second standard normal sample array.
        highlight_nz: Mask for the transformed points corresponding to the highlighted ring.
        lo: Lower edge of the highlighted radial bin in the unit disk.
        hi: Upper edge of the highlighted radial bin in the unit disk.
        out_dir: Directory to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(
        z1[~highlight_nz], z2[~highlight_nz], alpha=0.25, s=8, label="Other points"
    )
    ax.scatter(
        z1[highlight_nz],
        z2[highlight_nz],
        color="yellow",
        edgecolor="black",
        linewidth=0.3,
        s=18,
        zorder=3,
        label=f"From {lo:.2f} ≤ r < {hi:.2f}",
    )
    rho_in = float(np.sqrt(-2.0 * np.log(hi**2)))
    rho_out = float(np.sqrt(-2.0 * np.log(lo**2)))
    for rho in (rho_in, rho_out):
        circ = plt.Circle(  # type: ignore
            (0, 0), rho, color="black", linestyle="--", fill=False, alpha=0.6
        )
        ax.add_artist(circ)
    ax.set_aspect("equal")
    ax.set_xlabel("Z1 ~ N(0,1)")
    ax.set_ylabel("Z2 ~ N(0,1)")
    ax.set_title("Box-Muller: Image of Highlighted Ring")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(
        out_dir.joinpath("box_mueller_ring_mapped.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def main() -> None:
    """
    Run the end-to-end visualization pipeline and save all four figures to disk.
    """
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pairs = 3000
    bins_count = 12
    target_r = 0.60
    samples, _ = generate_uniform_pairs(n_pairs=n_pairs, seed=0)
    r2, r = radial_metrics(samples)
    mask_keep, mask_discard = selection_masks(r2)
    figure_square_disk(
        samples, r, r2, mask_keep, mask_discard, n_pairs, bins_count, out_dir
    )
    mask_highlight, lo, hi = figure_ring_highlight(
        samples, r, mask_keep, mask_discard, bins_count, target_r, out_dir
    )
    samples_keep = samples[:, mask_keep]
    r2_keep = r2[mask_keep]
    z1, z2, nz = box_muller_transform(samples_keep, r2_keep)
    highlight_keep = mask_highlight[mask_keep]
    highlight_nz = highlight_keep[nz]
    figure_normals(z1, z2, out_dir)
    figure_ring_mapping(z1, z2, highlight_nz, lo, hi, out_dir)


if __name__ == "__main__":
    main()
