"""
danielsinkin97@gmail.com
"""

import os
from enum import StrEnum
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal


class TargetFunction(StrEnum):
    """
    ELONGATED_GAUSSIAN is a single Ellipse

    MIXTURE_OF_GAUSSIANS consists of 5 different gaussians combined as a MoG distribution
    """

    ELONGATED_GAUSSIAN = "elongated_gaussian"
    MIXTURE_OF_GAUSSIANS = "mixture_of_gaussians"


def create_target_distribution(
    target_type: TargetFunction,
) -> Tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
    """
    Create target distribution based on the selected type

    Parameters:
    -----------
    target_type : TargetFunction
        Type of target distribution

    Returns:
    --------
    target_pdf : callable
        Target probability density function
    target_mean : array-like
        Mean of the target distribution (for single Gaussian)
    target_cov : array-like
        Covariance matrix of the target distribution (for single Gaussian)
    """
    if target_type == TargetFunction.ELONGATED_GAUSSIAN:
        target_mean = np.array([0.0, 0.0])
        target_cov = np.array([[2.0, 0.8], [0.8, 1.0]])
        target_pdf = lambda x: multivariate_normal.pdf(x, target_mean, target_cov)
    elif target_type == TargetFunction.MIXTURE_OF_GAUSSIANS:
        means = [
            np.array([-2.0, -2.0]),
            np.array([2.0, -2.0]),
            np.array([0.0, 2.0]),
            np.array([-3.0, 1.0]),
            np.array([3.0, 1.0]),
        ]

        covs = [
            np.array([[0.5, 0.2], [0.2, 0.5]]),
            np.array([[0.8, -0.3], [-0.3, 0.6]]),
            np.array([[0.6, 0.0], [0.0, 0.4]]),
            np.array([[0.4, 0.1], [0.1, 0.7]]),
            np.array([[0.7, 0.3], [0.3, 0.5]]),
        ]

        weights = np.array([0.25, 0.2, 0.2, 0.15, 0.2])

        def mog_pdf(x: np.ndarray) -> float:
            pdf_value = 0
            for i in range(5):
                pdf_value += weights[i] * multivariate_normal.pdf(x, means[i], covs[i])
            return pdf_value

        target_pdf = mog_pdf
        target_mean = np.average(means, weights=weights, axis=0)
        target_cov = np.zeros((2, 2))
        for i in range(5):
            diff = means[i] - target_mean
            target_cov += weights[i] * (covs[i] + np.outer(diff, diff))
    else:
        raise ValueError(f"Unsupported {target_type=}")

    return target_pdf, target_mean, target_cov


def metropolis_sampler_2d(
    initial_x: np.ndarray,
    n_samples: int,
    proposal_cov: np.ndarray,
    target_pdf: Callable[[np.ndarray], float],
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    2D Metropolis sampler that tracks accepted path and rejected proposals

    Parameters:
    -----------
    initial_x : array-like, shape (2,)
        Initial position for the sampler
    n_samples : int
        Number of samples to generate
    proposal_cov : array-like, shape (2, 2)
        Covariance matrix for the proposal distribution
    target_pdf : callable
        Target probability density function

    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Generated samples
    rejected_proposals : list of tuples
        List of (start_point, end_point) for rejected proposals
    """
    samples = np.zeros((n_samples, 2))
    samples[0] = initial_x

    rejected_proposals: List[Tuple[np.ndarray, np.ndarray]] = []

    for t in range(1, n_samples):
        current = samples[t - 1]

        proposed = multivariate_normal.rvs(mean=current, cov=proposal_cov)

        p_current = target_pdf(current)
        p_proposed = target_pdf(proposed)
        acceptance_ratio = min(1, p_proposed / p_current)

        if np.random.rand() < acceptance_ratio:
            samples[t] = proposed
        else:
            samples[t] = current
            rejected_proposals.append((current.copy(), proposed.copy()))

    return samples, rejected_proposals


def plot_heatmap(
    samples: np.ndarray,
    target_type: TargetFunction,
    target_pdf: Callable[[np.ndarray], float],
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    bins: int = 120,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create heatmap visualization of sample density

    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Samples from the Metropolis sampler
    target_type : TargetFunction
        Type of target distribution
    target_pdf : callable
        Target probability density function
    target_mean : array-like, shape (2,)
        Mean of the target distribution
    target_cov : array-like, shape (2, 2)
        Covariance matrix of the target distribution
    bins : int
        Number of bins for the 2D histogram
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    n_samples = len(samples)
    title = f'Sample Density Heatmap (n={n_samples})\nTarget: {target_type.value.replace("_", " ").title()}'
    ax.set_title(title, fontsize=16)

    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = ax.imshow(hist.T, origin="lower", extent=extent, cmap="hot", aspect="equal")

    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = target_pdf(pos[i, j])

    ax.contour(X, Y, Z, levels=10, colors="cyan", alpha=0.5, linewidths=1)

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    plt.colorbar(im, ax=ax, label="Sample count")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

    print(f"\nSample mean: {np.mean(samples, axis=0)}")
    print(f"Target mean: {target_mean}")
    print(f"\nSample covariance:\n{np.cov(samples.T)}")
    print(f"\nTarget covariance:\n{target_cov}")

    return fig


def plot_path(
    samples: np.ndarray,
    rejected_proposals: List[Tuple[np.ndarray, np.ndarray]],
    target_type: TargetFunction,
    target_pdf: Callable[[np.ndarray], float],
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create path visualization showing accepted samples and rejected proposals

    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Accepted samples from the Metropolis sampler
    rejected_proposals : list of tuples
        List of (start_point, end_point) for rejected proposals
    target_type : TargetFunction
        Type of target distribution
    target_pdf : callable
        Target probability density function
    target_mean : array-like, shape (2,)
        Mean of the target distribution
    target_cov : array-like, shape (2, 2)
        Covariance matrix of the target distribution
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    n_samples = len(samples)
    title = f'2D Metropolis Sampling Path (n={n_samples})\nTarget: {target_type.value.replace("_", " ").title()}\nGreen: Accepted path, Red: Rejected proposals'
    ax.set_title(title, fontsize=16)

    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = target_pdf(pos[i, j])

    ax.contour(X, Y, Z, levels=10, colors="blue", alpha=0.3)

    for start, end in rejected_proposals:
        ax.plot([start[0], end[0]], [start[1], end[1]], "r-", alpha=0.5, linewidth=1)

    ax.plot(
        samples[:, 0],
        samples[:, 1],
        "g-",
        alpha=0.8,
        linewidth=2,
        label="Accepted path",
    )

    ax.plot(samples[0, 0], samples[0, 1], "ko", markersize=12, label="Start")
    ax.plot(samples[-1, 0], samples[-1, 1], "k*", markersize=18, label="End")

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved path plot to {save_path}")

    acceptance_rate = len(np.unique(samples, axis=0)) / n_samples
    print(
        f"Path visualization - Acceptance rate: {acceptance_rate:.2%}, Rejected proposals: {len(rejected_proposals)}"
    )

    return fig


def main(
    target_function: TargetFunction = TargetFunction.ELONGATED_GAUSSIAN,
) -> None:
    """
    Main function to run the 2D Metropolis sampling visualization

    Parameters:
    -----------
    target_function : TargetFunction
        Type of target distribution to use

    Returns:
    --------
    fig_path : Figure
        Path visualization figure
    fig_heatmap : Figure
        Heatmap visualization figure
    """
    np.random.seed(0)

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    if target_function == TargetFunction.ELONGATED_GAUSSIAN:
        n_samples_path = 100
    else:
        n_samples_path = 1000
    n_samples_heatmap = 10000
    initial_x = np.array([3.0, 3.0])

    proposal_cov = np.eye(2) * 0.1

    target_pdf, target_mean, target_cov = create_target_distribution(target_function)

    print(
        f"Running 2D Metropolis sampler for {target_function.value.replace('_', ' ')}..."
    )
    samples_path, rejected_proposals = metropolis_sampler_2d(
        initial_x, n_samples_path, proposal_cov, target_pdf
    )

    # Generate save path for path plot
    path_save_path = f"plots/mcmc/mcmc_{target_function.value}_path.png"
    plot_path(
        samples_path,
        rejected_proposals,
        target_function,
        target_pdf,
        target_mean,
        target_cov,
        save_path=path_save_path,
    )
    plt.show()

    print(f"\nRunning 2D Metropolis sampler for heatmap (n={n_samples_heatmap})...")
    samples_heatmap, _ = metropolis_sampler_2d(
        initial_x, n_samples_heatmap, proposal_cov, target_pdf
    )

    heatmap_save_path = f"plots/mcmc/mcmc_{target_function.value}_heatmap.png"
    plot_heatmap(
        samples_heatmap,
        target_function,
        target_pdf,
        target_mean,
        target_cov,
        save_path=heatmap_save_path,
    )
    plt.show()


if __name__ == "__main__":
    main(TargetFunction.ELONGATED_GAUSSIAN)
    main(TargetFunction.MIXTURE_OF_GAUSSIANS)
