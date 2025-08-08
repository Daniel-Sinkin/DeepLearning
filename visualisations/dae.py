"""
danielsinkin97@gmail.com
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_dae_vector_field(
    manifold_points,
    grid_lim=2.5,
    grid_res=40,
    alpha=3.0,
    const=0.4,
    manifold_label="Data manifold",
    title="Denoising Autoencoder Vector Field",
    color="red",
    filepath: Path | None = None,
) -> None:
    """
    Plots a DAE-like vector field showing projection of noisy points to the manifold.

    Parameters
    ----------
    manifold_points : np.ndarray, shape (N, 2)
        The 2D coordinates of the manifold.
    grid_lim : float
        The +/- range of the grid.
    grid_res : int
        Number of grid points along each axis.
    alpha, const : float
        Parameters for nonlinear scaling of vector lengths.
    manifold_label : str
        Label for the manifold in the plot.
    title : str
        Plot title.
    color : str
        Color for manifold line/points.
    """

    def closest_point_on_manifold(point, manifold_points):
        diffs = manifold_points - point
        dists = np.linalg.norm(diffs, axis=1)
        return manifold_points[np.argmin(dists)]

    # Grid of points
    grid_x, grid_y = np.meshgrid(
        np.linspace(-grid_lim, grid_lim, grid_res),
        np.linspace(-grid_lim, grid_lim, grid_res),
    )
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Projection
    projected_points = np.array(
        [closest_point_on_manifold(p, manifold_points) for p in grid_points]
    )
    vectors = projected_points - grid_points

    # Nonlinear scaling
    lengths = np.linalg.norm(vectors, axis=1)
    scales = const / (1 + alpha * lengths)
    vectors_scaled = vectors * scales[:, None]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(
        manifold_points[:, 0],
        manifold_points[:, 1],
        color,
        linewidth=2,
        label=manifold_label,
    )
    plt.quiver(
        grid_points[:, 0],
        grid_points[:, 1],
        vectors_scaled[:, 0],
        vectors_scaled[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.6,
        label="DAE vector field (nonlinear scaled)",
    )
    plt.scatter(manifold_points[:, 0], manifold_points[:, 1], color=color, s=10)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    plt.show()


def example_h_line() -> None:
    x_line = np.linspace(-2.5, 2.5, 200)
    y_line = np.zeros_like(x_line)
    line = np.vstack((x_line, y_line)).T

    plot_dae_vector_field(
        manifold_points=line,
        manifold_label="Data manifold (horizontal line)",
        title="Denoising Autoencoder: Vector field toward horizontal manifold",
        color="red",
        filepath=Path("plots/dae/dae_spiral.png"),
    )


def example_spiral() -> None:
    theta = np.linspace(0, 4 * np.pi, 200)
    r = np.linspace(0.5, 2.0, 200)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)
    spiral = np.vstack((x_spiral, y_spiral)).T

    plot_dae_vector_field(
        manifold_points=spiral,
        manifold_label="Data manifold (spiral)",
        title="Denoising Autoencoder: Nonlinear scaling of vector field (Spiral)",
        color="red",
        filepath=Path("plots/dae/dae_h_line.png"),
    )


def main() -> None:
    example_spiral()
    example_h_line()


if __name__ == "__main__":
    main()
