"""
danielsinkin97@gmail.com
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def generate_error_surfaces(
    theta_bounds: Tuple[float, float],
    num_points: int,
    x: Tuple[float, float],
    y: float,
    param: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    theta0 = np.linspace(theta_bounds[0], theta_bounds[1], num_points)
    theta1 = np.linspace(theta_bounds[0], theta_bounds[1], num_points)
    theta0_grid, theta1_grid = np.meshgrid(theta0, theta1)

    E = (theta0_grid * x[0] + theta1_grid * x[1] - y) ** 2
    E_ridge = E + param * (theta0_grid**2 + theta1_grid**2)

    return theta0_grid, theta1_grid, E, E_ridge, (theta0, theta1)


def plot_error_surfaces() -> None:
    r"""
    Inspired by this: https://stats.stackexchange.com/a/151351

    Overfitting tends to go hand-in-hand with large parameter sizes, so to alleviate overfitting
    we want to penalise large weights. For linear regression the usual error
    $$
    E(\theta) := ||f(\underline{x}; \theta) - \underline{y}||^2
    $$
    gets an additional term dependant on the penalisation factor $\lambda \geq 0$
    $$
    \hat{E}_\lambda(\theta) = E(\theta) + \lambda ||\theta||^2
    $$
    and we call the modified problem Ridge Regression. The word ridge here refers to how the error surface looks,
    it can have a "ridge" in the surface which is unstable, the additional penalty term forces the model
    to become more convex.
    """
    theta_bounds = (-5, 5)
    num_points = 300
    x = (1.0, 1.01)
    y = 2.0
    lambd = 10.0

    theta0_grid, theta1_grid, E, E_ridge, _ = generate_error_surfaces(
        theta_bounds, num_points, x, y, lambd
    )

    slope = -x[0] / x[1]
    theta0_ridge_line = np.linspace(theta_bounds[0], theta_bounds[1], 200)
    theta1_ridge_line = slope * theta0_ridge_line

    mask = (theta1_ridge_line >= theta_bounds[0]) & (
        theta1_ridge_line <= theta_bounds[1]
    )
    theta0_ridge_line = theta0_ridge_line[mask]
    theta1_ridge_line = theta1_ridge_line[mask]
    ridge_line_error = (theta0_ridge_line * x[0] + theta1_ridge_line * x[1] - y) ** 2

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(theta0_grid, theta1_grid, E, cmap="viridis", alpha=0.9, zorder=1)
    ax1.plot(
        theta0_ridge_line,
        theta1_ridge_line,
        ridge_line_error,
        color="red",
        linewidth=4,
        zorder=100,
        label="Ridge Line",
    )
    ax1.set_xlim(theta_bounds)
    ax1.set_ylim(theta_bounds)
    ax1.set_title("Least Squared Error Surface")
    ax1.set_xlabel("$\\theta_0$")
    ax1.set_ylabel("$\\theta_1$")
    ax1.set_zlabel("Error")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(theta0_grid, theta1_grid, E_ridge, cmap="plasma", alpha=0.9)
    ax2.set_xlim(theta_bounds)
    ax2.set_ylim(theta_bounds)
    ax2.set_title("Ridge Regression Error Surface")
    ax2.set_xlabel("$\\theta_0$")
    ax2.set_ylabel("$\\theta_1$")
    ax2.set_zlabel("Error")

    plt.tight_layout()
    plt.savefig("plots/ridge_regression.png", dpi=300)


if __name__ == "__main__":
    plot_error_surfaces()
