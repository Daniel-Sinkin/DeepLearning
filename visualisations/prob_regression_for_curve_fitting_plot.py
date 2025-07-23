from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

Tensor: TypeAlias = NDArray[np.float64]


def func(x: Tensor, w: float = 5.0) -> Tensor:
    return -np.sin(np.exp(1 + x**2 / w))


def normal(x: Tensor, mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)


def prob_regression_for_curve_fitting_plot() -> None:
    rng = np.random.default_rng(0)

    beta = 8.0

    xs = np.linspace(-1.0, 2.0, 100)
    ys = func(xs)
    noise_ys: Tensor = ys + rng.normal(scale=1.0 / beta, size=len(xs))
    noise: Tensor = rng.normal(scale=1.0 / beta, size=(20,))

    x0 = np.array([1.0])
    y0 = func(x0)

    ts = np.linspace(-1, 1.0, 100)
    normals = normal(ts, sigma=1 / beta) / 15.0

    plt.title("Probabilistic Regression for Curve Fitting")
    plt.plot(
        xs,
        ys,
        label=r"$y(x, w) = -\sin\left(\exp\left(1 + \frac{x^2}{w}\right)\right)$",
    )
    plt.scatter(x0, y0, label=f"({x0[0]:.2f},{y0[0]:.2f})", c="red", zorder=10)
    plt.scatter(
        [x0[0]] * len(noise), y0 + noise, label="Noise Sample", alpha=0.5, c="orange"
    )
    plt.axvline(x=x0, ls="--", color="black", alpha=0.7)
    plt.plot(x0 + normals, y0 + ts, label="Noise Distribution", c="orange")
    plt.scatter(xs, noise_ys, color="orange", alpha=0.15)
    plt.legend()
    plt.savefig("prob_regression_for_curve_fitting_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    prob_regression_for_curve_fitting_plot()
