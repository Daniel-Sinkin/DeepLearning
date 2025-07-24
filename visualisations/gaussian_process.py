from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Tensor: TypeAlias = NDArray[np.float64]

n_sampled_functions = 3


def f(x: Tensor) -> Tensor:
    return np.sin(x - 0.5)


def get_covariance_from_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = X[:, np.newaxis]
    assert X.shape == (len(X), 1)
    Y = Y[np.newaxis, :]
    assert Y.shape == (1, len(Y))
    sq_dist = (X - Y) ** 2
    assert sq_dist.shape == (len(X), len(Y))
    return np.exp(-0.5 * sq_dist)


def main() -> None:
    xs = np.linspace(-2.0, 2.0, 100)
    ys = f(xs)
    n = len(xs)

    x_train = np.array([-1.0, 0.0, 1.0])
    y_train = f(x_train)
    n_train = len(x_train)

    K_XX = get_covariance_from_kernel(x_train, x_train) + 1e-6 * np.eye(n_train)
    K_XS = get_covariance_from_kernel(x_train, xs)
    K_SS = get_covariance_from_kernel(xs, xs) + 1e-6 * np.eye(n)

    means = K_XS.T @ np.linalg.inv(K_XX) @ y_train
    covariances = K_SS - K_XS.T @ np.linalg.inv(K_XX) @ K_XS

    std = np.sqrt(np.diag(covariances))

    samples = np.random.multivariate_normal(
        means, covariances, size=n_sampled_functions
    )

    plt.figure(figsize=(10, 6))  # type: ignore
    plt.plot(xs, ys, "k--", label="True function")  # type: ignore
    plt.plot(x_train, y_train, "ro", label="Training points", zorder=10)  # type: ignore
    plt.plot(xs, means, "b", label="GP mean")  # type: ignore
    plt.fill_between(  # type: ignore
        xs.flatten(),
        means - 2 * std,
        means + 2 * std,
        color="blue",
        alpha=0.2,
        label="±2 std dev",
    )

    for i in range(n_sampled_functions):
        plt.plot(xs, samples[i], label=f"Sample {i+1}", alpha=0.8)  # type: ignore

    plt.legend()  # type: ignore
    plt.title("GP Regression with SE Kernel and Posterior Samples")  # type: ignore
    plt.xlabel("x")  # type: ignore
    plt.ylabel("f(x)")  # type: ignore
    plt.grid(True)  # type: ignore

    plt.savefig("gaussian_process.png")  # type: ignore


if __name__ == "__main__":
    main()
