from typing import Callable, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@overload
def func(x: float) -> float: ...
@overload
def func(x: FloatArray) -> FloatArray: ...


def func(x):  # type: ignore[no-untyped-def]
    return -np.sin(np.pi * 2 * x) + 0.5


def bisection_method(
    f: Callable[[float], float], left: float, right: float, atol: float
) -> list[float]:
    """
    Applies the bisection method to find the root of the function f in the interval [left, right].

    Args:
        f: The function for which to find the root.
        left: The left boundary of the interval.
        right: The right boundary of the interval.
        atol: The absolute tolerance for stopping the iteration.

    Returns:
        A list of x-values at each bisection step.
    """
    x_curr = left
    x_steps: list[float] = []

    while abs(right - left) >= atol:
        x_steps.append(x_curr)
        y_left = f(left)
        y_right = f(right)
        assert y_left * y_right < 0.0

        x_mid = (left + right) / 2.0
        y_mid = f(x_mid)

        if y_left * y_mid > 0.0:
            left = x_mid
        else:
            right = x_mid

        x_curr = x_mid

    return x_steps


def plot_bisection(
    f: Callable[[float], float],
    x_steps: list[float],
    xs: FloatArray,
    ys: FloatArray,
    atol: float,
) -> None:
    """
    Plots the function and the bisection method steps.

    Args:
        f: The function being plotted.
        x_steps: The x-values from the bisection method.
        xs: The x-values used for plotting the function.
        ys: The corresponding y-values of the function.
        atol: The absolute tolerance used in the bisection method.
    """
    plt.axhline(y=0.0, color="black", ls="--", alpha=0.7)
    plt.scatter(x_steps, list(map(f, x_steps)), color="red", zorder=10, alpha=0.3)
    plt.plot(xs, ys, label="y(x) = -sin(2pi * x) + 0.5")
    plt.title(f"Bisection Root Finding\n{len(x_steps)} Steps for {atol=:.4f}")
    plt.savefig("plots/bisection.png", dpi=300)


if __name__ == "__main__":
    atol = 1e-3
    left = 0.0
    right = 0.25
    xs: FloatArray = np.linspace(0, 0.3, 100, dtype=np.float64)
    ys: FloatArray = func(xs)

    x_steps = bisection_method(func, left, right, atol)
    plot_bisection(func, x_steps, xs, ys, atol)
