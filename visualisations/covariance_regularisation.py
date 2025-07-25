"""
danielsinkin97@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def main() -> None:
    """
    Visualizes the effect of increasing regularization Sigma + alpha * I on a
    skewed covariance matrix by plotting contours of the resulting
    multivariate normal distributions and annotating eigenvalues and
    eigenvectors.
    """
    eigvals = np.array([20, 1])
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov = R @ np.diag(eigvals) @ R.T

    alphas: list[float] = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    x, y = np.mgrid[-10:10:0.05, -10:10:0.05]
    pos = np.dstack((x, y))
    mean = np.array([0, 0])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # type: ignore
    for ax, alpha in zip(axes.flat, alphas):
        reg_cov = cov + alpha * np.eye(2)
        eigvals_reg, eigvecs_reg = np.linalg.eigh(reg_cov)
        order = np.argsort(eigvals_reg)[::-1]
        eigvals_reg = eigvals_reg[order]
        eigvecs_reg = eigvecs_reg[:, order]

        for i in range(eigvecs_reg.shape[1]):
            if eigvecs_reg[1, i] < 0:
                eigvecs_reg[:, i] *= -1

        ratio = eigvals_reg[0] / eigvals_reg[1]
        rv = multivariate_normal(mean, reg_cov)  # type: ignore
        ax.contour(x, y, rv.pdf(pos), levels=5)  # type: ignore

        for i, (length, direction) in enumerate(zip(eigvals_reg, eigvecs_reg.T)):
            vec = np.sqrt(length) * direction
            ax.arrow(
                mean[0],
                mean[1],
                vec[0],
                vec[1],
                width=0.05,
                head_width=0.3,
                color="red",
                alpha=0.9,
            )
            ax.text(
                vec[0] * 1.1,
                vec[1] * 1.1,
                f"lambda{i+1} = {length:.2f}",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.6, pad=1),
            )

        ax.set_title(f"alpha = {alpha}\nlambda1/lambda2 = {ratio:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(alphas) :]:
        ax.axis("off")

    fig.suptitle(r"Effect of Regularization on Eigenvalues", fontsize=16)  # type: ignore
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig("covariance_regularisation.png", dpi=300)  # type: ignore
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
