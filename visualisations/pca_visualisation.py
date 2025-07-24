from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from scipy.stats import chi2

Tensor: TypeAlias = NDArray[np.float64]

rng = np.random.default_rng(0)


def get_d_conf() -> float:
    p = 0.954
    d_squared = chi2.ppf(p, df=2)
    return np.sqrt(d_squared)


def main() -> None:
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.5, 0.5], [0.5, 1.5]])

    samples = rng.multivariate_normal(mu, sigma, size=100)

    plt.scatter(samples[:, 0], samples[:, 1])

    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    for d in [get_d_conf()]:
        width, height = 2 * np.sqrt(eigvals * d**2)
        ellipse = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            edgecolor="black",
            fc="none",
            lw=1.5,
            label=f"{d}σ",
            alpha=0.4,
        )
        plt.gca().add_patch(ellipse)

    for i in range(2):
        vec = eigvecs[:, i]
        length = np.sqrt(eigvals[i]) * get_d_conf()
        arrow_end = mu + vec * length

        plt.arrow(
            mu[0],
            mu[1],
            vec[0] * length,
            vec[1] * length,
            color="red",
            width=0.03,
            head_width=0.2,
            length_includes_head=True,
            zorder=5,
        )

        plt.text(
            arrow_end[0] + 0.2,
            arrow_end[1],
            f"e_{i+1}",
            color="red",
            fontsize=10,
            ha="left",
            va="center",
        )

    plt.xlim((-5, 5))
    plt.ylim((-4, 4))
    plt.savefig("pca_visualisation.png", dpi=300)


if __name__ == "__main__":
    main()
