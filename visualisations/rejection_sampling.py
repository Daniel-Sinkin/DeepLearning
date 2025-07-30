from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np

FloatArray: TypeAlias = np.ndarray


@dataclass
class RejectionSamplingData:
    z: FloatArray
    p_prime: FloatArray
    k: float
    q: FloatArray
    mus: FloatArray
    sigmas: FloatArray


def normal_pdf(z: FloatArray, mu: float, sigma: float) -> FloatArray:
    """Evaluates the probability density function of a univariate normal distribution."""
    z = z.astype(np.float64)
    mu = np.float64(mu)
    sigma = np.float64(sigma)
    inv = np.float64(1.0) / (sigma * np.sqrt(np.float64(2.0) * np.pi))
    return inv * np.exp(np.float64(-0.5) * ((z - mu) / sigma) ** np.float64(2.0))


def compute_rejection_sampling_data() -> RejectionSamplingData:
    """Constructs an unnormalized mixture of Gaussians and a scaled proposal distribution for rejection sampling."""
    mus = np.array([-2.0, 0.75, 3.5], dtype=np.float64)
    sigmas = np.array([0.6, 0.9, 1.2], dtype=np.float64)
    z_min = float(mus.min() - 6.0 * sigmas.max())
    z_max = float(mus.max() + 6.0 * sigmas.max())
    z = np.linspace(z_min, z_max, 3000, dtype=np.float64)
    components = [normal_pdf(z, mu, sigma) for mu, sigma in zip(mus, sigmas)]
    p_prime = np.sum(components, axis=0).astype(np.float64)
    q_mu = float(np.mean(mus))
    q_sigma = float(2.5 * sigmas.max())
    q = normal_pdf(z, q_mu, q_sigma)
    ratio = p_prime / q
    k = 1.05 * float(np.max(ratio))
    return RejectionSamplingData(z=z, p_prime=p_prime, k=k, q=q, mus=mus, sigmas=sigmas)


def plot_rejection_sampling(data: RejectionSamplingData) -> None:
    """Plots the rejection sampling setup, including the target density, proposal envelope, and rejection region."""
    plt.figure(figsize=(8, 4.8))
    plt.plot(data.z, data.p_prime, linewidth=2, label="p'(z) = sum of 3 Gaussians")
    plt.plot(
        data.z,
        data.k * data.q,
        linestyle="--",
        linewidth=2,
        label="k · q(z) (envelope)",
    )
    plt.fill_between(
        data.z,
        data.p_prime,
        data.k * data.q,
        alpha=0.3,
        color="gray",
        label="rejection region",
    )
    for mu, sigma in zip(data.mus, data.sigmas):
        plt.plot(data.z, normal_pdf(data.z, mu, sigma), linewidth=1, alpha=0.5)
    plt.title("Rejection Sampling Illustration: Envelope k·q(z) ≥ p'(z)")
    plt.xlabel("z")
    plt.ylabel("density (unnormalized)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/rejection_sampling.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data = compute_rejection_sampling_data()
    plot_rejection_sampling(data)
