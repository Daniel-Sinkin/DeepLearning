"""
danielsinkin97@gmail.com

Implementation of the original 2017 *Attention Is All You Need* Transformer
architecture (only the encoder part) with optional QKV-fused projections
for faster attention. The module can be executed as a script to benchmark
a single forward pass under various back-ends (CPU, CUDA, Apple MPS) while
collecting profiler statistics.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Configs:
    """
    Compile-time switches controlling optional functionality.

    * use_fused_qkv - Route attention through a single `nn.Linear` that
      projects queries, keys and values jointly (QKV fusion). This reduces
      kernel launches and improves memory locality at the cost of marginally
      higher register pressure.

    * asserts_enabled - Enable shape and mask assertions throughout the
      model. These checks are helpful during development but incur a runtime
      hit, so they are disabled by default in production runs.

    * use_post_norm - If True, use post-layernorm architecture (residual → LN).
      If False, use pre-layernorm (LN → sub-layer → residual).
    """

    use_fused_qkv: bool = True
    asserts_enabled: bool = True
    use_post_norm: bool = True

    norm_eps: float = 1e-6  # Tensorflow default, 1e-5 is pytorch default

    @classmethod
    def print(cls) -> None:
        """Utility function that prints the settings."""
        print("Configs:")
        print(f"\tuse_fused_qkv   : {cls.use_fused_qkv}")
        print(f"\tasserts_enabled : {cls.asserts_enabled}")
        print(f"\tuse_post_norm   : {cls.use_post_norm}")


def assert_shape(x: Tensor, expected_shape: torch.Size | tuple[int, ...]) -> None:
    """Wrapper around shape assertion that is more readable"""
    if Configs.asserts_enabled:
        assert x.shape == expected_shape, f"{x.shape=} != {expected_shape=}"


def assert_same_shape(x: Tensor, y: Tensor) -> None:
    """Check that the shape of the two tensors is the same"""
    if Configs.asserts_enabled:
        assert x.shape == y.shape, f"{x.shape}!={y.shape}"


# For shape asserts so we have no magic numbers floating around
BROADCAST_SHAPE = 1
