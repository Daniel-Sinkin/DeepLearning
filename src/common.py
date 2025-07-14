"""
danielsinkin97@gmail.com

common.py
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Debug:
    """Debug Settings"""

    asserts_enabled: bool = False


@dataclass(frozen=True)
class Configs:
    """Holds different settings for the transforemr"""

    use_fused_qkv: bool = True
    use_post_norm: bool = True
    use_final_layer_norm: bool = False
    use_original_init: bool = True  # xavier with uniform bias
    tie_target_embedding_and_lm_head_weights: bool = False

    norm_eps: float = 1e-6  # Tensorflow default, 1e-5 is pytorch default

    @classmethod
    def print(cls) -> None:
        """Utility function that prints the settings."""
        print("Configs:")
        print(f"\tuse_fused_qkv        : {cls.use_fused_qkv}")
        print(f"\tuse_post_norm        : {cls.use_post_norm}")
        print(f"\tuse_final_layer_norm : {cls.use_final_layer_norm}")
        print(f"\tuse_original_init    : {cls.use_original_init}")
        print(f"\tnorm_eps             : {cls.norm_eps}")


def get_default_configs() -> Configs:
    """Returns the settings that are most aligned with the 2017 paper."""
    return Configs(
        use_fused_qkv=True,
        use_post_norm=True,
        use_final_layer_norm=False,
        use_original_init=True,
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )


def assert_shape(x: Tensor, expected_shape: torch.Size | tuple[int, ...]) -> None:
    """Wrapper around shape assertion that is more readable"""
    if Debug.asserts_enabled:
        assert x.shape == expected_shape, f"{x.shape=} != {expected_shape=}"


def assert_same_shape(x: Tensor, y: Tensor) -> None:
    """Check that the shape of the two tensors is the same"""
    if Debug.asserts_enabled:
        assert x.shape == y.shape, f"{x.shape}!={y.shape}"


# For shape asserts so we have no magic numbers floating around
BROADCAST_SHAPE = 1
