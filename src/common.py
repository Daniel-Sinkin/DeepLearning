"""
danielsinkin97@gmail.com

common.py
"""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, tensor


class WeightInitType(StrEnum):
    Kaiming = "kaiming"
    Xavier = "xavier"  # With uniform


@dataclass(frozen=True)
class Configs:
    """Holds different settings for the transformer"""

    use_fused_qkv: bool = True
    use_post_norm: bool = True
    use_final_layer_norm: bool = False
    weight_init_type: WeightInitType = WeightInitType.Xavier
    tie_target_embedding_and_lm_head_weights: bool = False

    asserts_enabled: bool = True
    norm_eps: float = 1e-6  # TensorFlow default, 1e-5 is PyTorch default

    def print(self) -> None:
        """Utility function that prints the settings."""
        print("Configs:")
        print(f"\tuse_fused_qkv        : {self.use_fused_qkv}")
        print(f"\tuse_post_norm        : {self.use_post_norm}")
        print(f"\tuse_final_layer_norm : {self.use_final_layer_norm}")
        print(f"\tweight_init_type     : {self.weight_init_type}")
        print(
            f"\ttie_target_embedding_and_lm_head_weights : {self.tie_target_embedding_and_lm_head_weights}"
        )
        print(f"\tasserts_enabled      : {self.asserts_enabled}")
        print(f"\tnorm_eps             : {self.norm_eps}")


def get_default_configs() -> Configs:
    """Returns the settings that are most aligned with the 2017 paper."""
    return Configs(
        use_fused_qkv=True,
        use_post_norm=True,
        use_final_layer_norm=False,
        weight_init_type=WeightInitType.Xavier,
        tie_target_embedding_and_lm_head_weights=False,
        norm_eps=1e-6,
    )


def assert_shape(x: Tensor, expected_shape: torch.Size | tuple[int, ...]) -> None:
    """Wrapper around shape assertion that is more readable"""
    assert x.shape == expected_shape, f"{x.shape=} != {expected_shape=}"


def assert_same_shape(x: Tensor, y: Tensor) -> None:
    """Check that the shape of the two tensors is the same"""
    assert x.shape == y.shape, f"{x.shape}!={y.shape}"


# For shape asserts so we have no magic numbers floating around
BROADCAST_SHAPE = 1


def erf(x: Tensor) -> Tensor:
    return torch.special.erf(x)  # type: ignore # pylint: disable=not-callable
