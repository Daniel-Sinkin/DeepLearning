"""
danielsinkin97@gmail.com

common.py
"""

from dataclasses import dataclass
from enum import StrEnum

from numpy import isin
import torch
from torch import Tensor
from torch import nn


class WeightInitType(StrEnum):
    """
    Weight initialisation strategy
    """

    Kaiming = "kaiming"
    Xavier = "xavier"


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


def share_memory_parameters(
    target: nn.Module, target_attr: str, source: nn.Module, source_attr: str
) -> None:
    """
    Makes target.<target_attr> share memory with source.<source_attr>.

    Usually used to tie parameters like weights or biases between modules.
    """
    if not hasattr(source, source_attr):
        raise AttributeError(
            f"source module {type(source).__name__} has no attribute '{source_attr}'"
        )
    if not hasattr(target, target_attr):
        raise AttributeError(
            f"target module {type(target).__name__} has no attribute '{target_attr}'"
        )

    source_param = getattr(source, source_attr)
    if not isinstance(source_param, nn.Parameter):
        raise TypeError(
            f"source.{source_attr} must be a torch.nn.Parameter, got {type(source_param)}"
        )

    setattr(target, target_attr, source_param)


def share_memory_weight(target: nn.Module, source: nn.Module) -> None:
    """Makes the 'weight' attr of both modules share the same memory."""
    share_memory_parameters(target, "weight", source, "weight")


def share_memory_bias(target: nn.Module, source: nn.Module) -> None:
    """Makes the 'bias' attr of both modules share the same memory."""
    share_memory_parameters(target, "bias", source, "bias")
