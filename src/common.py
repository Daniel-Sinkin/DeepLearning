"""
danielsinkin97@gmail.com
"""

from enum import StrEnum

import torch
from torch import Tensor, nn


class WeightInitType(StrEnum):
    """
    Weight initialisation strategy
    """

    Kaiming = "kaiming"
    Xavier = "xavier"


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
