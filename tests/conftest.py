"""Global fixtures & helpers used by the entire test-suite."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="function")
def _deterministic() -> Iterator[None]:  # type: ignore
    """
    Make every test repeatable:
      * Torch
      * NumPy
      * Python's random
    """
    torch.manual_seed(0)  # type: ignore
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Project root - handy when tests need to touch files."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def device() -> torch.device:
    """CPU-only so CI boxes don't explode."""
    return torch.device("cpu")
