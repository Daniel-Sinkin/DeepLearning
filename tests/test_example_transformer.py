import sys
from unittest import mock
import torch
import importlib


def test_cli_main_runs_cpu(monkeypatch):
    """Does the script start & stop without raising? (CPU path)."""
    import example_transformer

    class _Tiny:
        def __init__(self, *_, **__):
            pass

        def to(self, *_, **__):
            return self

        def __call__(self, *_, **__):
            return torch.zeros(1)

    monkeypatch.setattr(example_transformer, "Transformer", _Tiny, raising=True)

    monkeypatch.setattr(sys, "argv", ["example_transformer.py"])
    example_transformer.main()
