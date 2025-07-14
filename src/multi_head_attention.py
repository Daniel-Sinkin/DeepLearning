"""
danielsinkin97@gmail.com

multi_head_attention.py
"""

import math
from typing import cast

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from src.dropout import Dropout
from src.linear import Linear

from .common import Configs, assert_same_shape, assert_shape, BROADCAST_SHAPE


class _MultiHeadAttentionCore(nn.Module):
    def __init__(
        self,
        is_causal: bool,
        d_model: int,
        n_head: int,
        dropout: float,
        configs: Configs,
    ):
        super().__init__()  # type: ignore

        self.configs = configs

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_h = d_model // n_head
        self.dropout = Dropout(dropout)

        self.is_causal = is_causal
        if self.is_causal:
            self.register_buffer("_causal_mask", torch.empty(0, dtype=torch.bool))

        self.W_O = Linear(d_model, d_model, bias=True, configs=self.configs)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, len_q, d_model_q = queries.shape
        batch_k, len_k, d_model_k = keys.shape
        batch_v, len_v, d_model_v = values.shape
        assert batch == batch_k == batch_v, "batch dims differ"
        assert len_k == len_v, "key and value seq-lengths differ"
        assert (
            d_model_q == d_model_k == d_model_v == self.d_model
        ), f"{d_model_q=}, {d_model_k=}, {d_model_v=} != {self.d_model=}"

        # Avoids var shadowing
        _queries = queries.view(batch, len_q, self.n_head, self.d_h)
        _keys = keys.view(batch, len_k, self.n_head, self.d_h)
        _values = values.view(batch, len_k, self.n_head, self.d_h)

        assert_shape(_queries, (batch, len_q, self.n_head, self.d_h))
        assert_shape(_keys, (batch, len_k, self.n_head, self.d_h))
        assert_shape(_values, (batch, len_k, self.n_head, self.d_h))

        _queries: Tensor = _queries.permute(0, 2, 1, 3).contiguous()
        _keys: Tensor = _keys.permute(0, 2, 1, 3).contiguous()
        _values: Tensor = _values.permute(0, 2, 1, 3).contiguous()

        assert_shape(_queries, (batch, self.n_head, len_q, self.d_h))
        assert_shape(_keys, (batch, self.n_head, len_k, self.d_h))
        assert_shape(_values, (batch, self.n_head, len_k, self.d_h))

        similarity = torch.matmul(_queries, _keys.transpose(-2, -1)) / math.sqrt(
            self.d_h
        )
        assert_shape(similarity, (batch, self.n_head, len_q, len_k))
        neg_inf = torch.finfo(similarity.dtype).min

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            assert_shape(mask, (batch, BROADCAST_SHAPE, BROADCAST_SHAPE, len_k))
            similarity = similarity.masked_fill(mask, neg_inf)

        if self.is_causal:
            assert len_q == len_k, f"Causal masking needs {len_q=}=={len_k}"
            if self._causal_mask.size(-1) < len_q:  # type: ignore # pylint: disable=access-member-before-definition
                full_mask = torch.tril(
                    torch.ones(len_q, len_q, dtype=torch.bool, device=queries.device)
                )
                new_mask = full_mask.unsqueeze(0).unsqueeze(0)
                self._causal_mask = (  # pylint: disable=attribute-defined-outside-init
                    new_mask
                )
                assert_shape(
                    self._causal_mask, (BROADCAST_SHAPE, BROADCAST_SHAPE, len_q, len_q)
                )

            causal_mask = cast(
                Tensor, self._causal_mask[:, :, :len_q, :len_q]  # type:ignore
            )
            similarity = torch.where(causal_mask, similarity, neg_inf)

            if self.configs.asserts_enabled:
                assert_shape(
                    causal_mask, (BROADCAST_SHAPE, BROADCAST_SHAPE, len_q, len_q)
                )
                check_mask = ~causal_mask
                assert_same_shape(check_mask, causal_mask)
                masked_values = similarity[check_mask.expand_as(similarity)]
                expected = torch.full_like(masked_values, neg_inf)
                assert torch.all(
                    torch.isclose(masked_values, expected)
                ), "Causal mask failed"

        attention_weights = F.softmax(similarity, dim=-1)
        assert_shape(attention_weights, (batch, self.n_head, len_q, len_k))
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, _values)
        assert_shape(attention_output, (batch, self.n_head, len_q, self.d_h))

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        assert_shape(attention_output, (batch, len_q, self.n_head, self.d_h))

        attention_output = attention_output.view(batch, len_q, self.d_model)
        assert_shape(attention_output, (batch, len_q, self.d_model))

        output: Tensor = self.W_O(attention_output)
        assert_shape(output, (batch, len_q, self.d_model))
        output = self.dropout(output)

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        is_causal: bool,
        d_model: int,
        n_head: int,
        dropout: float,
        configs: Configs,
    ):
        """MHSA"""
        super().__init__()  # type: ignore

        self.configs = configs

        self.is_causal = is_causal
        self.core = _MultiHeadAttentionCore(
            is_causal=is_causal,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            configs=self.configs,
        )

        if self.configs.use_fused_qkv:
            self.W_QKV = Linear(d_model, 3 * d_model, bias=True, configs=self.configs)
            self.W_Q = self.W_K = self.W_V = None
        else:
            self.W_QKV = None
            self.W_Q = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_K = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_V = Linear(d_model, d_model, bias=True, configs=self.configs)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        if self.configs.asserts_enabled:
            _, _, d_model_input = x.shape
            assert x.ndim == 3, f"Expected (B, L, D), got {x.ndim=}"
            assert (
                d_model_input == self.core.d_model
            ), f"{d_model_input=} != {self.core.d_model=}"

        if self.configs.use_fused_qkv:
            if self.configs.asserts_enabled:
                assert all(weight is None for weight in (self.W_Q, self.W_K, self.W_V))
            assert self.W_QKV is not None
            qkv: Tensor = self.W_QKV(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            assert self.W_Q is not None
            assert self.W_K is not None
            assert self.W_V is not None
            q: Tensor = self.W_Q(x)
            k: Tensor = self.W_K(x)
            v: Tensor = self.W_V(x)

        assert_shape(q, (x.shape[0], x.shape[1], self.core.d_model))
        assert_same_shape(q, k)
        assert_same_shape(k, v)

        return self.core(q, k, v, key_padding_mask=key_padding_mask)


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention wrapper:
        Q comes from the decoder states,
        K and V come from the encoder states.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float, configs: Configs):
        super().__init__()  # type: ignore

        self.configs = configs

        self.core = _MultiHeadAttentionCore(
            is_causal=False,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            configs=self.configs,
        )

        if self.configs.use_fused_qkv:
            self.W_Q = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_KV = Linear(d_model, 2 * d_model, bias=True, configs=self.configs)
            self.W_K = self.W_V = None
        else:
            self.W_Q = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_K = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_V = Linear(d_model, d_model, bias=True, configs=self.configs)
            self.W_KV = None

    def forward(
        self, q_input: Tensor, kv_input: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        batch_q, _, d_q = q_input.shape
        batch_kv, _, d_kv = kv_input.shape

        if self.configs.asserts_enabled:
            assert (
                d_q == d_kv == self.core.d_model
            ), f"{d_q=}, {d_kv=} != {self.core.d_model=}"
            assert batch_q == batch_kv, f"{batch_q=} != {batch_kv=}"

        if self.configs.use_fused_qkv:
            assert self.W_K is None
            assert self.W_V is None
            assert self.W_KV is not None
            q = self.W_Q(q_input)
            kv: Tensor = self.W_KV(kv_input)
            k, v = kv.chunk(2, dim=-1)
        else:
            assert self.W_K is not None
            assert self.W_V is not None
            assert self.W_KV is None
            q: Tensor = self.W_Q(q_input)
            k: Tensor = self.W_K(kv_input)
            v: Tensor = self.W_V(kv_input)

        return self.core(q, k, v, key_padding_mask=key_padding_mask)
