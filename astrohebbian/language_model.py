"""Small causal language-model prototype using the causal attention module."""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .causal import CausalAstrocyteHebbianAttention
from .model import SpikingFFN


class CausalAstrocyteHebbianBlock(nn.Module):
    """Pre-norm Transformer block using Causal Astrocyte-Hebbian attention."""

    def __init__(self, d_model=128, num_heads=4, expansion=4, alpha=10.0, gradient_mode="exact"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalAstrocyteHebbianAttention(
            d_model, num_heads, alpha=alpha, gradient_mode=gradient_mode
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, expansion, gradient_mode=gradient_mode)

    def forward(
        self,
        x: torch.Tensor,
        implementation: str = "recurrent",
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        normed_x = self.norm1(x)
        if return_state or state is not None:
            attn_out, next_state = self.attention(
                normed_x, implementation=implementation, state=state, return_state=True
            )
        else:
            attn_out = self.attention(normed_x, implementation=implementation)
            next_state = None

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_state or state is not None:
            return x, next_state
        return x


class CausalAstrocyteLanguageModel(nn.Module):
    """Byte-level causal LM supporting deep stacked blocks and streaming execution."""

    def __init__(
        self,
        vocab_size=256,
        d_model=64,
        seq_len=128,
        num_heads=4,
        num_layers=1,
        expansion=2,
        alpha=10.0,
        attention_implementation="parallel",
        gradient_mode="exact",
    ):
        super().__init__()
        if vocab_size <= 0 or seq_len <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, seq_len, and num_layers must be positive")
        self.seq_len = seq_len
        if attention_implementation not in ("recurrent", "parallel"):
            raise ValueError("attention_implementation must be recurrent or parallel")
        if gradient_mode not in ("surrogate", "exact", "reciprocal"):
            raise ValueError("gradient_mode must be 'exact', 'surrogate', or 'reciprocal'")
        self.gradient_mode = gradient_mode
        self.attention_implementation = attention_implementation
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CausalAstrocyteHebbianBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    expansion=expansion,
                    alpha=alpha,
                    gradient_mode=gradient_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_states: bool = False,
    ):
        if (
            token_ids.ndim != 2
            or start_pos < 0
            or start_pos + token_ids.shape[1] > self.seq_len
        ):
            raise ValueError("token_ids must have shape (batch, sequence), start_pos >= 0, and fit in seq_len")
        seq_len_in = token_ids.shape[1]
        hidden = self.embedding(token_ids) + self.pos_encoder[:, start_pos : start_pos + seq_len_in, :]

        impl = "recurrent" if (states is not None or return_states) else self.attention_implementation

        next_states = [] if (return_states or states is not None) else None
        for i, block in enumerate(self.blocks):
            block_state = states[i] if states is not None else None
            if return_states or states is not None:
                hidden, next_st = block(
                    hidden,
                    implementation=impl,
                    state=block_state,
                    return_state=True,
                )
                next_states.append(next_st)
            else:
                hidden = block(hidden, implementation=impl)

        logits = self.lm_head(self.norm_f(hidden))
        if return_states or states is not None:
            return logits, next_states
        return logits