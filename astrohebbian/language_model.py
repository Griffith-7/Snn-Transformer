"""Small causal language-model prototype using the causal attention module."""

import torch
import torch.nn as nn

from .causal import CausalAstrocyteHebbianAttention
from .model import SpikingFFN


class CausalAstrocyteLanguageModel(nn.Module):
    """Byte-level causal LM for small proof-of-concept experiments."""

    def __init__(
        self, vocab_size=256, d_model=64, seq_len=128, num_heads=4,
        attention_implementation="parallel",
    ):
        super().__init__()
        if vocab_size <= 0 or seq_len <= 0:
            raise ValueError("vocab_size and seq_len must be positive")
        self.seq_len = seq_len
        if attention_implementation not in ("recurrent", "parallel"):
            raise ValueError("attention_implementation must be recurrent or parallel")
        self.attention_implementation = attention_implementation
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalAstrocyteHebbianAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, expansion=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or token_ids.shape[1] > self.seq_len:
            raise ValueError("token_ids must have shape (batch, sequence <= seq_len)")
        hidden = self.embedding(token_ids) + self.pos_encoder[:, : token_ids.shape[1], :]
        hidden = hidden + self.attention(
            self.norm1(hidden), implementation=self.attention_implementation
        )
        hidden = hidden + self.ffn(self.norm2(hidden))
        return self.lm_head(hidden)