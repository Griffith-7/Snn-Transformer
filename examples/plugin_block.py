"""Drop the Astrocyte-Hebbian block into any transformer.

This shows the plugin usage: import a reusable nn.Module and wire it into
your own architecture. The package does not impose a training loop or a full
model on you.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astrohebbian import AstrocyteHebbianAttention, AstrocyteHebbianBlock


def plug_attention_into_your_own_model():
    # A model of your own design that happens to use the plugin's attention.
    class MyModel(torch.nn.Module):
        def __init__(self, d_model=64, num_heads=4):
            super().__init__()
            self.embedding = torch.nn.Linear(3, d_model)
            self.attention = AstrocyteHebbianAttention(
                d_model=d_model, num_heads=num_heads, gradient_mode="exact"
            )
            self.norm = torch.nn.LayerNorm(d_model)
            self.head = torch.nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.embedding(x)
            x = x + self.attention(self.norm(x))
            return self.head(x)

    model = MyModel()
    logits = model(torch.randn(2, 16, 3))
    print("plug-in attention output:", tuple(logits.shape))  # (2, 16, 10)


def stack_plugin_blocks():
    block = AstrocyteHebbianBlock(d_model=64, num_heads=4, expansion=2)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == x.shape
    print("stacked plugin block output:", tuple(out.shape))


if __name__ == "__main__":
    plug_attention_into_your_own_model()
    stack_plugin_blocks()