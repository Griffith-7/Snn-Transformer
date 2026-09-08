"""Example wrappers built on top of the plugin API.

AstrocyteHebbianClassifier and CausalAstrocyteLanguageModel are whole-model
examples assembled from the plugin components -- handy starting points for a
sequence classifier or a small byte-level causal language model.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astrohebbian import AstrocyteHebbianClassifier, CausalAstrocyteLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classifier_example():
    model = AstrocyteHebbianClassifier(
        input_dim=1,
        d_model=64,
        seq_len=784,
        num_classes=10,
        num_heads=4,
        v_levels=1,
        gradient_mode="exact",
    ).to(DEVICE)
    pixels = torch.randn(4, 784, 1, device=DEVICE)
    logits = model(pixels)
    print("classifier logits:", tuple(logits.shape))  # (4, 10)


def causal_lm_example():
    model = CausalAstrocyteLanguageModel(
        vocab_size=256,
        d_model=64,
        seq_len=128,
        num_heads=4,
        gradient_mode="exact",
    ).to(DEVICE)
    tokens = torch.randint(0, 256, (2, 128), device=DEVICE)
    logits = model(tokens)
    print("causal LM logits:", tuple(logits.shape))  # (2, 128, 256)


if __name__ == "__main__":
    classifier_example()
    causal_lm_example()