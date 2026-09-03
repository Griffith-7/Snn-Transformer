"""Standalone Astrocyte-Hebbian spiking Transformer components."""

from .causal import CausalAstrocyteHebbianAttention
from .language_model import CausalAstrocyteLanguageModel
from .model import (
    AstrocyteHebbianAttention,
    AstrocyteHebbianBlock,
    AstrocyteHebbianClassifier,
    SpikingFFN,
    SurrogateHeaviside,
    spike_fn,
)

__all__ = [
    "AstrocyteHebbianAttention",
    "AstrocyteHebbianBlock",
    "AstrocyteHebbianClassifier",
    "SpikingFFN",
    "SurrogateHeaviside",
    "spike_fn",
    "CausalAstrocyteHebbianAttention",
    "CausalAstrocyteLanguageModel",
]
