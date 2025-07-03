"""
Models module initialization.
Exports genome and neural network components.
"""

from .genome import (
    NetworkArchitecture,
    GenomeMetadata,
    NeuralGenome,
    GenomeSerializer,
    NetworkBuilder,
)

from .networks import (
    FeedForwardNetwork,
    LSTMNetwork,
    TransformerNetwork,
    EnsembleNetwork,
    NetworkFactory,
)

__all__ = [
    # Genome classes
    "NetworkArchitecture",
    "GenomeMetadata",
    "NeuralGenome",
    "GenomeSerializer",
    "NetworkBuilder",
    # Network classes
    "FeedForwardNetwork",
    "LSTMNetwork",
    "TransformerNetwork",
    "EnsembleNetwork",
    "NetworkFactory",
]
