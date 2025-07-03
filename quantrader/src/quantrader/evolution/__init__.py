"""
Evolution module initialization.
Exports NEAT and genetic algorithm components.
"""

from .neat import (
    NodeType,
    ActivationFunction,
    NodeGene,
    ConnectionGene,
    NEATConfig,
    NEATGenome,
)

from .population import Species, Population, simple_xor_fitness

__all__ = [
    # NEAT components
    "NodeType",
    "ActivationFunction",
    "NodeGene",
    "ConnectionGene",
    "NEATConfig",
    "NEATGenome",
    # Population management
    "Species",
    "Population",
    "simple_xor_fitness",
]
