#!/usr/bin/env python3

import sys

sys.path.append("src")

import torch
import torch.nn as nn
from quantrader.models.genome import NetworkBuilder, create_genome_from_network


def create_trading_network():
    """Create a simple feedforward network for testing."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


# Create network and check shapes
network = create_trading_network()
print("Original network state dict shapes:")
for k, v in network.state_dict().items():
    print(f"{k}: {v.shape}")

# Create genome
genome = create_genome_from_network(network)
print(f"\nGenome weight_shapes: {genome.weight_shapes}")

# Try to rebuild
rebuilt_network = NetworkBuilder.build_network(genome)
print("\nRebuilt network state dict shapes:")
for k, v in rebuilt_network.state_dict().items():
    print(f"{k}: {v.shape}")
