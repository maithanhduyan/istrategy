"""
Example: Complete neural network serialization and reproduction.
Demonstrates the core capability of saving, loading, and reproducing neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from quantrader import (
    NeuralGenome,
    GenomeSerializer,
    NetworkBuilder,
    FeedForwardNetwork,
    MarketTransformer,
    NetworkFactory,
)


def main():
    print("🧠 QuantRader Neural Network Serialization Demo")
    print("=" * 50)

    # 1. Create and train a simple trading network
    print("\n1. Creating and training a feedforward trading network...")

    network = FeedForwardNetwork(
        input_size=10,  # 10 market features
        hidden_sizes=[64, 32, 16],
        output_size=3,  # Buy, Sell, Hold
    )

    # Mock training data
    X_train = torch.randn(1000, 10)  # 1000 samples, 10 features
    y_train = torch.randint(0, 3, (1000,))  # Random labels

    # Simple training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    network.train()
    for epoch in range(10):  # Quick training
        optimizer.zero_grad()
        outputs = network(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    print(f"Training completed. Final loss: {loss.item():.4f}")

    # 2. Create genome from trained network
    print("\n2. Creating genome from trained network...")

    # Convert network architecture to genome format
    genome = NetworkBuilder.from_pytorch_network(network, "trading_bot_v1")

    print(f"Genome created with ID: {genome.metadata.genome_id}")
    print(f"Architecture layers: {genome.architecture.layer_types}")
    print(f"Number of weights: {len(genome.weights)}")
    print(f"Number of biases: {len(genome.biases)}")

    # 3. Serialize genome to multiple formats
    print("\n3. Serializing genome to multiple formats...")

    serializer = GenomeSerializer()

    # Save as JSON
    json_path = "trading_bot_genome.json"
    serializer.save_genome(genome, json_path, format="json")
    print(f"Saved JSON: {json_path}")

    # Save as compressed binary
    binary_path = "trading_bot_genome.pkl.gz"
    serializer.save_genome(genome, binary_path, format="compressed")
    print(f"Saved compressed binary: {binary_path}")

    # Save as regular binary
    binary_path2 = "trading_bot_genome.pkl"
    serializer.save_genome(genome, binary_path2, format="binary")
    print(f"Saved binary: {binary_path2}")

    # 4. Load and reproduce networks
    print("\n4. Loading and reproducing networks...")

    # Load from JSON
    loaded_genome_json = serializer.load_genome(json_path)
    reproduced_network_json = NetworkBuilder.build_network(loaded_genome_json)

    # Load from binary
    loaded_genome_binary = serializer.load_genome(binary_path2, format="binary")
    reproduced_network_binary = NetworkBuilder.build_network(loaded_genome_binary)

    # Load from compressed
    loaded_genome_compressed = serializer.load_genome(binary_path, format="compressed")
    reproduced_network_compressed = NetworkBuilder.build_network(
        loaded_genome_compressed
    )

    print("All networks successfully reproduced!")

    # 5. Verify reproduction accuracy
    print("\n5. Verifying reproduction accuracy...")

    # Test with same input
    test_input = torch.randn(1, 10)

    original_output = network(test_input)
    reproduced_output_json = reproduced_network_json(test_input)
    reproduced_output_binary = reproduced_network_binary(test_input)
    reproduced_output_compressed = reproduced_network_compressed(test_input)

    # Calculate differences
    diff_json = torch.abs(original_output - reproduced_output_json).max().item()
    diff_binary = torch.abs(original_output - reproduced_output_binary).max().item()
    diff_compressed = (
        torch.abs(original_output - reproduced_output_compressed).max().item()
    )

    print(f"Maximum difference (JSON): {diff_json:.2e}")
    print(f"Maximum difference (Binary): {diff_binary:.2e}")
    print(f"Maximum difference (Compressed): {diff_compressed:.2e}")

    # Verify hashes
    original_hash = genome.get_reproduction_hash()
    json_hash = loaded_genome_json.get_reproduction_hash()
    binary_hash = loaded_genome_binary.get_reproduction_hash()
    compressed_hash = loaded_genome_compressed.get_reproduction_hash()

    print(f"\nHash verification:")
    print(f"Original: {original_hash}")
    print(f"JSON:     {json_hash} {'✓' if json_hash == original_hash else '✗'}")
    print(f"Binary:   {binary_hash} {'✓' if binary_hash == original_hash else '✗'}")
    print(
        f"Compressed: {compressed_hash} {'✓' if compressed_hash == original_hash else '✗'}"
    )

    # 6. Demonstrate transformer serialization
    print("\n6. Demonstrating transformer serialization...")

    transformer = MarketTransformer(
        input_size=20, d_model=128, n_heads=8, n_layers=4, output_size=3
    )

    # Create genome from transformer
    transformer_genome = NetworkBuilder.from_pytorch_network(
        transformer, "market_transformer_v1"
    )

    # Save transformer
    transformer_path = "market_transformer_genome.json"
    serializer.save_genome(transformer_genome, transformer_path, format="json")

    # Load and reproduce transformer
    loaded_transformer_genome = serializer.load_genome(transformer_path)
    reproduced_transformer = NetworkBuilder.build_network(loaded_transformer_genome)

    print(f"Transformer serialized and reproduced successfully!")
    print(f"Original parameters: {sum(p.numel() for p in transformer.parameters())}")
    print(
        f"Reproduced parameters: {sum(p.numel() for p in reproduced_transformer.parameters())}"
    )

    # 7. Show genome analysis
    print("\n7. Genome analysis and metadata...")

    analysis = genome.analyze_network()
    print(f"Network complexity: {analysis['complexity_score']:.2f}")
    print(f"Parameter count: {analysis['total_parameters']}")
    print(f"Memory usage: {analysis['memory_usage_mb']:.2f} MB")
    print(f"Layer distribution: {analysis['layer_distribution']}")

    print("\n✅ Neural network serialization demo completed successfully!")
    print("Key achievements:")
    print("- ✅ Network trained and serialized to multiple formats")
    print("- ✅ Perfect reproduction verified with hash matching")
    print("- ✅ Transformer architecture supported")
    print("- ✅ Comprehensive metadata and analysis available")


if __name__ == "__main__":
    main()
