#!/usr/bin/env python3
"""
Test script to verify the saved NEAT genome can be loaded and analyzed.
"""

from src.quantrader.models.genome import GenomeSerializer
import json


def test_genome_loading():
    """Test loading and analyzing the saved genome."""
    try:
        # Load the saved genome
        genome = GenomeSerializer.load_genome("best_trading_bot.json")
        print("✅ Successfully loaded genome from JSON")
        print(f"Genome ID: {genome.metadata.genome_id}")
        print(f"Fitness Score: {genome.metadata.fitness_score}")
        print(f"Species ID: {genome.metadata.species_id}")
        print(f"Total weights: {sum(len(w) for w in genome.weights.values())}")
        print(f"Total biases: {sum(len(b) for b in genome.biases.values())}")

        # Test serialization round-trip
        json_str = GenomeSerializer.to_json(genome)
        reloaded_genome = GenomeSerializer.from_json(json_str)
        print("✅ Round-trip serialization successful")

        # Verify reproduction hash
        original_hash = genome.get_reproduction_hash()
        reloaded_hash = reloaded_genome.get_reproduction_hash()
        print(f"Original hash: {original_hash}")
        print(f"Reloaded hash: {reloaded_hash}")
        print(f"Hash match: {original_hash == reloaded_hash}")

        print("\n🎯 Genome analysis:")
        analysis = genome.analyze_network()

        print(f'  Total parameters: {analysis["total_parameters"]}')
        print(f'  Complexity score: {analysis["complexity_score"]:.2f}')
        print(f'  Memory usage: {analysis["memory_usage_mb"]:.2f} MB')
        print(f'  Layer distribution: {analysis["layer_distribution"]}')

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_genome_loading()
    if success:
        print("\n🎉 All tests passed! NEAT genome serialization is working perfectly.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
