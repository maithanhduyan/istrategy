"""
Tests for neural genome serialization and reproduction.
"""

import unittest
import tempfile
import os
import torch
import torch.nn as nn
import numpy as np
from quantrader import (
    NeuralGenome,
    NetworkArchitecture,
    GenomeSerializer,
    NetworkBuilder,
    FeedForwardNetwork,
    MarketTransformer,
)


class TestGenomeSerialization(unittest.TestCase):
    """Test neural genome serialization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test architecture
        self.architecture = NetworkArchitecture(
            name="test_network",
            type="feedforward",
            layers=[
                {"type": "input", "size": 5, "activation": "linear"},
                {"type": "hidden", "size": 10, "activation": "relu"},
                {"type": "output", "size": 3, "activation": "softmax"},
            ],
        )

        # Create test weights and biases
        self.weights = {
            "layer_0_1": np.random.randn(5, 10).tolist(),
            "layer_1_2": np.random.randn(10, 3).tolist(),
        }

        self.biases = {
            "layer_1": np.random.randn(10).tolist(),
            "layer_2": np.random.randn(3).tolist(),
        }

        # Create test genome
        self.genome = NeuralGenome(
            genome_id="test_001",
            architecture=self.architecture,
            weights=self.weights,
            biases=self.biases,
            metadata={"test": True, "version": "1.0"},
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_genome_creation(self):
        """Test genome creation."""
        self.assertEqual(self.genome.genome_id, "test_001")
        self.assertEqual(self.genome.architecture.name, "test_network")
        self.assertIn("layer_0_1", self.genome.weights)
        self.assertIn("layer_1", self.genome.biases)

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        serializer = GenomeSerializer()
        filepath = os.path.join(self.temp_dir, "test_genome.json")

        # Save genome
        serializer.save_genome(self.genome, filepath, format="json")
        self.assertTrue(os.path.exists(filepath))

        # Load genome
        loaded_genome = serializer.load_genome(filepath)

        # Verify integrity
        self.assertEqual(loaded_genome.genome_id, self.genome.genome_id)
        self.assertEqual(loaded_genome.architecture.name, self.genome.architecture.name)

        # Check weights
        for key in self.genome.weights:
            np.testing.assert_array_almost_equal(
                np.array(loaded_genome.weights[key]), np.array(self.genome.weights[key])
            )

    def test_pickle_serialization(self):
        """Test pickle serialization."""
        serializer = GenomeSerializer()
        filepath = os.path.join(self.temp_dir, "test_genome.pkl")

        # Save and load
        serializer.save_genome(self.genome, filepath, format="pickle")
        loaded_genome = serializer.load_genome(filepath)

        # Verify
        self.assertEqual(loaded_genome.genome_id, self.genome.genome_id)

    def test_compressed_serialization(self):
        """Test compressed serialization."""
        serializer = GenomeSerializer()
        filepath = os.path.join(self.temp_dir, "test_genome_compressed.json")

        # Save compressed
        serializer.save_genome(self.genome, filepath, format="json", compress=True)

        # Verify compressed file exists
        self.assertTrue(os.path.exists(f"{filepath}.gz"))

        # Load compressed
        loaded_genome = serializer.load_genome(filepath, compressed=True)
        self.assertEqual(loaded_genome.genome_id, self.genome.genome_id)

    def test_reproduction_hash(self):
        """Test reproduction hash consistency."""
        hash1 = self.genome.get_reproduction_hash()

        # Serialize and load
        serializer = GenomeSerializer()
        filepath = os.path.join(self.temp_dir, "test_hash.json")
        serializer.save_genome(self.genome, filepath, format="json")
        loaded_genome = serializer.load_genome(filepath)

        hash2 = loaded_genome.get_reproduction_hash()

        # Hashes should match
        self.assertEqual(hash1, hash2)

    def test_network_building(self):
        """Test building networks from genomes."""
        network = NetworkBuilder.build_network(self.genome)

        self.assertIsInstance(network, nn.Module)

        # Test forward pass
        test_input = torch.randn(1, 5)
        output = network(test_input)

        self.assertEqual(output.shape, (1, 3))

    def test_pytorch_network_conversion(self):
        """Test converting PyTorch networks to genomes."""
        # Create a PyTorch network
        network = FeedForwardNetwork(input_size=8, hidden_sizes=[16, 8], output_size=3)

        # Convert to genome
        genome = NetworkBuilder.from_pytorch_network(network, "pytorch_test")

        # Verify genome
        self.assertEqual(genome.genome_id, "pytorch_test")
        self.assertEqual(genome.architecture.type, "FeedForward")

        # Rebuild network
        rebuilt_network = NetworkBuilder.build_network(genome)

        # Test with same input
        test_input = torch.randn(1, 8)
        original_output = network(test_input)
        rebuilt_output = rebuilt_network(test_input)

        # Should be very close (within floating point precision)
        max_diff = torch.abs(original_output - rebuilt_output).max().item()
        self.assertLess(max_diff, 1e-5)


class TestNEATGenome(unittest.TestCase):
    """Test NEAT genome functionality."""

    def test_neat_genome_creation(self):
        """Test NEAT genome creation."""
        from quantrader import NEATConfig, NEATGenome

        config = NEATConfig(num_inputs=5, num_outputs=2, population_size=10)

        genome = NEATGenome(genome_id=1, config=config)

        # Check basic structure
        self.assertEqual(len(genome.nodes), 7)  # 5 inputs + 2 outputs
        self.assertGreater(
            len(genome.connections), 0
        )  # Should have initial connections

    def test_neat_mutation(self):
        """Test NEAT mutation operations."""
        from quantrader import NEATConfig, NEATGenome

        config = NEATConfig(
            num_inputs=3,
            num_outputs=1,
            node_add_prob=1.0,  # Force node addition
            conn_add_prob=1.0,  # Force connection addition
        )

        genome = NEATGenome(genome_id=1, config=config)
        initial_node_count = len(genome.nodes)
        initial_conn_count = len(genome.connections)

        # Mutate
        genome.mutate()

        # Should have more nodes or connections
        self.assertTrue(
            len(genome.nodes) > initial_node_count
            or len(genome.connections) > initial_conn_count
        )

    def test_neat_crossover(self):
        """Test NEAT crossover operation."""
        from quantrader import NEATConfig, NEATGenome

        config = NEATConfig(num_inputs=3, num_outputs=1)

        genome1 = NEATGenome(genome_id=1, config=config)
        genome2 = NEATGenome(genome_id=2, config=config)

        # Set different fitness
        genome1.fitness = 10.0
        genome2.fitness = 5.0

        # Crossover
        child = genome1.crossover(genome2)

        # Child should be valid
        self.assertIsInstance(child, NEATGenome)
        self.assertGreater(child.genome_id, 2)

    def test_neat_serialization(self):
        """Test NEAT genome serialization."""
        from quantrader import NEATConfig, NEATGenome

        config = NEATConfig(num_inputs=4, num_outputs=2)
        genome = NEATGenome(genome_id=42, config=config)

        # Serialize
        genome_dict = genome.to_dict()

        # Deserialize
        loaded_genome = NEATGenome.from_dict(genome_dict)

        # Verify
        self.assertEqual(loaded_genome.genome_id, genome.genome_id)
        self.assertEqual(len(loaded_genome.nodes), len(genome.nodes))
        self.assertEqual(len(loaded_genome.connections), len(genome.connections))


class TestTransformerModels(unittest.TestCase):
    """Test transformer model functionality."""

    def test_market_transformer_creation(self):
        """Test MarketTransformer creation."""
        transformer = MarketTransformer(
            input_size=10, d_model=64, n_heads=4, n_layers=2, output_size=3
        )

        self.assertIsInstance(transformer, nn.Module)

        # Test forward pass
        batch_size, seq_len = 2, 20
        test_input = torch.randn(batch_size, seq_len, 10)

        output = transformer(test_input)
        self.assertEqual(output.shape, (batch_size, 3))

    def test_transformer_serialization(self):
        """Test transformer serialization."""
        transformer = MarketTransformer(
            input_size=5, d_model=32, n_heads=2, n_layers=1, output_size=2
        )

        # Convert to genome
        genome = NetworkBuilder.from_pytorch_network(transformer, "transformer_test")

        # Verify genome
        self.assertEqual(genome.architecture.type, "MarketTransformer")

        # Test serialization
        serializer = GenomeSerializer()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            serializer.save_genome(genome, filepath, format="json")
            loaded_genome = serializer.load_genome(filepath)

            self.assertEqual(loaded_genome.genome_id, "transformer_test")

        finally:
            os.unlink(filepath)


class TestDataProviders(unittest.TestCase):
    """Test data provider functionality."""

    def test_technical_indicators(self):
        """Test technical indicator calculations."""
        from quantrader import TechnicalIndicators

        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

        # Test RSI
        rsi = TechnicalIndicators.rsi(prices, window=5)
        self.assertFalse(rsi.isna().all())

        # Test SMA
        sma = TechnicalIndicators.sma(prices, window=3)
        self.assertEqual(len(sma), len(prices))

        # Test MACD
        macd_data = TechnicalIndicators.macd(prices)
        self.assertIn("macd", macd_data)
        self.assertIn("signal", macd_data)

    def test_feature_engineering(self):
        """Test feature engineering."""
        from quantrader import FeatureEngineer

        # Create sample OHLCV data
        data = pd.DataFrame(
            {
                "datetime": pd.date_range("2023-01-01", periods=50, freq="D"),
                "open": np.random.uniform(100, 110, 50),
                "high": np.random.uniform(110, 120, 50),
                "low": np.random.uniform(90, 100, 50),
                "close": np.random.uniform(95, 115, 50),
                "volume": np.random.uniform(1000, 5000, 50),
            }
        )

        engineer = FeatureEngineer()
        features_df = engineer.create_features(data)

        # Should have more columns than original
        self.assertGreater(len(features_df.columns), len(data.columns))

        # Should have technical indicators
        self.assertTrue(any("rsi" in col for col in features_df.columns))
        self.assertTrue(any("macd" in col for col in features_df.columns))


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestGenomeSerialization))
    suite.addTest(unittest.makeSuite(TestNEATGenome))
    suite.addTest(unittest.makeSuite(TestTransformerModels))
    suite.addTest(unittest.makeSuite(TestDataProviders))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import pandas as pd

    success = run_tests()
    exit(0 if success else 1)
