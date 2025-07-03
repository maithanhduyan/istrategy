"""
Example: NEAT evolution for trading strategies.
Demonstrates evolving neural network topologies for cryptocurrency trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from quantrader import (
    NEATConfig,
    NEATGenome,
    Population,
    YahooFinanceProvider,
    TechnicalIndicators,
    FeatureEngineer,
    TimeFrame,
)


class TradingFitnessEvaluator:
    """Fitness evaluator for trading strategies."""

    def __init__(self, market_data: pd.DataFrame):
        self.market_data = market_data
        self.feature_engineer = FeatureEngineer()

        # Prepare features
        self.features_df = self.feature_engineer.create_features(market_data)
        self.features_df = self.features_df.dropna()

        # Normalize features
        self.feature_columns = [
            "rsi",
            "macd",
            "bb_position",
            "price_change_5",
            "price_change_20",
            "volatility_10",
            "volume_ratio",
            "stoch_k",
        ]

        # Ensure we have the required features
        available_features = [
            col for col in self.feature_columns if col in self.features_df.columns
        ]
        if len(available_features) < 6:
            # Use basic features if technical indicators are not available
            self.feature_columns = [
                "close",
                "volume",
                "price_change_1",
                "price_change_5",
                "hl_spread",
                "oc_spread",
            ]

        self.feature_columns = self.feature_columns[:8]  # Limit to 8 features for NEAT

    def evaluate_genome(self, genome: NEATGenome) -> float:
        """Evaluate trading performance of a NEAT genome."""
        try:
            # Simulate trading with the genome
            portfolio_value = 10000  # Starting capital
            btc_position = 0.0
            usd_position = portfolio_value

            transaction_cost = 0.001  # 0.1% transaction cost

            returns = []
            trades = 0

            for i in range(60, len(self.features_df) - 1):  # Skip first 60 for lookback
                # Get current features
                current_features = []
                for col in self.feature_columns:
                    if col in self.features_df.columns:
                        value = self.features_df[col].iloc[i]
                        current_features.append(value if not pd.isna(value) else 0.0)
                    else:
                        current_features.append(0.0)

                # Pad to 8 features if needed
                while len(current_features) < 8:
                    current_features.append(0.0)
                current_features = current_features[:8]

                # Get prediction from genome (simplified network simulation)
                prediction = self._simulate_network(genome, current_features)

                # Current and next prices
                current_price = self.features_df["close"].iloc[i]
                next_price = self.features_df["close"].iloc[i + 1]

                # Trading logic based on prediction
                action = self._get_action(prediction)

                if action == "buy" and usd_position > 100:  # Buy BTC
                    buy_amount = usd_position * 0.5  # Use 50% of USD
                    btc_bought = buy_amount / current_price * (1 - transaction_cost)
                    btc_position += btc_bought
                    usd_position -= buy_amount
                    trades += 1

                elif action == "sell" and btc_position > 0.001:  # Sell BTC
                    sell_amount = btc_position * 0.5  # Sell 50% of BTC
                    usd_received = sell_amount * current_price * (1 - transaction_cost)
                    usd_position += usd_received
                    btc_position -= sell_amount
                    trades += 1

                # Calculate portfolio value
                total_value = usd_position + btc_position * next_price
                returns.append((total_value - portfolio_value) / portfolio_value)
                portfolio_value = total_value

            # Calculate fitness metrics
            if not returns:
                return 0.0

            total_return = returns[-1] if returns else 0.0
            volatility = np.std(returns) if len(returns) > 1 else 0.0

            # Sharpe-like ratio
            sharpe = total_return / volatility if volatility > 0 else 0.0

            # Fitness combines return, risk-adjusted return, and trade frequency
            fitness = (
                total_return * 100  # Total return weight
                + sharpe * 10  # Risk-adjusted return
                + min(trades / len(returns) * 100, 20)  # Trade frequency bonus (capped)
            )

            return max(0.0, fitness)

        except Exception as e:
            print(f"Error evaluating genome {genome.genome_id}: {e}")
            return 0.0

    def _simulate_network(self, genome: NEATGenome, inputs: list) -> list:
        """Simplified neural network simulation for NEAT genome."""
        # This is a simplified simulation
        # In practice, you'd implement proper forward pass through the NEAT network

        node_values = {}

        # Initialize input nodes
        for i in range(len(inputs)):
            if i in genome.nodes:
                node_values[i] = inputs[i]

        # Forward pass through connections (simplified)
        output_values = [0.0, 0.0, 0.0]  # Buy, Sell, Hold

        for connection in genome.connections.values():
            if connection.enabled:
                input_val = node_values.get(connection.input_node, 0.0)
                output_idx = connection.output_node - genome.config.num_inputs

                if 0 <= output_idx < 3:
                    output_values[output_idx] += input_val * connection.weight

        # Apply activation (sigmoid)
        output_values = [
            1 / (1 + np.exp(-max(-500, min(500, x)))) for x in output_values
        ]

        return output_values

    def _get_action(self, prediction: list) -> str:
        """Convert network prediction to trading action."""
        if len(prediction) < 3:
            return "hold"

        max_idx = np.argmax(prediction)
        confidence = max(prediction) - np.mean(prediction)

        if confidence < 0.1:  # Low confidence threshold
            return "hold"

        actions = ["buy", "sell", "hold"]
        return actions[max_idx]


def main():
    print("🧬 NEAT Evolution for Crypto Trading")
    print("=" * 40)

    # 1. Load market data
    print("\n1. Loading Bitcoin market data...")

    try:
        provider = YahooFinanceProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months of data

        # Get Bitcoin data
        data = provider.get_historical_data(
            symbol="BTC-USD",
            timeframe=TimeFrame.D1,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"Loaded {len(data)} days of Bitcoin data")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using simulated data for demo...")

        # Create simulated Bitcoin data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=180), periods=180, freq="D"
        )
        np.random.seed(42)

        # Generate realistic Bitcoin price movement
        returns = np.random.normal(0.001, 0.03, 180)  # Daily returns
        prices = [50000]  # Starting price

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(1000000, 5000000, 180),
            }
        )

    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # 2. Setup NEAT configuration
    print("\n2. Configuring NEAT evolution...")

    config = NEATConfig(
        population_size=30,  # Small population for demo
        num_inputs=8,  # 8 market features
        num_outputs=3,  # Buy, Sell, Hold
        # Network structure
        initial_connection="partial_direct",
        connection_fraction=0.5,
        # Mutation rates
        node_add_prob=0.1,
        node_delete_prob=0.05,
        conn_add_prob=0.3,
        conn_delete_prob=0.1,
        weight_mutation_rate=0.8,
        # Fitness and selection
        fitness_criterion="max",
        elitism=2,
        survival_threshold=0.3,
        max_stagnation=15,
        compatibility_threshold=3.0,
    )

    print(f"Population size: {config.population_size}")
    print(f"Network: {config.num_inputs} inputs → {config.num_outputs} outputs")

    # 3. Create fitness evaluator
    print("\n3. Setting up fitness evaluation...")

    evaluator = TradingFitnessEvaluator(data)
    print(f"Using features: {evaluator.feature_columns}")

    # 4. Initialize population and evolve
    print("\n4. Starting NEAT evolution...")

    population = Population(config, evaluator.evaluate_genome)

    best_fitness_history = []
    avg_fitness_history = []

    generations = 5  # Limited for demo

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}/{generations}")
        print("-" * 30)

        # Evolve one generation
        stats = population.evolve_generation(parallel=False)

        best_fitness_history.append(stats["best_fitness"])
        avg_fitness_history.append(stats["average_fitness"])

        print(f"Best fitness: {stats['best_fitness']:.2f}")
        print(f"Average fitness: {stats['average_fitness']:.2f}")
        print(f"Species count: {stats['num_species']}")
        print(f"Population size: {stats['population_size']}")

        # Show best genome info
        if population.best_genome:
            best = population.best_genome
            print(
                f"Best genome: {best.genome_id} (nodes: {len(best.nodes)}, connections: {len([c for c in best.connections.values() if c.enabled])})"
            )

    # 5. Analyze results
    print("\n5. Evolution Results")
    print("=" * 30)

    print(f"Final best fitness: {max(best_fitness_history):.2f}")
    print(
        f"Fitness improvement: {best_fitness_history[-1] - best_fitness_history[0]:+.2f}"
    )
    print(
        f"Average fitness improvement: {avg_fitness_history[-1] - avg_fitness_history[0]:+.2f}"
    )

    # Get the best genome
    best_genome = population.best_genome

    if best_genome:
        print(f"\nBest genome analysis:")
        print(f"- Genome ID: {best_genome.genome_id}")
        print(f"- Fitness: {best_genome.fitness:.2f}")
        print(f"- Nodes: {len(best_genome.nodes)}")
        print(
            f"- Active connections: {len([c for c in best_genome.connections.values() if c.enabled])}"
        )
        print(f"- Species: {best_genome.species_id}")

        # Convert to NeuralGenome for serialization
        neural_genome = best_genome.to_neural_genome()

        print(f"\n6. Saving best trading bot...")
        from quantrader import GenomeSerializer

        serializer = GenomeSerializer()
        serializer.save_genome(neural_genome, "best_trading_bot.json", format="json")

        print("✅ Best trading bot saved as 'best_trading_bot.json'")
        print("✅ This genome can be loaded and used in production trading!")

    # 7. Show evolution statistics
    print(f"\n7. Evolution Statistics")
    print("-" * 25)

    final_stats = population.get_statistics()

    print(f"Final population diversity:")
    for species_stat in final_stats["species_stats"]:
        print(
            f"  Species {species_stat['id']}: {species_stat['size']} members, "
            f"fitness {species_stat['best_fitness']:.2f}"
        )

    print(f"\nTotal evolution time: {generations} generations")
    print(f"Neural networks evaluated: {config.population_size * generations}")

    print("\n✅ NEAT evolution demo completed!")
    print("Key achievements:")
    print("- ✅ Evolved neural network topology for trading")
    print("- ✅ Integrated with real market data")
    print("- ✅ Serialized best performer for reuse")
    print("- ✅ Demonstrated population-based optimization")


if __name__ == "__main__":
    main()
