"""
Example: Mass Trading Bot Evolution.
Demonstrates training 100+ trading bots using genetic algorithms and evolution.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from quantrader import (
    NEATConfig,
    NEATGenome,
    Population,
    YahooFinanceProvider,
    FeatureEngineer,
    TimeFrame,
    GenomeSerializer,
    NetworkBuilder,
)


class MassTradingEvolution:
    """Mass evolution system for trading bots."""

    def __init__(self, population_size: int = 100, num_generations: int = 50):
        self.population_size = population_size
        self.num_generations = num_generations
        self.feature_engineer = FeatureEngineer()
        self.serializer = GenomeSerializer()

        # Evolution parameters
        self.mutation_rate = 0.8
        self.crossover_rate = 0.7
        self.elite_percentage = 0.1
        self.tournament_size = 5

        # Trading parameters
        self.initial_capital = 10000.0
        self.transaction_cost = 0.001  # 0.1%
        self.max_position_ratio = 0.8  # Max 80% of capital in one position

        # Feature configuration
        self.feature_columns = [
            "close",
            "volume",
            "rsi",
            "macd",
            "bb_position",
            "price_change_1",
            "price_change_5",
            "volatility_10",
        ]
        self.num_inputs = len(self.feature_columns)
        self.num_outputs = 3  # Buy, Hold, Sell

    def load_market_data(self) -> pd.DataFrame:
        """Load and prepare market data."""
        print("📊 Loading market data...")

        try:
            provider = YahooFinanceProvider()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 11)  # 11 years data

            data = provider.get_historical_data(
                symbol="BTC-USD",
                timeframe=TimeFrame.D1,
                start_date=start_date,
                end_date=end_date,
            )

            print(f"✅ Loaded {len(data)} days of real market data")

        except Exception as e:
            print(f"⚠️ Error loading real data: {e}")
            print("🔄 Generating simulated market data...")

            # Generate more sophisticated simulated data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 11)
            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            np.random.seed(42)

            # Generate realistic Bitcoin-like price movements
            returns = []
            for i in range(len(dates)):
                # Add market regime changes
                if i < len(dates) * 0.3:  # Bear market
                    daily_return = np.random.normal(-0.001, 0.06)
                elif i < len(dates) * 0.7:  # Bull market
                    daily_return = np.random.normal(0.003, 0.08)
                else:  # Consolidation
                    daily_return = np.random.normal(0.0005, 0.04)

                returns.append(daily_return)

            # Convert returns to prices
            prices = [45000]  # Starting price
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Generate volumes with correlation to price movement
            volumes = []
            for i in range(len(prices)):
                base_volume = 1000000
                volatility_multiplier = abs(returns[i]) * 10 + 1
                volume = (
                    base_volume * volatility_multiplier * np.random.uniform(0.5, 2.0)
                )
                volumes.append(volume)

            data = pd.DataFrame(
                {
                    "datetime": dates,
                    "open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                    "high": [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
                    "low": [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
                    "close": prices,
                    "volume": volumes,
                }
            )

            print(f"✅ Generated {len(data)} days of simulated market data")

        return data

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for training."""
        print("🔧 Engineering features...")

        features_df = self.feature_engineer.create_features(data)
        features_df = features_df.dropna()

        # Ensure we have required features
        available_features = [
            col for col in self.feature_columns if col in features_df.columns
        ]

        if len(available_features) < 6:
            # Fallback features
            self.feature_columns = [
                "close",
                "volume",
                "price_change_1",
                "price_change_5",
                "hl_spread",
                "oc_spread",
                "vwap",
                "volatility_5",
            ]
            available_features = [
                col for col in self.feature_columns if col in features_df.columns
            ]

        # Normalize features
        feature_data = features_df[available_features].copy()

        # Simple normalization (z-score)
        for col in feature_data.columns:
            mean_val = feature_data[col].mean()
            std_val = feature_data[col].std()
            if std_val > 0:
                feature_data[col] = (feature_data[col] - mean_val) / std_val

        print(f"✅ Prepared {len(feature_data.columns)} normalized features")
        print(f"📈 Feature data shape: {feature_data.shape}")

        return feature_data

    def create_initial_population(self) -> List[Dict[str, Any]]:
        """Create initial population of trading bots."""
        print(f"🧬 Creating initial population of {self.population_size} bots...")

        population = []

        for i in range(self.population_size):
            # Create random neural network genome
            genome = {
                "id": i,
                "generation": 0,
                "fitness": 0.0,
                "architecture": self._create_random_architecture(),
                "weights": {},
                "biases": {},
                "parents": [],
                "mutations": [],
                "birth_time": datetime.now().isoformat(),
            }

            # Initialize random weights and biases
            self._initialize_genome_parameters(genome)

            population.append(genome)

        print(f"✅ Created population with diverse architectures")
        return population

    def _create_random_architecture(self) -> Dict[str, Any]:
        """Create random neural network architecture."""
        # Random hidden layers (1-3 layers)
        num_hidden_layers = np.random.randint(1, 4)

        layer_sizes = [self.num_inputs]
        for _ in range(num_hidden_layers):
            # Hidden layer size between 8-64 neurons
            layer_size = np.random.choice([8, 16, 24, 32, 48, 64])
            layer_sizes.append(layer_size)
        layer_sizes.append(self.num_outputs)

        # Random activation functions
        activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
        layer_activations = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:  # Output layer
                layer_activations.append("softmax")
            else:
                layer_activations.append(np.random.choice(activations))

        # Random dropout rates
        dropout_rates = [np.random.uniform(0.0, 0.3) for _ in range(num_hidden_layers)]

        return {
            "layer_sizes": layer_sizes,
            "activations": layer_activations,
            "dropout_rates": dropout_rates,
            "use_batch_norm": np.random.choice([True, False]),
            "use_skip_connections": (
                np.random.choice([True, False]) if num_hidden_layers > 1 else False
            ),
        }

    def _initialize_genome_parameters(self, genome: Dict[str, Any]) -> None:
        """Initialize weights and biases for genome."""
        arch = genome["architecture"]
        layer_sizes = arch["layer_sizes"]

        # Xavier/Glorot initialization
        for i in range(len(layer_sizes) - 1):
            layer_name = f"layer_{i}"

            # Weight matrix
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))

            weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
            genome["weights"][layer_name] = weights.tolist()

            # Bias vector
            biases = np.zeros(fan_out)
            genome["biases"][layer_name] = biases.tolist()

    def evaluate_genome_fitness(
        self, genome: Dict[str, Any], features_df: pd.DataFrame
    ) -> float:
        """Evaluate trading performance of a genome."""
        try:
            # Simulate trading
            portfolio_value = self.initial_capital
            btc_position = 0.0
            usd_position = portfolio_value

            trade_history = []
            portfolio_values = []

            lookback_period = 30

            for i in range(lookback_period, len(features_df) - 1):
                # Get features
                current_features = []
                for col in self.feature_columns:
                    if col in features_df.columns:
                        value = features_df[col].iloc[i]
                        current_features.append(value if not pd.isna(value) else 0.0)
                    else:
                        current_features.append(0.0)

                # Ensure feature vector has correct size
                current_features = current_features[: self.num_inputs]
                while len(current_features) < self.num_inputs:
                    current_features.append(0.0)

                # Forward pass through network
                prediction = self._forward_pass(genome, current_features)

                # Get action (0=sell, 1=hold, 2=buy)
                action = np.argmax(prediction)
                confidence = np.max(prediction)

                # Current price
                current_price = features_df["close"].iloc[i]

                # Execute trades based on prediction
                if action == 2 and confidence > 0.6 and usd_position > 100:  # Buy
                    trade_amount = min(usd_position * 0.3, usd_position * confidence)
                    btc_bought = (
                        trade_amount / current_price * (1 - self.transaction_cost)
                    )
                    btc_position += btc_bought
                    usd_position -= trade_amount

                    trade_history.append(
                        {
                            "timestamp": i,
                            "action": "buy",
                            "price": current_price,
                            "amount": btc_bought,
                            "confidence": confidence,
                        }
                    )

                elif action == 0 and confidence > 0.6 and btc_position > 0.001:  # Sell
                    sell_amount = min(btc_position * 0.3, btc_position * confidence)
                    usd_received = (
                        sell_amount * current_price * (1 - self.transaction_cost)
                    )
                    usd_position += usd_received
                    btc_position -= sell_amount

                    trade_history.append(
                        {
                            "timestamp": i,
                            "action": "sell",
                            "price": current_price,
                            "amount": sell_amount,
                            "confidence": confidence,
                        }
                    )

                # Update portfolio value
                total_value = usd_position + btc_position * current_price
                portfolio_values.append(total_value)

            # Calculate fitness metrics
            if not portfolio_values:
                return 0.0

            final_value = portfolio_values[-1]
            total_return = (final_value - self.initial_capital) / self.initial_capital

            # Calculate Sharpe ratio
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            if len(returns) > 1:
                sharpe_ratio = (
                    np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                )
            else:
                sharpe_ratio = 0

            # Calculate maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Calculate win rate
            winning_trades = 0
            total_trades = len(trade_history)
            if total_trades > 0:
                # Simplified win rate calculation
                for trade in trade_history:
                    if trade["confidence"] > 0.7:
                        winning_trades += 1
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0

            # Combined fitness score
            fitness = (
                total_return * 100  # Total return weight
                + sharpe_ratio * 50  # Risk-adjusted return
                + win_rate * 30  # Win rate bonus
                + max(0, (1 - max_drawdown)) * 20  # Drawdown penalty
                - abs(total_trades - 50) * 0.1  # Optimal trade frequency
            )

            return max(0.0, fitness)

        except Exception as e:
            print(f"⚠️ Error evaluating genome {genome['id']}: {e}")
            return 0.0

    def _forward_pass(self, genome: Dict[str, Any], inputs: List[float]) -> List[float]:
        """Forward pass through neural network."""
        arch = genome["architecture"]
        layer_sizes = arch["layer_sizes"]
        activations = arch["activations"]

        current_output = np.array(inputs)

        # Forward pass through each layer
        for i in range(len(layer_sizes) - 1):
            layer_name = f"layer_{i}"
            weights = np.array(genome["weights"][layer_name])
            biases = np.array(genome["biases"][layer_name])

            # Linear transformation
            current_output = np.dot(current_output, weights) + biases

            # Apply activation
            activation = activations[i]
            current_output = self._apply_activation(current_output, activation)

        return current_output.tolist()

    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif activation == "softmax":
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            return x  # Linear

    def mutate_genome(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a genome to create variation."""
        mutated = genome.copy()
        mutated["id"] = f"{genome['id']}_mutated_{int(time.time())}"
        mutated["generation"] = genome["generation"] + 1
        mutated["parents"] = [genome["id"]]
        mutated["mutations"] = []

        # Weight mutation
        if np.random.random() < 0.8:
            self._mutate_weights(mutated)
            mutated["mutations"].append("weight_mutation")

        # Bias mutation
        if np.random.random() < 0.6:
            self._mutate_biases(mutated)
            mutated["mutations"].append("bias_mutation")

        # Architecture mutation (rare)
        if np.random.random() < 0.1:
            self._mutate_architecture(mutated)
            mutated["mutations"].append("architecture_mutation")

        return mutated

    def _mutate_weights(self, genome: Dict[str, Any]) -> None:
        """Mutate weights of the network."""
        for layer_name in genome["weights"]:
            weights = np.array(genome["weights"][layer_name])

            # Add Gaussian noise
            noise_scale = 0.1
            noise = np.random.normal(0, noise_scale, weights.shape)

            # Apply mutation to random subset of weights
            mutation_mask = np.random.random(weights.shape) < 0.2
            weights[mutation_mask] += noise[mutation_mask]

            # Clip weights to reasonable range
            weights = np.clip(weights, -5, 5)

            genome["weights"][layer_name] = weights.tolist()

    def _mutate_biases(self, genome: Dict[str, Any]) -> None:
        """Mutate biases of the network."""
        for layer_name in genome["biases"]:
            biases = np.array(genome["biases"][layer_name])

            # Add small random changes
            noise = np.random.normal(0, 0.05, biases.shape)
            mutation_mask = np.random.random(biases.shape) < 0.3
            biases[mutation_mask] += noise[mutation_mask]

            # Clip biases
            biases = np.clip(biases, -2, 2)

            genome["biases"][layer_name] = biases.tolist()

    def _mutate_architecture(self, genome: Dict[str, Any]) -> None:
        """Mutate architecture (rare but potentially beneficial)."""
        arch = genome["architecture"]

        # Randomly change activation function
        if np.random.random() < 0.5 and len(arch["activations"]) > 1:
            idx = np.random.randint(
                0, len(arch["activations"]) - 1
            )  # Don't change output activation
            activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
            arch["activations"][idx] = np.random.choice(activations)

        # Randomly change dropout rate
        if "dropout_rates" in arch and arch["dropout_rates"]:
            idx = np.random.randint(0, len(arch["dropout_rates"]))
            arch["dropout_rates"][idx] = np.random.uniform(0.0, 0.4)

    def crossover_genomes(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create offspring from two parent genomes."""
        child = parent1.copy()
        child["id"] = f"child_{int(time.time())}_{np.random.randint(1000)}"
        child["generation"] = max(parent1["generation"], parent2["generation"]) + 1
        child["parents"] = [parent1["id"], parent2["id"]]
        child["mutations"] = ["crossover"]

        # Uniform crossover for weights and biases
        for layer_name in child["weights"]:
            if layer_name in parent2["weights"]:
                w1 = np.array(parent1["weights"][layer_name])
                w2 = np.array(parent2["weights"][layer_name])

                if w1.shape == w2.shape:
                    # Uniform crossover
                    mask = np.random.random(w1.shape) < 0.5
                    child_weights = np.where(mask, w1, w2)
                    child["weights"][layer_name] = child_weights.tolist()

        for layer_name in child["biases"]:
            if layer_name in parent2["biases"]:
                b1 = np.array(parent1["biases"][layer_name])
                b2 = np.array(parent2["biases"][layer_name])

                if b1.shape == b2.shape:
                    mask = np.random.random(b1.shape) < 0.5
                    child_biases = np.where(mask, b1, b2)
                    child["biases"][layer_name] = child_biases.tolist()

        return child

    def tournament_selection(
        self, population: List[Dict[str, Any]], tournament_size: int = 5
    ) -> Dict[str, Any]:
        """Tournament selection for choosing parents."""
        tournament = np.random.choice(population, tournament_size, replace=False)
        winner = max(tournament, key=lambda x: x["fitness"])
        return winner

    def evolve_generation(
        self, population: List[Dict[str, Any]], features_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Evolve one generation of the population."""
        # Evaluate fitness for all genomes
        print("🧠 Evaluating population fitness...")

        # Use multiprocessing for fitness evaluation
        with ProcessPoolExecutor(
            max_workers=min(8, multiprocessing.cpu_count())
        ) as executor:
            futures = []
            for genome in population:
                future = executor.submit(
                    self.evaluate_genome_fitness, genome, features_df
                )
                futures.append((future, genome))

            for future, genome in futures:
                try:
                    fitness = future.result(timeout=30)  # 30 second timeout per genome
                    genome["fitness"] = fitness
                except Exception as e:
                    print(f"⚠️ Error evaluating genome {genome['id']}: {e}")
                    genome["fitness"] = 0.0

        # Sort by fitness
        population.sort(key=lambda x: x["fitness"], reverse=True)

        # Print generation stats
        fitnesses = [g["fitness"] for g in population]
        print(f"📊 Generation stats:")
        print(f"   Best fitness: {max(fitnesses):.2f}")
        print(f"   Average fitness: {np.mean(fitnesses):.2f}")
        print(f"   Worst fitness: {min(fitnesses):.2f}")

        # Create next generation
        next_generation = []

        # Elite selection
        elite_count = int(self.population_size * self.elite_percentage)
        elites = population[:elite_count]
        next_generation.extend(elites)
        print(f"🏆 Preserved {elite_count} elite genomes")

        # Generate offspring
        while len(next_generation) < self.population_size:
            if np.random.random() < self.crossover_rate and len(population) > 1:
                # Crossover
                parent1 = self.tournament_selection(population, self.tournament_size)
                parent2 = self.tournament_selection(population, self.tournament_size)
                child = self.crossover_genomes(parent1, parent2)

                # Apply mutation
                if np.random.random() < self.mutation_rate:
                    child = self.mutate_genome(child)

                next_generation.append(child)
            else:
                # Mutation only
                parent = self.tournament_selection(population, self.tournament_size)
                child = self.mutate_genome(parent)
                next_generation.append(child)

        return next_generation[: self.population_size]

    def save_generation(
        self, population: List[Dict[str, Any]], generation: int
    ) -> None:
        """Save current generation to disk."""
        output_dir = f"training_results/generation_{generation:03d}"
        os.makedirs(output_dir, exist_ok=True)

        # Save best genomes
        population.sort(key=lambda x: x["fitness"], reverse=True)

        # Save top 10 genomes
        for i, genome in enumerate(population[:10]):
            filename = (
                f"{output_dir}/top_{i+1:02d}_fitness_{genome['fitness']:.2f}.json"
            )
            # Convert genome to JSON-serializable format
            serializable_genome = self._make_json_serializable(genome)
            with open(filename, "w") as f:
                json.dump(serializable_genome, f, indent=2)

        # Save generation summary
        summary = {
            "generation": generation,
            "population_size": len(population),
            "best_fitness": population[0]["fitness"],
            "average_fitness": np.mean([g["fitness"] for g in population]),
            "worst_fitness": population[-1]["fitness"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(f"{output_dir}/generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Saved generation {generation} results to {output_dir}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert NumPy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def run_evolution(self) -> Dict[str, Any]:
        """Run the complete evolution process."""
        print("🚀 Starting mass trading bot evolution!")
        print(f"📈 Population size: {self.population_size}")
        print(f"🔄 Generations: {self.num_generations}")
        print("=" * 60)

        # Prepare data
        market_data = self.load_market_data()
        features_df = self.prepare_features(market_data)

        # Create initial population
        population = self.create_initial_population()

        # Evolution loop
        evolution_history = []

        for generation in range(self.num_generations):
            print(f"\n🧬 Generation {generation + 1}/{self.num_generations}")
            print("-" * 40)

            start_time = time.time()

            # Evolve population
            population = self.evolve_generation(population, features_df)

            # Record history
            best_fitness = max(g["fitness"] for g in population)
            avg_fitness = np.mean([g["fitness"] for g in population])

            generation_stats = {
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "time_taken": time.time() - start_time,
            }
            evolution_history.append(generation_stats)

            # Save results
            self.save_generation(population, generation + 1)

            print(
                f"⏱️ Generation completed in {generation_stats['time_taken']:.1f} seconds"
            )

            # Early stopping if fitness plateaus
            if generation > 10:
                recent_best = [h["best_fitness"] for h in evolution_history[-5:]]
                if max(recent_best) - min(recent_best) < 1.0:
                    print("🔄 Fitness plateau detected - continuing evolution...")

        # Final results
        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_bot = population[0]

        print("\n" + "=" * 60)
        print("🏆 EVOLUTION COMPLETED!")
        print(f"🥇 Best bot fitness: {best_bot['fitness']:.2f}")
        print(f"🧠 Best bot architecture: {best_bot['architecture']['layer_sizes']}")
        print(f"🔄 Best bot generation: {best_bot['generation']}")
        print(f"💾 All results saved to training_results/")

        return {
            "best_bot": best_bot,
            "final_population": population,
            "evolution_history": evolution_history,
            "total_generations": self.num_generations,
            "population_size": self.population_size,
        }


def main():
    """Main execution function."""
    print("🤖 Mass Trading Bot Evolution System")
    print("=" * 50)

    # Configuration
    population_size = 100
    num_generations = 100

    # Create evolution system
    evolution = MassTradingEvolution(
        population_size=population_size, num_generations=num_generations
    )

    # Run evolution
    try:
        results = evolution.run_evolution()

        # Additional analysis
        print("\n📊 Final Analysis")
        print("-" * 30)

        best_bot = results["best_bot"]
        print(f"Best Bot ID: {best_bot['id']}")
        print(f"Architecture: {best_bot['architecture']}")
        print(f"Mutations applied: {best_bot.get('mutations', [])}")

        # Save final best bot
        os.makedirs("final_results", exist_ok=True)
        serializable_best_bot = evolution._make_json_serializable(best_bot)
        with open("final_results/best_trading_bot.json", "w") as f:
            json.dump(serializable_best_bot, f, indent=2)

        # Save evolution history
        with open("final_results/evolution_history.json", "w") as f:
            json.dump(results["evolution_history"], f, indent=2)

        print("\n✅ Evolution completed successfully!")
        print(
            "💡 Check training_results/ and final_results/ directories for detailed results"
        )

    except KeyboardInterrupt:
        print("\n⚠️ Evolution interrupted by user")
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
