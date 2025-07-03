# QuantTrader

🚀 **Advanced Quantitative Trading Framework with Neural Network Evolution**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

QuantTrader is a cutting-edge quantitative trading framework that combines:

- **Neural Network Evolution (NEAT)** for strategy discovery
- **Transformer Models** for market prediction  
- **Advanced Neural Network Serialization** for model persistence and reuse
- **Real-time Trading Execution** with multiple broker integrations
- **Comprehensive Backtesting** with performance analytics
- **LLM Integration** for market sentiment analysis

## ✨ Key Features

### 🧬 Neural Network Evolution
- NEAT (NeuroEvolution of Augmenting Topologies) algorithm
- Complete genome serialization and reconstruction
- Genetic algorithm optimization for trading strategies
- Species diversity maintenance

### 🤖 Transformer Integration
- Pre-trained transformer models for financial data
- Custom attention mechanisms for time-series
- Multi-modal input support (price, volume, news, sentiment)
- Fine-tuning capabilities for specific assets

### 💾 Advanced Model Persistence
- Complete neural network state serialization
- Cross-platform model compatibility
- Version control for model evolution
- Model compression and optimization

### 📊 Trading Infrastructure
- Multi-exchange connectivity (Binance, Coinbase, etc.)
- Real-time data streaming
- Order management system
- Risk management and position sizing

## 🚀 Quick Start

### Installation

```bash
pip install quantrader
```

### Basic Usage

```python
import quantrader as qt

# Create a trading bot with neural network
bot = qt.TradingBot(
    model_type="neat",
    initial_capital=10000,
    risk_tolerance=0.02
)

# Load historical data
data = qt.load_market_data(
    symbol="BTC/USD",
    timeframe="1h",
    period="1y"
)

# Train the neural network
evolution = qt.NEATEvolution(
    population_size=50,
    generations=100
)

best_genome = evolution.evolve(bot, data)

# Save the trained model
qt.save_model(best_genome, "btc_trader_v1.qnn")

# Load and use the model
loaded_model = qt.load_model("btc_trader_v1.qnn")
prediction = loaded_model.predict(current_market_state)
```

### Advanced: Transformer Model

```python
# Create a transformer-based trading model
transformer = qt.TransformerTrader(
    model_name="quantrader/crypto-transformer-base",
    context_length=512,
    prediction_horizon=24
)

# Fine-tune on your data
transformer.fine_tune(
    data=data,
    epochs=10,
    learning_rate=1e-5
)

# Generate trading signals
signals = transformer.generate_signals(
    market_data=current_data,
    confidence_threshold=0.8
)
```

## 🏗️ Architecture

```
quantrader/
├── src/quantrader/
│   ├── core/              # Core trading infrastructure
│   │   ├── base.py        # Base classes and interfaces
│   │   ├── broker.py      # Broker integrations
│   │   ├── portfolio.py   # Portfolio management
│   │   └── risk.py        # Risk management
│   ├── models/            # Neural network models
│   │   ├── neat/          # NEAT evolution algorithms
│   │   ├── genome.py      # Neural network serialization
│   │   ├── networks.py    # Network architectures
│   │   └── persistence.py # Model saving/loading
│   ├── transformers/      # Transformer models
│   │   ├── attention.py   # Custom attention mechanisms
│   │   ├── tokenizer.py   # Financial data tokenization
│   │   └── trainer.py     # Training infrastructure
│   ├── data/              # Data management
│   │   ├── providers.py   # Data source integrations
│   │   ├── processing.py  # Data preprocessing
│   │   └── streaming.py   # Real-time data
│   ├── evolution/         # Genetic algorithms
│   │   ├── neat.py        # NEAT implementation
│   │   ├── operators.py   # Genetic operators
│   │   └── selection.py   # Selection strategies
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── examples/              # Example scripts
└── docs/                  # Documentation
```

## 🔧 Neural Network Serialization

QuantTrader's core innovation is complete neural network serialization:

```python
# Save complete network state
genome = {
    'architecture': network.get_architecture(),
    'weights': network.get_weights(),
    'metadata': {
        'generation': 42,
        'fitness': 0.95,
        'parents': [12, 23],
        'mutations': ['add_node', 'modify_weight']
    },
    'performance': {
        'backtest_results': {...},
        'live_trading_stats': {...}
    }
}

qt.save_genome(genome, "elite_trader_gen42.qnn")

# Reconstruct exact network
loaded_genome = qt.load_genome("elite_trader_gen42.qnn")
reconstructed_net = qt.build_network_from_genome(loaded_genome)
```

## 📈 Performance

- **Backtesting**: 10+ years of historical data processing
- **Real-time**: Sub-millisecond prediction latency
- **Scalability**: Handle 1000+ trading pairs simultaneously
- **Memory**: Efficient model compression (10-100x size reduction)

## 🧪 Examples

Check out `/examples` directory:

- `basic_neat_evolution.py` - Simple NEAT trading bot
- `transformer_sentiment.py` - News sentiment analysis
- `multi_asset_portfolio.py` - Portfolio optimization
- `model_serialization.py` - Save/load neural networks
- `live_trading_demo.py` - Real-time trading example

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: https://quantrader.readthedocs.io
- **PyPI**: https://pypi.org/project/quantrader/
- **GitHub**: https://github.com/quantrader/quantrader
- **Discord**: https://discord.gg/quantrader

---

*Built with ❤️ for quantitative traders and ML researchers*
