"""
QuantRader - Advanced Neural Network Trading Framework

A modern, clean package focused on neural network storage and simulation for trading bots.
Provides complete serialization/deserialization capabilities for reproducing, analyzing,
or reusing networks for other models like transformers and LLMs.

Key Features:
- Complete neural network serialization/deserialization
- NEAT (NeuroEvolution of Augmenting Topologies) support
- Transformer and LLM architectures for trading
- Modern data providers (Yahoo Finance, CCXT)
- Technical indicator feature engineering
- Comprehensive backtesting and evaluation
- Clean, extensible architecture

Usage:
    from quantrader import NeuralGenome, NEATConfig, Population
    from quantrader import MarketTransformer, TransformerTradingStrategy
    from quantrader import YahooFinanceProvider, TechnicalIndicators
"""

__version__ = "0.1.0"
__author__ = "QuantRader Team"
__license__ = "MIT"

# Core imports
from .core import (
    # Base classes
    TradingStrategy,
    NeuralStrategy,
    EvolvableStrategy,
    DataProvider,
    Broker,
    Serializable,
    Reproducible,
    MetricsCalculator,
    # Data structures
    OrderType,
    OrderSide,
    TimeFrame,
    MarketData,
    Order,
    Position,
)

# Model imports
from .models import (
    # Genome classes
    NetworkArchitecture,
    GenomeMetadata,
    NeuralGenome,
    GenomeSerializer,
    NetworkBuilder,
    # Neural networks
    FeedForwardNetwork,
    LSTMNetwork,
    TransformerNetwork,
    EnsembleNetwork,
    NetworkFactory,
)

# Evolution imports
from .evolution import (
    # NEAT components
    NodeType,
    ActivationFunction,
    NEATConfig,
    NEATGenome,
    # Population management
    Species,
    Population,
)

# Transformer imports
from .transformers import (
    MarketTransformer,
    MarketLLM,
    TransformerTradingStrategy,
    TransformerFactory,
)

# Data imports
from .data import (
    YahooFinanceProvider,
    CCXTProvider,
    TechnicalIndicators,
    FeatureEngineer,
)

# Utility imports
from .utils import (
    SerializationUtils,
    ValidationUtils,
    HashUtils,
    PathUtils,
    ConfigUtils,
    ArrayUtils,
    timer,
)


# Version and dependency checks
def check_dependencies():
    """Check and report on optional dependencies."""
    optional_deps = {
        "torch": "PyTorch for neural networks",
        "transformers": "Hugging Face Transformers for LLMs",
        "yfinance": "Yahoo Finance data provider",
        "ccxt": "Cryptocurrency exchange data",
        "pandas": "Data manipulation",
        "numpy": "Numerical computing",
        "matplotlib": "Plotting and visualization",
        "seaborn": "Statistical visualization",
    }

    available = {}
    missing = {}

    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            available[dep] = description
        except ImportError:
            missing[dep] = description

    return available, missing


def get_system_info():
    """Get system and package information."""
    import sys
    import platform

    available_deps, missing_deps = check_dependencies()

    info = {
        "quantrader_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "available_dependencies": list(available_deps.keys()),
        "missing_dependencies": list(missing_deps.keys()),
    }

    return info


def print_system_info():
    """Print system and package information."""
    info = get_system_info()

    print(f"QuantRader v{info['quantrader_version']}")
    print(f"Python: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"Available dependencies: {', '.join(info['available_dependencies'])}")

    if info["missing_dependencies"]:
        print(
            f"Missing optional dependencies: {', '.join(info['missing_dependencies'])}"
        )
        print("Install missing dependencies for full functionality:")
        for dep in info["missing_dependencies"]:
            print(f"  pip install {dep}")


# Export all public classes and functions
__all__ = [
    # Core classes
    "TradingStrategy",
    "NeuralStrategy",
    "EvolvableStrategy",
    "DataProvider",
    "Broker",
    "Serializable",
    "Reproducible",
    "MetricsCalculator",
    # Data structures
    "OrderType",
    "OrderSide",
    "TimeFrame",
    "MarketData",
    "Order",
    "Position",
    # Model classes
    "NetworkArchitecture",
    "GenomeMetadata",
    "NeuralGenome",
    "GenomeSerializer",
    "NetworkBuilder",
    "FeedForwardNetwork",
    "LSTMNetwork",
    "TransformerNetwork",
    "EnsembleNetwork",
    "NetworkFactory",
    # Evolution classes
    "NodeType",
    "ActivationFunction",
    "NEATConfig",
    "NEATGenome",
    "Species",
    "Population",
    # Transformer classes
    "MarketTransformer",
    "MarketLLM",
    "TransformerTradingStrategy",
    "TransformerFactory",
    # Data classes
    "YahooFinanceProvider",
    "CCXTProvider",
    "TechnicalIndicators",
    "FeatureEngineer",
    # Utility classes
    "SerializationUtils",
    "ValidationUtils",
    "HashUtils",
    "PathUtils",
    "ConfigUtils",
    "ArrayUtils",
    "timer",
    # Functions
    "check_dependencies",
    "get_system_info",
    "print_system_info",
]

# Perform startup checks
if __name__ != "__main__":
    # Only run checks when imported, not when running as script
    try:
        available, missing = check_dependencies()
        if missing:
            import warnings

            warnings.warn(
                f"Some optional dependencies are missing: {list(missing.keys())}. "
                "Some features may not be available. Run quantrader.print_system_info() for details.",
                ImportWarning,
            )
    except Exception:
        pass  # Silently fail dependency checks to avoid import issues
