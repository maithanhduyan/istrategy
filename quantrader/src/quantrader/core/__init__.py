"""
Core module initialization.
Exports main abstractions and base classes.
"""

from .base import (
    # Enums
    OrderType,
    OrderSide,
    TimeFrame,
    # Data structures
    MarketData,
    Order,
    Position,
    # Abstract base classes
    TradingStrategy,
    NeuralStrategy,
    EvolvableStrategy,
    DataProvider,
    Broker,
    Serializable,
    Reproducible,
    # Utilities
    MetricsCalculator,
)

__all__ = [
    "OrderType",
    "OrderSide",
    "TimeFrame",
    "MarketData",
    "Order",
    "Position",
    "TradingStrategy",
    "NeuralStrategy",
    "EvolvableStrategy",
    "DataProvider",
    "Broker",
    "Serializable",
    "Reproducible",
    "MetricsCalculator",
]
