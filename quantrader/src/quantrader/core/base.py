"""
Base classes and interfaces for the quantrader package.
Provides foundational abstractions for trading systems, neural networks, and evolution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn

T = TypeVar("T")


class OrderType(Enum):
    """Types of trading orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class TimeFrame(Enum):
    """Trading timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class MarketData:
    """Market data structure."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame
    features: Optional[Dict[str, float]] = None  # Technical indicators


@dataclass
class Order:
    """Trading order structure."""

    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None


@dataclass
class Position:
    """Trading position structure."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    timestamp: datetime


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the strategy."""
        pass

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Order]:
        """Generate trading signal based on market data."""
        pass

    @abstractmethod
    def on_order_filled(self, order: Order, fill_price: float) -> None:
        """Handle order fill event."""
        pass


class NeuralStrategy(TradingStrategy):
    """Base class for neural network-based trading strategies."""

    def __init__(self, name: str, network: Optional[nn.Module] = None):
        super().__init__(name)
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_network(self, network: nn.Module) -> None:
        """Set the neural network."""
        self.network = network.to(self.device)

    @abstractmethod
    def preprocess_data(self, market_data: MarketData) -> torch.Tensor:
        """Preprocess market data for neural network input."""
        pass

    @abstractmethod
    def postprocess_output(self, output: torch.Tensor) -> Optional[Order]:
        """Convert neural network output to trading signal."""
        pass

    def generate_signal(self, market_data: MarketData) -> Optional[Order]:
        """Generate signal using neural network."""
        if not self.network:
            return None

        self.network.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_data(market_data)
            output = self.network(input_tensor)
            return self.postprocess_output(output)


class EvolvableStrategy(NeuralStrategy):
    """Strategy that can be evolved using genetic algorithms."""

    def __init__(self, name: str, genome_id: Optional[str] = None):
        super().__init__(name)
        self.genome_id = genome_id
        self.fitness_score = 0.0
        self.generation = 0

    @abstractmethod
    def mutate(self, mutation_rate: float) -> "EvolvableStrategy":
        """Create a mutated copy of this strategy."""
        pass

    @abstractmethod
    def crossover(self, other: "EvolvableStrategy") -> "EvolvableStrategy":
        """Create offspring by crossing over with another strategy."""
        pass

    def evaluate_fitness(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate fitness score based on performance metrics."""
        # Default fitness function - can be overridden
        return_pct = performance_metrics.get("total_return", 0.0)
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)

        # Composite fitness score
        fitness = return_pct * 0.4 + sharpe_ratio * 0.4 - abs(max_drawdown) * 0.2

        self.fitness_score = fitness
        return fitness


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get historical market data."""
        pass

    @abstractmethod
    def get_real_time_data(self, symbol: str) -> MarketData:
        """Get real-time market data."""
        pass


class Broker(ABC):
    """Abstract base class for broker interfaces."""

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place a trading order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Get account balance."""
        pass


class Serializable(ABC, Generic[T]):
    """Abstract base class for serializable objects."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> T:
        """Deserialize from dictionary."""
        pass

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json

        return json.dumps(self.to_dict(), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> T:
        """Deserialize from JSON string."""
        import json

        return cls.from_dict(json.loads(json_str))


class Reproducible(ABC):
    """Abstract base class for reproducible objects."""

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for reproduction."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for reproduction."""
        pass

    @abstractmethod
    def get_reproduction_hash(self) -> str:
        """Get hash for verifying reproduction accuracy."""
        pass


class MetricsCalculator:
    """Utility class for calculating trading performance metrics."""

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series."""
        return prices.pct_change().dropna()

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + MetricsCalculator.calculate_returns(prices)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_performance_metrics(prices: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        returns = MetricsCalculator.calculate_returns(prices)

        return {
            "total_return": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
            "annualized_return": returns.mean() * 252 * 100,
            "volatility": returns.std() * np.sqrt(252) * 100,
            "sharpe_ratio": MetricsCalculator.calculate_sharpe_ratio(returns),
            "max_drawdown": MetricsCalculator.calculate_max_drawdown(prices) * 100,
            "win_rate": (returns > 0).mean() * 100,
            "avg_return": returns.mean() * 100,
            "avg_win": returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
            "avg_loss": returns[returns < 0].mean() * 100 if (returns < 0).any() else 0,
        }
