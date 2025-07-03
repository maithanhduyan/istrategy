"""
Transformer-based models for trading strategies.
Includes market-specific transformer architectures and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ..models.networks import PositionalEncoding
from ..core.base import NeuralStrategy, MarketData, Order


class MarketAttention(nn.Module):
    """Custom attention mechanism for market data."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = query.size()[:2]

        # Linear transformations
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Attention computation
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Final linear transformation
        output = self.w_o(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with market attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MarketAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class MarketTransformer(nn.Module):
    """Transformer model specifically designed for market data."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 1000,
        output_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.output_size = output_size
        self.dropout_rate = dropout

        # Input embedding and positional encoding
        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Use the last token for prediction
        x = x[:, -1, :]  # (batch_size, d_model)

        # Output projection
        x = self.output_norm(x)
        output = self.output_projection(self.dropout(x))

        return output

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "MarketTransformer",
            "input_size": self.input_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_length": self.max_seq_length,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
        }


class CrossAttentionBlock(nn.Module):
    """Cross-attention between different market features."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.price_attention = MarketAttention(d_model, n_heads, dropout)
        self.volume_attention = MarketAttention(d_model, n_heads, dropout)
        self.cross_attention = MarketAttention(d_model, n_heads, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_model * 4, dropout)

    def forward(
        self, price_features: torch.Tensor, volume_features: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention on price features
        price_attn, _ = self.price_attention(
            price_features, price_features, price_features
        )
        price_features = self.norm1(price_features + price_attn)

        # Self-attention on volume features
        volume_attn, _ = self.volume_attention(
            volume_features, volumeFeatures, volumeFeatures
        )
        volume_features = self.norm2(volume_features + volume_attn)

        # Cross-attention between price and volume
        cross_attn, _ = self.cross_attention(
            price_features, volume_features, volume_features
        )
        combined = self.norm3(price_features + cross_attn)

        # Feed-forward
        output = self.feed_forward(combined)

        return combined + output


class MarketLLM(nn.Module):
    """Large Language Model architecture adapted for market prediction."""

    def __init__(
        self,
        vocab_size: int = 10000,  # Market event vocabulary
        d_model: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        num_market_features: int = 50,
        output_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.num_market_features = num_market_features
        self.output_size = output_size
        self.dropout_rate = dropout

        # Market text embeddings (for news, events, etc.)
        self.text_embedding = nn.Embedding(vocab_size, d_model)

        # Market numerical feature projection
        self.feature_projection = nn.Linear(num_market_features, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Multi-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            d_model, n_heads, dropout, batch_first=True
        )

        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        embeddings = []

        # Process text input
        if text_input is not None:
            text_emb = self.text_embedding(text_input) * math.sqrt(self.d_model)
            text_emb = self.positional_encoding(text_emb.transpose(0, 1)).transpose(
                0, 1
            )
            embeddings.append(text_emb)

        # Process numerical features
        if numerical_features is not None:
            num_emb = self.feature_projection(numerical_features) * math.sqrt(
                self.d_model
            )
            num_emb = self.positional_encoding(num_emb.transpose(0, 1)).transpose(0, 1)
            embeddings.append(num_emb)

        if not embeddings:
            raise ValueError("At least one input type must be provided")

        # Combine embeddings
        if len(embeddings) == 1:
            x = embeddings[0]
        else:
            # Multi-modal fusion using cross-attention
            x = embeddings[0]
            for emb in embeddings[1:]:
                fused, _ = self.fusion_layer(x, emb, emb)
                x = x + fused

        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Global average pooling
        if mask is not None:
            # Masked average
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            x = x_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)  # (batch_size, d_model)

        # Output projection
        x = self.output_norm(x)
        output = self.output_projection(self.dropout(x))

        return output

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "MarketLLM",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_length": self.max_seq_length,
            "num_market_features": self.num_market_features,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
        }


class TransformerTradingStrategy(NeuralStrategy):
    """Trading strategy using transformer models."""

    def __init__(self, name: str, model_type: str = "transformer"):
        super().__init__(name)
        self.model_type = model_type
        self.feature_scaler = None
        self.lookback_window = 60

    def initialize(self, **kwargs) -> None:
        """Initialize the transformer strategy."""
        config = kwargs.get("config", {})

        self.lookback_window = config.get("lookback_window", 60)
        input_size = config.get("input_size", 20)

        if self.model_type == "transformer":
            self.network = MarketTransformer(
                input_size=input_size,
                d_model=config.get("d_model", 256),
                n_heads=config.get("n_heads", 8),
                n_layers=config.get("n_layers", 6),
                max_seq_length=config.get("max_seq_length", 1000),
                output_size=3,  # Buy, Sell, Hold
            )
        elif self.model_type == "llm":
            self.network = MarketLLM(
                vocab_size=config.get("vocab_size", 10000),
                d_model=config.get("d_model", 512),
                n_heads=config.get("n_heads", 16),
                n_layers=config.get("n_layers", 12),
                num_market_features=input_size,
                output_size=3,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.network = self.network.to(self.device)
        self.is_initialized = True

    def preprocess_data(self, market_data: MarketData) -> torch.Tensor:
        """Preprocess market data for transformer input."""
        # Use only the features in the correct order and count
        if hasattr(self, "available_features") and self.available_features:
            features = [
                float(market_data.features.get(feat, 0.0))
                for feat in self.available_features
            ]
        elif hasattr(market_data, "features") and market_data.features:
            features = list(market_data.features.values())
        else:
            features = [
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
                market_data.volume,
            ]
        # Pad or truncate to expected input size
        input_size = self.network.input_embedding.in_features
        while len(features) < input_size:
            features.append(0.0)
        features = features[:input_size]
        sequence = torch.tensor([features] * self.lookback_window, dtype=torch.float32)
        sequence = sequence.unsqueeze(0)  # Add batch dimension
        return sequence.to(self.device)

    def postprocess_output(self, output: torch.Tensor) -> Optional[Order]:
        """Convert transformer output to trading signal."""
        probabilities = F.softmax(output, dim=-1)
        action = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities.max().item()

        # Only trade if confidence is high enough
        if confidence < 0.6:
            return None

        from ..core.base import OrderSide, OrderType

        if action == 0:  # Buy
            return Order(
                symbol="BTC/USD",  # Would be dynamic in practice
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=0.1,  # Would be calculated based on risk management
            )
        elif action == 1:  # Sell
            return Order(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=0.1,
            )
        else:  # Hold
            return None

    def on_order_filled(self, order: Order, fill_price: float) -> None:
        """Handle order fill event."""
        print(f"Order filled: {order.side.value} {order.quantity} at {fill_price}")


class TransformerFactory:
    """Factory for creating transformer-based models."""

    @staticmethod
    def create_model(architecture: Dict[str, Any]) -> nn.Module:
        """Create a transformer model from architecture specification."""
        model_type = architecture["type"]

        if model_type == "MarketTransformer":
            return MarketTransformer(
                input_size=architecture["input_size"],
                d_model=architecture.get("d_model", 256),
                n_heads=architecture.get("n_heads", 8),
                n_layers=architecture.get("n_layers", 6),
                d_ff=architecture.get("d_ff", 1024),
                max_seq_length=architecture.get("max_seq_length", 1000),
                output_size=architecture.get("output_size", 3),
                dropout=architecture.get("dropout_rate", 0.1),
            )
        elif model_type == "MarketLLM":
            return MarketLLM(
                vocab_size=architecture.get("vocab_size", 10000),
                d_model=architecture.get("d_model", 512),
                n_heads=architecture.get("n_heads", 16),
                n_layers=architecture.get("n_layers", 12),
                d_ff=architecture.get("d_ff", 2048),
                max_seq_length=architecture.get("max_seq_length", 2048),
                num_market_features=architecture.get("num_market_features", 50),
                output_size=architecture.get("output_size", 3),
                dropout=architecture.get("dropout_rate", 0.1),
            )
        else:
            raise ValueError(f"Unknown transformer type: {model_type}")

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive information about a transformer model."""
        info = {
            "type": type(model).__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        if hasattr(model, "get_architecture_dict"):
            info["architecture"] = model.get_architecture_dict()

        return info
