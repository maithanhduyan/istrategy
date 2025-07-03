"""
Neural network architectures for trading strategies.
Includes feedforward, LSTM, transformer, and ensemble models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math


class FeedForwardNetwork(nn.Module):
    """Simple feedforward neural network for trading signals."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "FeedForward",
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
        }


class LSTMNetwork(nn.Module):
    """LSTM network for time series trading data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, hidden = self.lstm(x, hidden)

        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Apply dropout and final linear layer
        output = self.fc(self.dropout(last_output))
        return output

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class TransformerNetwork(nn.Module):
    """Transformer network for trading strategies."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
        dropout_rate: float = 0.1,
        max_sequence_length: int = 1000,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)

        # Project input to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Apply transformer encoder
        output = self.transformer_encoder(x, mask)

        # Use the last output for prediction
        last_output = output[:, -1, :]  # (batch_size, d_model)

        # Apply dropout and output projection
        return self.output_projection(self.dropout(last_output))

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "Transformer",
            "input_size": self.input_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
        }


class EnsembleNetwork(nn.Module):
    """Ensemble of multiple neural networks."""

    def __init__(self, models: List[nn.Module], aggregation_method: str = "mean"):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.aggregation_method = aggregation_method

        # Verify all models have the same output size
        output_sizes = [self._get_output_size(model) for model in models]
        if len(set(output_sizes)) > 1:
            raise ValueError("All models must have the same output size")

        self.output_size = output_sizes[0]

        # Learnable weights for weighted average
        if aggregation_method == "weighted":
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def _get_output_size(self, model: nn.Module) -> int:
        """Get output size of a model."""
        if hasattr(model, "output_size"):
            return model.output_size
        elif hasattr(model, "fc") and hasattr(model.fc, "out_features"):
            return model.fc.out_features
        elif hasattr(model, "output_projection") and hasattr(
            model.output_projection, "out_features"
        ):
            return model.output_projection.out_features
        else:
            # Try to infer from the last linear layer
            for layer in reversed(list(model.modules())):
                if isinstance(layer, nn.Linear):
                    return layer.out_features
            raise ValueError(f"Cannot determine output size for model {type(model)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for model in self.models:
            output = model(x)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, output_size)

        if self.aggregation_method == "mean":
            return torch.mean(outputs, dim=0)
        elif self.aggregation_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            weighted_outputs = outputs * weights.view(-1, 1, 1)
            return torch.sum(weighted_outputs, dim=0)
        elif self.aggregation_method == "max":
            return torch.max(outputs, dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def get_architecture_dict(self) -> Dict[str, Any]:
        """Get architecture parameters for serialization."""
        return {
            "type": "Ensemble",
            "models": [
                (
                    model.get_architecture_dict()
                    if hasattr(model, "get_architecture_dict")
                    else {"type": type(model).__name__}
                )
                for model in self.models
            ],
            "aggregation_method": self.aggregation_method,
            "output_size": self.output_size,
        }


class NetworkFactory:
    """Factory for creating neural networks from architecture specifications."""

    @staticmethod
    def create_network(architecture: Dict[str, Any]) -> nn.Module:
        """Create a neural network from architecture specification."""
        network_type = architecture["type"]

        if network_type == "FeedForward":
            return FeedForwardNetwork(
                input_size=architecture["input_size"],
                hidden_sizes=architecture["hidden_sizes"],
                output_size=architecture["output_size"],
                dropout_rate=architecture.get("dropout_rate", 0.1),
            )
        elif network_type == "LSTM":
            return LSTMNetwork(
                input_size=architecture["input_size"],
                hidden_size=architecture["hidden_size"],
                num_layers=architecture["num_layers"],
                output_size=architecture["output_size"],
                dropout_rate=architecture.get("dropout_rate", 0.1),
            )
        elif network_type == "Transformer":
            return TransformerNetwork(
                input_size=architecture["input_size"],
                d_model=architecture["d_model"],
                nhead=architecture["nhead"],
                num_layers=architecture["num_layers"],
                output_size=architecture["output_size"],
                dropout_rate=architecture.get("dropout_rate", 0.1),
                max_sequence_length=architecture.get("max_sequence_length", 1000),
            )
        elif network_type == "Ensemble":
            models = [
                NetworkFactory.create_network(model_arch)
                for model_arch in architecture["models"]
            ]
            return EnsembleNetwork(
                models=models,
                aggregation_method=architecture.get("aggregation_method", "mean"),
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    @staticmethod
    def get_network_info(network: nn.Module) -> Dict[str, Any]:
        """Get comprehensive information about a network."""
        info = {
            "type": type(network).__name__,
            "total_parameters": sum(p.numel() for p in network.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in network.parameters() if p.requires_grad
            ),
        }

        if hasattr(network, "get_architecture_dict"):
            info["architecture"] = network.get_architecture_dict()

        return info
