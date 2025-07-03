"""
Core models for neural network representation and serialization.

This module provides the foundation for saving, loading, and reconstructing
neural networks with complete state preservation.
"""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, validator


class NetworkArchitecture(BaseModel):
    """Represents the architecture of a neural network."""

    layer_types: List[str] = Field(
        ..., description="Types of layers (linear, conv, lstm, etc.)"
    )
    layer_sizes: List[Tuple[int, ...]] = Field(
        ..., description="Dimensions of each layer"
    )
    activation_functions: List[str] = Field(
        ..., description="Activation function for each layer"
    )
    connections: List[Tuple[int, int]] = Field(
        default_factory=list, description="Custom connections (from_layer, to_layer)"
    )

    @validator("layer_types")
    def validate_layer_types(cls, v):
        valid_types = {
            "linear",
            "conv1d",
            "conv2d",
            "lstm",
            "gru",
            "attention",
            "embedding",
        }
        for layer_type in v:
            if layer_type not in valid_types:
                raise ValueError(f"Invalid layer type: {layer_type}")
        return v


class GenomeMetadata(BaseModel):
    """Metadata for a neural network genome."""

    genome_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = Field(default=0)
    parent_ids: List[str] = Field(default_factory=list)
    creation_method: str = Field(default="random")
    creation_time: datetime = Field(default_factory=datetime.now)
    species_id: Optional[str] = None
    fitness_score: Optional[float] = None

    # Evolution tracking
    mutation_history: List[Dict[str, Any]] = Field(default_factory=list)
    crossover_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance metrics
    training_metrics: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    trading_metrics: Dict[str, Any] = Field(default_factory=dict)


class NeuralGenome(BaseModel):
    """
    Complete representation of a neural network that can be saved and reconstructed.

    This is the core class for neural network serialization in QuantTrader.
    It stores everything needed to reconstruct the exact same network.
    """

    # Core network definition
    architecture: NetworkArchitecture
    weights: Dict[str, List[float]] = Field(
        ..., description="All network weights as flat lists"
    )
    biases: Dict[str, List[float]] = Field(
        default_factory=dict, description="All network biases"
    )
    weight_shapes: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Original tensor shapes for weight reconstruction",
    )

    # Metadata
    metadata: GenomeMetadata = Field(default_factory=GenomeMetadata)

    # Training state
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None

    # Custom attributes for specific architectures
    attention_weights: Optional[Dict[str, List[float]]] = None
    embedding_matrices: Optional[Dict[str, List[List[float]]]] = None

    # Performance tracking
    performance_history: List[Dict[str, float]] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
        }

    def add_performance_record(self, metrics: Dict[str, float]):
        """Add a performance record to the history."""
        record = {"timestamp": datetime.now().isoformat(), **metrics}
        self.performance_history.append(record)

    def get_reproduction_hash(self) -> str:
        """Get a hash for verifying reproduction accuracy."""
        import hashlib

        # Create a deterministic representation
        data = {
            "architecture": self.architecture.dict(),
            "weights": self.weights,
            "biases": self.biases,
        }

        # Convert to JSON and hash
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:16]

    def analyze_network(self) -> Dict[str, Any]:
        """Analyze the network and return comprehensive statistics."""
        complexity = self.calculate_complexity()

        # Calculate weight statistics
        all_weights = []
        for weight_list in self.weights.values():
            all_weights.extend(weight_list)

        weight_stats = {
            "mean": sum(all_weights) / len(all_weights) if all_weights else 0,
            "std": np.std(all_weights) if all_weights else 0,
            "min": min(all_weights) if all_weights else 0,
            "max": max(all_weights) if all_weights else 0,
        }

        # Calculate bias statistics
        all_biases = []
        for bias_list in self.biases.values():
            all_biases.extend(bias_list)

        bias_stats = {
            "mean": sum(all_biases) / len(all_biases) if all_biases else 0,
            "std": np.std(all_biases) if all_biases else 0,
            "min": min(all_biases) if all_biases else 0,
            "max": max(all_biases) if all_biases else 0,
        }

        # Calculate complexity score
        total_params = complexity["total_parameters"]
        num_layers = complexity["num_layers"]
        complexity_score = total_params / (num_layers + 1)  # Simple complexity metric

        # Calculate memory usage (rough estimate)
        memory_usage_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32

        # Layer distribution
        layer_distribution = {}
        for layer_type in self.architecture.layer_types:
            layer_distribution[layer_type] = layer_distribution.get(layer_type, 0) + 1

        return {
            "complexity": complexity,
            "complexity_score": complexity_score,
            "total_parameters": total_params,
            "memory_usage_mb": memory_usage_mb,
            "layer_distribution": layer_distribution,
            "weight_statistics": weight_stats,
            "bias_statistics": bias_stats,
            "architecture_summary": {
                "layer_types": self.architecture.layer_types,
                "layer_count": len(self.architecture.layer_types),
                "total_layers": len(self.architecture.layer_sizes),
                "activation_functions": self.architecture.activation_functions,
            },
            "metadata": {
                "genome_id": self.metadata.genome_id,
                "generation": self.metadata.generation,
                "creation_time": self.metadata.creation_time,
                "fitness_score": self.metadata.fitness_score,
            },
        }

    def get_weights_as_arrays(self) -> Dict[str, np.ndarray]:
        """Convert weight lists back to numpy arrays."""
        return {name: np.array(weights) for name, weights in self.weights.items()}

    def get_biases_as_arrays(self) -> Dict[str, np.ndarray]:
        """Convert bias lists back to numpy arrays."""
        return {name: np.array(biases) for name, biases in self.biases.items()}

    def calculate_complexity(self) -> Dict[str, int]:
        """Calculate network complexity metrics."""
        total_params = sum(len(w) for w in self.weights.values())
        total_biases = sum(len(b) for b in self.biases.values())

        return {
            "total_parameters": total_params + total_biases,
            "total_weights": total_params,
            "total_biases": total_biases,
            "num_layers": len(self.architecture.layer_types),
            "num_connections": len(self.architecture.connections),
        }


class GenomeSerializer:
    """
    Advanced serialization for neural network genomes.

    Supports multiple formats and compression techniques.
    """

    @staticmethod
    def to_json(genome: NeuralGenome, compress: bool = False) -> str:
        """Serialize genome to JSON string."""
        data = genome.dict()

        # Handle datetime serialization
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_str = json.dumps(
            data, indent=2 if not compress else None, default=default_serializer
        )

        if compress:
            import gzip

            return gzip.compress(json_str.encode()).hex()

        return json_str

    @staticmethod
    def from_json(json_str: str, compressed: bool = False) -> NeuralGenome:
        """Deserialize genome from JSON string."""
        if compressed:
            import gzip

            json_str = gzip.decompress(bytes.fromhex(json_str)).decode()

        data = json.loads(json_str)

        # Handle datetime parsing
        if "metadata" in data and "creation_time" in data["metadata"]:
            if isinstance(data["metadata"]["creation_time"], str):
                data["metadata"]["creation_time"] = datetime.fromisoformat(
                    data["metadata"]["creation_time"]
                )

        return NeuralGenome(**data)

    @staticmethod
    def to_binary(genome: NeuralGenome) -> bytes:
        """Serialize genome to binary format using pickle."""
        return pickle.dumps(genome.dict())

    @staticmethod
    def from_binary(binary_data: bytes) -> NeuralGenome:
        """Deserialize genome from binary format."""
        data = pickle.loads(binary_data)
        return NeuralGenome(**data)

    @staticmethod
    def to_torch_state(genome: NeuralGenome) -> Dict[str, torch.Tensor]:
        """Convert genome to PyTorch state dict format."""
        state_dict = {}

        # Convert weights
        for name, weights in genome.weights.items():
            tensor = torch.tensor(weights, dtype=torch.float32)

            # Use stored weight shapes if available
            if hasattr(genome, "weight_shapes") and name in genome.weight_shapes:
                shape = genome.weight_shapes[name]
                tensor = tensor.reshape(shape)
            elif "weight" in name and len(weights) > 1:
                # Fallback: Try to infer the correct shape based on the parameter name and number of elements
                total_elements = len(weights)

                # Common patterns for linear layer weights
                possible_shapes = []
                for layer_size in genome.architecture.layer_sizes:
                    if len(layer_size) == 2:
                        out_features, in_features = layer_size[1], layer_size[0]
                        if out_features * in_features == total_elements:
                            possible_shapes.append((out_features, in_features))

                if possible_shapes:
                    # Use the first matching shape
                    tensor = tensor.reshape(possible_shapes[0])

            state_dict[name] = tensor

        # Convert biases
        for name, biases in genome.biases.items():
            state_dict[name] = torch.tensor(biases, dtype=torch.float32)

        return state_dict

    @staticmethod
    def from_torch_state(
        state_dict: Dict[str, torch.Tensor],
        architecture: NetworkArchitecture,
        metadata: Optional[GenomeMetadata] = None,
    ) -> NeuralGenome:
        """Create genome from PyTorch state dict."""
        weights = {}
        biases = {}

        for name, tensor in state_dict.items():
            flat_values = tensor.flatten().tolist()
            if "bias" in name.lower():
                biases[name] = flat_values
            else:
                weights[name] = flat_values

        return NeuralGenome(
            architecture=architecture,
            weights=weights,
            biases=biases,
            metadata=metadata or GenomeMetadata(),
        )

    @staticmethod
    def save_genome(
        genome: NeuralGenome, file_path: Union[str, Path], format: str = "json"
    ):
        """Save genome to file in specified format."""
        path = Path(file_path)

        if format.lower() == "json":
            with open(path, "w") as f:
                f.write(GenomeSerializer.to_json(genome))
        elif format.lower() == "binary":
            with open(path, "wb") as f:
                f.write(GenomeSerializer.to_binary(genome))
        elif format.lower() == "compressed":
            with open(path, "w") as f:
                f.write(GenomeSerializer.to_json(genome, compress=True))
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'json', 'binary', or 'compressed'"
            )

    @staticmethod
    def load_genome(file_path: Union[str, Path], format: str = "json") -> NeuralGenome:
        """Load genome from file."""
        path = Path(file_path)

        if format.lower() == "json":
            with open(path, "r") as f:
                return GenomeSerializer.from_json(f.read())
        elif format.lower() == "binary":
            with open(path, "rb") as f:
                return GenomeSerializer.from_binary(f.read())
        elif format.lower() == "compressed":
            with open(path, "r") as f:
                return GenomeSerializer.from_json(f.read(), compressed=True)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'json', 'binary', or 'compressed'"
            )


class NetworkBuilder:
    """
    Builds PyTorch networks from genome specifications.

    This class can reconstruct the exact neural network from a genome.
    """

    @staticmethod
    def build_feedforward(genome: NeuralGenome) -> nn.Module:
        """Build a feedforward network from genome."""
        layers = []

        # Build layers based on the architecture and weight shapes
        linear_layer_idx = 0
        for i, layer_type in enumerate(genome.architecture.layer_types):
            if layer_type == "linear":
                layer_size = genome.architecture.layer_sizes[linear_layer_idx]
                in_features, out_features = layer_size[0], layer_size[1]
                layers.append(nn.Linear(in_features, out_features))
                linear_layer_idx += 1

                # Add activation based on the next layer if it's an activation
                if linear_layer_idx < len(genome.architecture.activation_functions):
                    activation = genome.architecture.activation_functions[
                        linear_layer_idx - 1
                    ]
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "softmax":
                        layers.append(nn.Softmax(dim=-1))

        network = nn.Sequential(*layers)

        # Load weights
        state_dict = GenomeSerializer.to_torch_state(genome)
        network.load_state_dict(state_dict, strict=False)

        return network

    @staticmethod
    def build_recurrent(genome: NeuralGenome) -> nn.Module:
        """Build a recurrent network from genome."""

        # Implementation for LSTM/GRU networks
        class RecurrentNetwork(nn.Module):
            def __init__(self, genome: NeuralGenome):
                super().__init__()
                self.genome = genome

                # Build layers based on architecture
                # This is a simplified implementation
                input_size = genome.architecture.layer_sizes[0][-1]
                hidden_size = genome.architecture.layer_sizes[1][-1]
                output_size = genome.architecture.layer_sizes[-1][-1]

                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.output = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.output(lstm_out[:, -1, :])  # Last timestep

        network = RecurrentNetwork(genome)

        # Load weights
        state_dict = GenomeSerializer.to_torch_state(genome)
        network.load_state_dict(state_dict, strict=False)

        return network

    @staticmethod
    def build_transformer(genome: NeuralGenome) -> nn.Module:
        """Build a transformer network from genome."""

        # Implementation for transformer-based networks
        class TransformerNetwork(nn.Module):
            def __init__(self, genome: NeuralGenome):
                super().__init__()
                self.genome = genome

                # Extract transformer parameters from architecture
                d_model = genome.architecture.layer_sizes[0][-1]
                nhead = 8  # Could be stored in architecture
                num_layers = len(
                    [t for t in genome.architecture.layer_types if t == "attention"]
                )

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output = nn.Linear(
                    d_model, genome.architecture.layer_sizes[-1][-1]
                )

            def forward(self, x):
                transformer_out = self.transformer(x)
                return self.output(
                    transformer_out.mean(dim=1)
                )  # Global average pooling

        network = TransformerNetwork(genome)

        # Load weights if available
        state_dict = GenomeSerializer.to_torch_state(genome)
        network.load_state_dict(state_dict, strict=False)

        return network

    @staticmethod
    def build_network(genome: NeuralGenome) -> nn.Module:
        """
        Build a neural network from genome specification.

        Automatically determines the network type based on architecture.
        """
        # Check if it's a transformer-based network
        if any(
            layer_type == "attention" for layer_type in genome.architecture.layer_types
        ):
            return NetworkBuilder.build_transformer(genome)
        elif any(
            layer_type == "lstm" for layer_type in genome.architecture.layer_types
        ):
            return NetworkBuilder.build_recurrent(genome)
        else:
            # Default to feedforward
            return NetworkBuilder.build_feedforward(genome)

    @staticmethod
    def build_recurrent(genome: NeuralGenome) -> nn.Module:
        """Build a recurrent (LSTM/GRU) network from genome."""

        class RecurrentNetwork(nn.Module):
            def __init__(self, genome: NeuralGenome):
                super().__init__()
                self.genome = genome

                # Find LSTM layers
                lstm_layers = []
                linear_layers = []

                for i, layer_type in enumerate(genome.architecture.layer_types):
                    if layer_type == "lstm":
                        size = genome.architecture.layer_sizes[i]
                        lstm_layers.append(nn.LSTM(size[0], size[1], batch_first=True))
                    elif layer_type == "linear":
                        size = genome.architecture.layer_sizes[i]
                        linear_layers.append(nn.Linear(size[0], size[1]))

                self.lstm_layers = nn.ModuleList(lstm_layers)
                self.linear_layers = nn.ModuleList(linear_layers)

            def forward(self, x):
                # Pass through LSTM layers
                for lstm in self.lstm_layers:
                    x, _ = lstm(x)

                # Use last timestep output
                x = x[:, -1, :]

                # Pass through linear layers
                for linear in self.linear_layers:
                    x = linear(x)

                return x

        network = RecurrentNetwork(genome)

        # Load weights if available
        state_dict = GenomeSerializer.to_torch_state(genome)
        network.load_state_dict(state_dict, strict=False)

        return network

    @staticmethod
    def from_pytorch_network(
        network: nn.Module,
        network_name: str = "unnamed_network",
        metadata: Optional[GenomeMetadata] = None,
    ) -> "NeuralGenome":
        """
        Create a genome from an existing PyTorch network.

        This is a convenience method that wraps create_genome_from_network.
        """
        if metadata is None:
            metadata = GenomeMetadata(
                name=network_name,
                description=f"Genome created from PyTorch network: {type(network).__name__}",
            )
        return create_genome_from_network(network, metadata)


def create_genome_from_network(
    network: nn.Module, metadata: Optional[GenomeMetadata] = None
) -> NeuralGenome:
    """
    Create a genome from an existing PyTorch network.

    This function extracts the complete state of a network for serialization.
    """
    # Extract architecture
    layer_types = []
    layer_sizes = []
    activation_functions = []

    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            layer_types.append("linear")
            layer_sizes.append((module.in_features, module.out_features))
        elif isinstance(module, nn.LSTM):
            layer_types.append("lstm")
            layer_sizes.append((module.input_size, module.hidden_size))
        elif isinstance(module, nn.TransformerEncoderLayer):
            layer_types.append("attention")
            layer_sizes.append((module.self_attn.embed_dim,))
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax)):
            activation_functions.append(module.__class__.__name__.lower())

    architecture = NetworkArchitecture(
        layer_types=layer_types,
        layer_sizes=layer_sizes,
        activation_functions=activation_functions,
    )

    # Extract weights and biases
    weights = {}
    biases = {}
    weight_shapes = {}  # Store original shapes for reconstruction

    state_dict = network.state_dict()
    for name, tensor in state_dict.items():
        flat_values = tensor.flatten().tolist()
        if "bias" in name:
            biases[name] = flat_values
        else:
            weights[name] = flat_values
            weight_shapes[name] = list(tensor.shape)  # Store shape for reconstruction

    return NeuralGenome(
        architecture=architecture,
        weights=weights,
        biases=biases,
        weight_shapes=weight_shapes,
        metadata=metadata or GenomeMetadata(),
    )
