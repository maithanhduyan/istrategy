"""
NEAT (NeuroEvolution of Augmenting Topologies) implementation for trading strategies.
Provides complete evolution of neural network topologies and weights.
"""

import random
import copy
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from ..core.base import EvolvableStrategy, Serializable, Reproducible
from ..models.genome import NeuralGenome, GenomeMetadata, NetworkArchitecture


class NodeType(Enum):
    """Types of nodes in NEAT networks."""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class ActivationFunction(Enum):
    """Activation functions for NEAT nodes."""

    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    SOFTMAX = "softmax"


@dataclass
class NodeGene:
    """Node gene in NEAT genome."""

    node_id: int
    node_type: NodeType
    activation: ActivationFunction
    bias: float = 0.0
    response: float = 1.0  # Response multiplier

    def mutate(self, config: "NEATConfig") -> None:
        """Mutate node parameters."""
        if random.random() < config.bias_mutation_rate:
            self.bias += random.gauss(0, config.bias_mutation_power)
            self.bias = max(
                -config.bias_max_value, min(config.bias_max_value, self.bias)
            )

        if random.random() < config.response_mutation_rate:
            self.response += random.gauss(0, config.response_mutation_power)
            self.response = max(
                config.response_min_value, min(config.response_max_value, self.response)
            )


@dataclass
class ConnectionGene:
    """Connection gene in NEAT genome."""

    innovation_number: int
    input_node: int
    output_node: int
    weight: float
    enabled: bool = True

    def mutate(self, config: "NEATConfig") -> None:
        """Mutate connection weight."""
        if random.random() < config.weight_mutation_rate:
            if random.random() < config.weight_replace_rate:
                # Replace weight completely
                self.weight = random.gauss(0, config.weight_init_stdev)
            else:
                # Perturb weight
                self.weight += random.gauss(0, config.weight_mutation_power)

        # Clamp weight
        self.weight = max(
            -config.weight_max_value, min(config.weight_max_value, self.weight)
        )


@dataclass
class NEATConfig:
    """Configuration for NEAT evolution."""

    # Population settings
    population_size: int = 150
    fitness_criterion: str = "max"  # 'max' or 'mean'
    fitness_threshold: float = None
    no_fitness_termination: bool = False
    reset_on_extinction: bool = False

    # Genome settings
    num_inputs: int = 10
    num_outputs: int = 3
    num_hidden: int = 0
    initial_connection: str = (
        "full_direct"  # 'full_direct', 'full_nodirect', 'partial_direct', 'partial_nodirect'
    )
    connection_fraction: float = None

    # Node parameters
    bias_init_mean: float = 0.0
    bias_init_stdev: float = 1.0
    bias_init_type: str = "gaussian"
    bias_replace_rate: float = 0.1
    bias_mutation_rate: float = 0.7
    bias_mutation_power: float = 0.5
    bias_max_value: float = 30.0
    bias_min_value: float = -30.0

    response_init_mean: float = 1.0
    response_init_stdev: float = 0.0
    response_init_type: str = "gaussian"
    response_replace_rate: float = 0.0
    response_mutation_rate: float = 0.0
    response_mutation_power: float = 0.0
    response_max_value: float = 30.0
    response_min_value: float = -30.0

    # Connection parameters
    weight_init_mean: float = 0.0
    weight_init_stdev: float = 1.0
    weight_init_type: str = "gaussian"
    weight_replace_rate: float = 0.1
    weight_mutation_rate: float = 0.8
    weight_mutation_power: float = 0.5
    weight_max_value: float = 30.0
    weight_min_value: float = -30.0

    # Structural mutation
    node_add_prob: float = 0.2
    node_delete_prob: float = 0.2
    conn_add_prob: float = 0.5
    conn_delete_prob: float = 0.5

    # Activation functions
    activation_default: str = "sigmoid"
    activation_mutate_rate: float = 0.0
    activation_options: List[str] = field(default_factory=lambda: ["sigmoid"])

    # Reproduction
    elitism: int = 2
    survival_threshold: float = 0.2
    min_species_size: int = 2

    # Speciation
    compatibility_threshold: float = 3.0
    compatibility_disjoint_coefficient: float = 1.0
    compatibility_weight_coefficient: float = 0.5
    max_stagnation: int = 20
    species_fitness_func: str = "max"  # 'max', 'mean'


class NEATGenome(Serializable, Reproducible):
    """NEAT genome representing a neural network topology and weights."""

    def __init__(self, genome_id: int, config: NEATConfig):
        self.genome_id = genome_id
        self.config = config
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.species_id: Optional[int] = None

        self._node_indexer = 0
        self._create_initial_genome()

    def _create_initial_genome(self) -> None:
        """Create initial genome with input and output nodes."""
        # Create input nodes
        for i in range(self.config.num_inputs):
            self.nodes[i] = NodeGene(
                node_id=i,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.LINEAR,
            )

        # Create output nodes
        for i in range(self.config.num_outputs):
            node_id = self.config.num_inputs + i
            self.nodes[node_id] = NodeGene(
                node_id=node_id,
                node_type=NodeType.OUTPUT,
                activation=ActivationFunction(self.config.activation_default),
            )

        self._node_indexer = self.config.num_inputs + self.config.num_outputs

        # Create initial connections
        if self.config.initial_connection == "full_direct":
            self._create_full_direct_connections()
        elif self.config.initial_connection == "partial_direct":
            self._create_partial_direct_connections()

    def _create_full_direct_connections(self) -> None:
        """Create full direct connections from inputs to outputs."""
        innovation_number = 0
        for input_id in range(self.config.num_inputs):
            for output_id in range(
                self.config.num_inputs, self.config.num_inputs + self.config.num_outputs
            ):
                weight = random.gauss(
                    self.config.weight_init_mean, self.config.weight_init_stdev
                )
                self.connections[innovation_number] = ConnectionGene(
                    innovation_number=innovation_number,
                    input_node=input_id,
                    output_node=output_id,
                    weight=weight,
                )
                innovation_number += 1

    def _create_partial_direct_connections(self) -> None:
        """Create partial direct connections."""
        if self.config.connection_fraction is None:
            fraction = 0.5  # Default
        else:
            fraction = self.config.connection_fraction

        innovation_number = 0
        for input_id in range(self.config.num_inputs):
            for output_id in range(
                self.config.num_inputs, self.config.num_inputs + self.config.num_outputs
            ):
                if random.random() < fraction:
                    weight = random.gauss(
                        self.config.weight_init_mean, self.config.weight_init_stdev
                    )
                    self.connections[innovation_number] = ConnectionGene(
                        innovation_number=innovation_number,
                        input_node=input_id,
                        output_node=output_id,
                        weight=weight,
                    )
                innovation_number += 1

    def mutate(self) -> None:
        """Mutate the genome."""
        # Mutate existing nodes
        for node in self.nodes.values():
            if node.node_type != NodeType.INPUT:  # Don't mutate input nodes
                node.mutate(self.config)

        # Mutate existing connections
        for connection in self.connections.values():
            connection.mutate(self.config)

        # Structural mutations
        if random.random() < self.config.node_add_prob:
            self._mutate_add_node()

        if random.random() < self.config.conn_add_prob:
            self._mutate_add_connection()

        if random.random() < self.config.node_delete_prob:
            self._mutate_delete_node()

        if random.random() < self.config.conn_delete_prob:
            self._mutate_delete_connection()

    def _mutate_add_node(self) -> None:
        """Add a new node by splitting an existing connection."""
        if not self.connections:
            return

        # Choose random connection to split
        connection_key = random.choice(list(self.connections.keys()))
        connection = self.connections[connection_key]

        if not connection.enabled:
            return

        # Disable the connection
        connection.enabled = False

        # Create new node
        new_node_id = self._node_indexer
        self._node_indexer += 1

        self.nodes[new_node_id] = NodeGene(
            node_id=new_node_id,
            node_type=NodeType.HIDDEN,
            activation=ActivationFunction(self.config.activation_default),
        )

        # Create two new connections
        # Input to new node (weight = 1.0)
        new_innovation_1 = max(self.connections.keys()) + 1 if self.connections else 0
        self.connections[new_innovation_1] = ConnectionGene(
            innovation_number=new_innovation_1,
            input_node=connection.input_node,
            output_node=new_node_id,
            weight=1.0,
        )

        # New node to output (weight = original weight)
        new_innovation_2 = new_innovation_1 + 1
        self.connections[new_innovation_2] = ConnectionGene(
            innovation_number=new_innovation_2,
            input_node=new_node_id,
            output_node=connection.output_node,
            weight=connection.weight,
        )

    def _mutate_add_connection(self) -> None:
        """Add a new connection between existing nodes."""
        possible_inputs = [
            node_id
            for node_id, node in self.nodes.items()
            if node.node_type in [NodeType.INPUT, NodeType.HIDDEN]
        ]
        possible_outputs = [
            node_id
            for node_id, node in self.nodes.items()
            if node.node_type in [NodeType.HIDDEN, NodeType.OUTPUT]
        ]

        # Remove existing connections
        existing_connections = {
            (conn.input_node, conn.output_node) for conn in self.connections.values()
        }

        possible_new_connections = []
        for input_node in possible_inputs:
            for output_node in possible_outputs:
                if (input_node, output_node) not in existing_connections:
                    # Check for cycles (simplified)
                    if not self._creates_cycle(input_node, output_node):
                        possible_new_connections.append((input_node, output_node))

        if possible_new_connections:
            input_node, output_node = random.choice(possible_new_connections)
            new_innovation = max(self.connections.keys()) + 1 if self.connections else 0
            weight = random.gauss(
                self.config.weight_init_mean, self.config.weight_init_stdev
            )

            self.connections[new_innovation] = ConnectionGene(
                innovation_number=new_innovation,
                input_node=input_node,
                output_node=output_node,
                weight=weight,
            )

    def _creates_cycle(self, input_node: int, output_node: int) -> bool:
        """Check if adding a connection would create a cycle."""
        # Simplified cycle detection
        if input_node == output_node:
            return True

        # If output is an input node, it would create a cycle
        if self.nodes[output_node].node_type == NodeType.INPUT:
            return True

        return False

    def _mutate_delete_node(self) -> None:
        """Delete a random hidden node and its connections."""
        hidden_nodes = [
            node_id
            for node_id, node in self.nodes.items()
            if node.node_type == NodeType.HIDDEN
        ]

        if not hidden_nodes:
            return

        node_to_delete = random.choice(hidden_nodes)

        # Remove the node
        del self.nodes[node_to_delete]

        # Remove all connections involving this node
        connections_to_remove = []
        for key, connection in self.connections.items():
            if (
                connection.input_node == node_to_delete
                or connection.output_node == node_to_delete
            ):
                connections_to_remove.append(key)

        for key in connections_to_remove:
            del self.connections[key]

    def _mutate_delete_connection(self) -> None:
        """Delete a random connection."""
        if not self.connections:
            return

        connection_key = random.choice(list(self.connections.keys()))
        del self.connections[connection_key]

    def crossover(self, other: "NEATGenome") -> "NEATGenome":
        """Create offspring through crossover with another genome."""
        # Determine which parent is more fit
        if self.fitness > other.fitness:
            parent1, parent2 = self, other
        elif self.fitness < other.fitness:
            parent1, parent2 = other, self
        else:
            # Equal fitness, randomly choose
            parent1, parent2 = (self, other) if random.random() < 0.5 else (other, self)

        # Create child genome
        child = NEATGenome(
            genome_id=max(self.genome_id, other.genome_id) + 1, config=self.config
        )

        # Inherit nodes from both parents
        all_node_ids = set(parent1.nodes.keys()) | set(parent2.nodes.keys())
        child.nodes = {}
        child._node_indexer = max(all_node_ids) + 1 if all_node_ids else 0

        for node_id in all_node_ids:
            if node_id in parent1.nodes and node_id in parent2.nodes:
                # Both parents have this node - randomly choose
                source_node = random.choice(
                    [parent1.nodes[node_id], parent2.nodes[node_id]]
                )
                child.nodes[node_id] = copy.deepcopy(source_node)
            elif node_id in parent1.nodes:
                # Only parent1 has this node
                child.nodes[node_id] = copy.deepcopy(parent1.nodes[node_id])
            else:
                # Only parent2 has this node
                child.nodes[node_id] = copy.deepcopy(parent2.nodes[node_id])

        # Inherit connections
        child.connections = {}
        all_innovations = set(parent1.connections.keys()) | set(
            parent2.connections.keys()
        )

        for innovation in all_innovations:
            if innovation in parent1.connections and innovation in parent2.connections:
                # Matching gene - randomly choose
                source_conn = random.choice(
                    [parent1.connections[innovation], parent2.connections[innovation]]
                )
                child.connections[innovation] = copy.deepcopy(source_conn)
            elif innovation in parent1.connections:
                # Disjoint/excess gene from more fit parent
                child.connections[innovation] = copy.deepcopy(
                    parent1.connections[innovation]
                )

        return child

    def distance(self, other: "NEATGenome") -> float:
        """Calculate compatibility distance between genomes."""
        if not self.connections and not other.connections:
            return 0.0

        # Get all innovation numbers
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        if not innovations1 and not innovations2:
            return 0.0

        # Find matching and disjoint genes
        matching = innovations1 & innovations2
        disjoint = innovations1 ^ innovations2

        # Count disjoint and excess genes
        if innovations1 and innovations2:
            max_innovation1 = max(innovations1)
            max_innovation2 = max(innovations2)
            excess_threshold = min(max_innovation1, max_innovation2)

            excess = sum(1 for innov in disjoint if innov > excess_threshold)
            disjoint_count = len(disjoint) - excess
        else:
            excess = len(disjoint)
            disjoint_count = 0

        # Calculate average weight difference for matching genes
        if matching:
            weight_diff = sum(
                abs(self.connections[innov].weight - other.connections[innov].weight)
                for innov in matching
            ) / len(matching)
        else:
            weight_diff = 0.0

        # Normalization factor
        N = max(len(self.connections), len(other.connections), 1)

        # Calculate distance
        distance = (
            self.config.compatibility_disjoint_coefficient
            * (disjoint_count + excess)
            / N
            + self.config.compatibility_weight_coefficient * weight_diff
        )

        return distance

    def to_neural_genome(self) -> NeuralGenome:
        """Convert NEAT genome to NeuralGenome for serialization."""
        # Create network architecture compatible with NetworkArchitecture class
        # Input layer (KHÔNG thêm 'input' vào layer_types, chỉ dùng cho shape)
        input_shape = (self.config.num_inputs,)

        # Hidden layers (NEAT không có layer truyền thống, chỉ ước lượng)
        hidden_nodes = [
            node for node in self.nodes.values() if node.node_type == NodeType.HIDDEN
        ]
        layer_types = []
        layer_sizes = []
        activation_functions = []
        connections = []

        if hidden_nodes:
            layer_types.append("linear")
            layer_sizes.append((self.config.num_inputs, len(hidden_nodes)))
            activation_functions.append("relu")
            prev_size = len(hidden_nodes)
        else:
            prev_size = self.config.num_inputs
        # Output layer
        layer_types.append("linear")
        layer_sizes.append((prev_size, self.config.num_outputs))
        activation_functions.append("sigmoid")

        # Convert NEAT connections to connection indices
        for conn_key, connection in self.connections.items():
            if connection.enabled:
                connections.append((connection.input_node, connection.output_node))
        architecture = NetworkArchitecture(
            layer_types=layer_types,
            layer_sizes=layer_sizes,
            activation_functions=activation_functions,
            connections=connections,
        )

        # Convert to weights và biases (luôn wrap thành list)
        weights = {}
        biases = {}
        for conn_key, connection in self.connections.items():
            if connection.enabled:
                key = f"conn_{connection.input_node}_{connection.output_node}"
                weights[key] = [connection.weight]  # wrap float thành list
        for node_id, node in self.nodes.items():
            key = f"node_{node_id}"
            biases[key] = [getattr(node, "bias", 0.0)]  # wrap float thành list

        return NeuralGenome(
            architecture=architecture,
            weights=weights,
            biases=biases,
            metadata=GenomeMetadata(
                genome_id=str(self.genome_id),
                fitness_score=self.fitness,
                species_id=(
                    str(self.species_id) if hasattr(self, "species_id") else None
                ),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize NEAT genome to dictionary."""
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "adjusted_fitness": self.adjusted_fitness,
            "species_id": self.species_id,
            "nodes": {
                str(k): {
                    "node_id": v.node_id,
                    "node_type": v.node_type.value,
                    "activation": v.activation.value,
                    "bias": v.bias,
                    "response": v.response,
                }
                for k, v in self.nodes.items()
            },
            "connections": {
                str(k): {
                    "innovation_number": v.innovation_number,
                    "input_node": v.input_node,
                    "output_node": v.output_node,
                    "weight": v.weight,
                    "enabled": v.enabled,
                }
                for k, v in self.connections.items()
            },
            "config": self.config.__dict__,
            "_node_indexer": self._node_indexer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NEATGenome":
        """Deserialize NEAT genome from dictionary."""
        config = NEATConfig(**data["config"])
        genome = cls.__new__(cls)
        genome.genome_id = data["genome_id"]
        genome.config = config
        genome.fitness = data["fitness"]
        genome.adjusted_fitness = data["adjusted_fitness"]
        genome.species_id = data["species_id"]
        genome._node_indexer = data["_node_indexer"]

        # Reconstruct nodes
        genome.nodes = {}
        for k, v in data["nodes"].items():
            genome.nodes[int(k)] = NodeGene(
                node_id=v["node_id"],
                node_type=NodeType(v["node_type"]),
                activation=ActivationFunction(v["activation"]),
                bias=v["bias"],
                response=v["response"],
            )

        # Reconstruct connections
        genome.connections = {}
        for k, v in data["connections"].items():
            genome.connections[int(k)] = ConnectionGene(
                innovation_number=v["innovation_number"],
                input_node=v["input_node"],
                output_node=v["output_node"],
                weight=v["weight"],
                enabled=v["enabled"],
            )

        return genome

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for reproduction."""
        return self.to_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for reproduction."""
        loaded = self.from_dict(state_dict)
        self.__dict__.update(loaded.__dict__)

    def get_reproduction_hash(self) -> str:
        """Get hash for verifying reproduction accuracy."""
        import hashlib
        import json

        # Create a deterministic representation
        repr_dict = {
            "nodes": sorted(
                [
                    (
                        k,
                        v.node_id,
                        v.node_type.value,
                        v.activation.value,
                        round(v.bias, 6),
                        round(v.response, 6),
                    )
                    for k, v in self.nodes.items()
                ]
            ),
            "connections": sorted(
                [
                    (
                        k,
                        v.innovation_number,
                        v.input_node,
                        v.output_node,
                        round(v.weight, 6),
                        v.enabled,
                    )
                    for k, v in self.connections.items()
                ]
            ),
        }

        json_str = json.dumps(repr_dict, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
