"""
Population management and evolution algorithms.
Handles species formation, selection, reproduction, and evaluation.
"""

import random
import statistics
import copy
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .neat import NEATGenome, NEATConfig


@dataclass
class Species:
    """Species for speciation in NEAT."""

    species_id: int
    representative: NEATGenome
    members: List[NEATGenome]
    fitness_history: List[float]
    stagnation_count: int = 0

    def update_representative(self) -> None:
        """Update species representative."""
        if self.members:
            # Choose the most fit member as representative
            self.representative = max(self.members, key=lambda g: g.fitness)

    def calculate_adjusted_fitness(self) -> None:
        """Calculate adjusted fitness for all members."""
        species_size = len(self.members)
        for member in self.members:
            member.adjusted_fitness = member.fitness / species_size

    def get_average_fitness(self) -> float:
        """Get average fitness of species."""
        if not self.members:
            return 0.0
        return statistics.mean(member.fitness for member in self.members)

    def is_stagnant(self, max_stagnation: int) -> bool:
        """Check if species is stagnant."""
        return self.stagnation_count >= max_stagnation


class Population:
    """NEAT population manager."""

    def __init__(
        self, config: NEATConfig, fitness_function: Callable[[NEATGenome], float]
    ):
        self.config = config
        self.fitness_function = fitness_function
        self.generation = 0
        self.population: List[NEATGenome] = []
        self.species: List[Species] = []
        self.best_genome: Optional[NEATGenome] = None
        self.innovation_number = 0

        # Statistics
        self.fitness_history: List[float] = []
        self.species_count_history: List[int] = []

        # Initialize population
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Initialize the population with random genomes."""
        self.population = []
        for i in range(self.config.population_size):
            genome = NEATGenome(genome_id=i, config=self.config)
            self.population.append(genome)

    def evolve_generation(self, parallel: bool = False) -> Dict[str, Any]:
        """Evolve one generation."""
        # Evaluate fitness
        self._evaluate_fitness(parallel)

        # Update best genome
        current_best = max(self.population, key=lambda g: g.fitness)
        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = current_best

        # Record statistics
        avg_fitness = statistics.mean(g.fitness for g in self.population)
        self.fitness_history.append(avg_fitness)

        # Speciation
        self._speciate()
        self.species_count_history.append(len(self.species))

        # Calculate adjusted fitness
        for species in self.species:
            species.calculate_adjusted_fitness()

        # Selection and reproduction
        new_population = self._reproduce()

        # Replace population
        self.population = new_population
        self.generation += 1

        # Update stagnation
        self._update_stagnation()

        # Remove extinct species
        self._remove_extinct_species()

        return {
            "generation": self.generation,
            "best_fitness": self.best_genome.fitness if self.best_genome else 0,
            "average_fitness": avg_fitness,
            "num_species": len(self.species),
            "population_size": len(self.population),
        }

    def _evaluate_fitness(self, parallel: bool = False) -> None:
        """Evaluate fitness for all genomes."""
        if parallel and len(self.population) > 10:
            # Parallel evaluation
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.fitness_function, genome): genome
                    for genome in self.population
                }

                for future in as_completed(futures):
                    genome = futures[future]
                    try:
                        genome.fitness = future.result()
                    except Exception as e:
                        print(f"Error evaluating genome {genome.genome_id}: {e}")
                        genome.fitness = 0.0
        else:
            # Sequential evaluation
            for genome in self.population:
                try:
                    genome.fitness = self.fitness_function(genome)
                except Exception as e:
                    print(f"Error evaluating genome {genome.genome_id}: {e}")
                    genome.fitness = 0.0

    def _speciate(self) -> None:
        """Organize population into species."""
        # Reset species membership
        for species in self.species:
            species.members = []

        # Assign genomes to species
        unassigned = list(self.population)

        for genome in self.population[:]:
            assigned = False

            # Try to assign to existing species
            for species in self.species:
                distance = genome.distance(species.representative)
                if distance < self.config.compatibility_threshold:
                    species.members.append(genome)
                    genome.species_id = species.species_id
                    assigned = True
                    break

            # Create new species if not assigned
            if not assigned:
                new_species_id = len(self.species)
                new_species = Species(
                    species_id=new_species_id,
                    representative=genome,
                    members=[genome],
                    fitness_history=[],
                )
                genome.species_id = new_species_id
                self.species.append(new_species)

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives
        for species in self.species:
            species.update_representative()

    def _reproduce(self) -> List[NEATGenome]:
        """Create new population through reproduction."""
        new_population = []

        # Calculate total adjusted fitness
        total_adjusted_fitness = sum(
            sum(member.adjusted_fitness for member in species.members)
            for species in self.species
        )

        if total_adjusted_fitness == 0:
            # If no fitness, create random population
            return [
                NEATGenome(i, self.config) for i in range(self.config.population_size)
            ]

        # Calculate offspring allocation for each species
        offspring_counts = {}
        remaining_offspring = self.config.population_size

        for species in self.species:
            species_fitness = sum(member.adjusted_fitness for member in species.members)
            proportion = species_fitness / total_adjusted_fitness
            offspring_count = max(1, int(proportion * self.config.population_size))
            offspring_counts[species.species_id] = min(
                offspring_count, remaining_offspring
            )
            remaining_offspring -= offspring_counts[species.species_id]

        # Elitism - preserve best genomes
        if self.config.elitism > 0:
            elite = sorted(self.population, key=lambda g: g.fitness, reverse=True)[
                : self.config.elitism
            ]
            new_population.extend(elite)

        # Reproduce each species
        for species in self.species:
            offspring_count = offspring_counts.get(species.species_id, 0)
            if offspring_count == 0:
                continue

            # Adjust for elitism
            if self.config.elitism > 0 and any(
                g in species.members for g in new_population
            ):
                elite_in_species = sum(
                    1 for g in new_population if g in species.members
                )
                offspring_count = max(0, offspring_count - elite_in_species)

            species_offspring = self._reproduce_species(species, offspring_count)
            new_population.extend(species_offspring)

        # Fill remaining slots with random offspring
        while len(new_population) < self.config.population_size:
            if self.species:
                random_species = random.choice(self.species)
                offspring = self._reproduce_species(random_species, 1)
                new_population.extend(offspring)
            else:
                # Create random genome
                new_population.append(NEATGenome(len(new_population), self.config))

        # Assign new genome IDs
        for i, genome in enumerate(new_population):
            genome.genome_id = i

        return new_population[: self.config.population_size]

    def _reproduce_species(
        self, species: Species, offspring_count: int
    ) -> List[NEATGenome]:
        """Reproduce offspring for a species."""
        if not species.members or offspring_count <= 0:
            return []

        offspring = []

        # Sort members by fitness
        sorted_members = sorted(species.members, key=lambda g: g.fitness, reverse=True)

        # Select parents from top performers
        survival_count = max(
            1, int(len(sorted_members) * self.config.survival_threshold)
        )
        breeding_pool = sorted_members[:survival_count]

        for _ in range(offspring_count):
            if len(breeding_pool) == 1:
                # Asexual reproduction (mutation only)
                parent = breeding_pool[0]
                child = NEATGenome(0, self.config)  # ID will be reassigned
                child.nodes = {k: copy.deepcopy(v) for k, v in parent.nodes.items()}
                child.connections = {
                    k: copy.deepcopy(v) for k, v in parent.connections.items()
                }
                child._node_indexer = parent._node_indexer
            else:
                # Sexual reproduction (crossover)
                parent1 = random.choice(breeding_pool)
                parent2 = random.choice(breeding_pool)
                child = parent1.crossover(parent2)

            # Mutate offspring
            child.mutate()
            offspring.append(child)

        return offspring

    def _update_stagnation(self) -> None:
        """Update stagnation counters for species."""
        for species in self.species:
            current_fitness = species.get_average_fitness()

            if not species.fitness_history:
                species.fitness_history.append(current_fitness)
                species.stagnation_count = 0
            else:
                best_historical = max(species.fitness_history)
                if current_fitness > best_historical:
                    species.stagnation_count = 0
                else:
                    species.stagnation_count += 1

                species.fitness_history.append(current_fitness)

    def _remove_extinct_species(self) -> None:
        """Remove extinct (stagnant) species."""
        # Keep at least 2 species
        if len(self.species) <= 2:
            return

        # Remove stagnant species
        self.species = [
            s for s in self.species if not s.is_stagnant(self.config.max_stagnation)
        ]

        # Ensure minimum species count
        if len(self.species) < 2:
            # Recreate species from current population
            self.species = []
            self._speciate()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        if not self.population:
            return {}

        fitness_values = [g.fitness for g in self.population]

        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "num_species": len(self.species),
            "best_fitness": max(fitness_values),
            "average_fitness": statistics.mean(fitness_values),
            "median_fitness": statistics.median(fitness_values),
            "stdev_fitness": (
                statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
            ),
            "fitness_history": self.fitness_history.copy(),
            "species_count_history": self.species_count_history.copy(),
        }

        if self.best_genome:
            stats["best_genome"] = {
                "id": self.best_genome.genome_id,
                "fitness": self.best_genome.fitness,
                "num_nodes": len(self.best_genome.nodes),
                "num_connections": len(
                    [c for c in self.best_genome.connections.values() if c.enabled]
                ),
                "species_id": self.best_genome.species_id,
            }

        # Species statistics
        stats["species_stats"] = []
        for species in self.species:
            species_fitness = [m.fitness for m in species.members]
            stats["species_stats"].append(
                {
                    "id": species.species_id,
                    "size": len(species.members),
                    "best_fitness": max(species_fitness) if species_fitness else 0,
                    "average_fitness": (
                        statistics.mean(species_fitness) if species_fitness else 0
                    ),
                    "stagnation_count": species.stagnation_count,
                }
            )

        return stats

    def save_checkpoint(self, filepath: str) -> None:
        """Save population checkpoint."""
        import json

        checkpoint = {
            "generation": self.generation,
            "config": self.config.__dict__,
            "population": [genome.to_dict() for genome in self.population],
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "innovation_number": self.innovation_number,
            "fitness_history": self.fitness_history,
            "species_count_history": self.species_count_history,
        }

        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, filepath: str) -> None:
        """Load population checkpoint."""
        import json
        from .neat import NEATGenome, NEATConfig

        with open(filepath, "r") as f:
            checkpoint = json.load(f)

        self.generation = checkpoint["generation"]
        self.config = NEATConfig(**checkpoint["config"])
        self.innovation_number = checkpoint["innovation_number"]
        self.fitness_history = checkpoint["fitness_history"]
        self.species_count_history = checkpoint["species_count_history"]

        # Restore population
        self.population = [
            NEATGenome.from_dict(genome_data)
            for genome_data in checkpoint["population"]
        ]

        # Restore best genome
        if checkpoint["best_genome"]:
            self.best_genome = NEATGenome.from_dict(checkpoint["best_genome"])
        else:
            self.best_genome = None

        # Rebuild species
        self._speciate()


def simple_xor_fitness(genome: NEATGenome) -> float:
    """Simple XOR fitness function for testing."""
    # Convert NEAT genome to a simple network and test XOR
    # This is a simplified example

    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_outputs = [0, 1, 1, 0]

    # Simplified network evaluation
    error = 0.0
    for inputs, expected in zip(xor_inputs, xor_outputs):
        # Very simplified forward pass
        output = sum(inputs) % 2  # Placeholder logic
        error += abs(output - expected)

    return 4.0 - error  # Max fitness is 4
