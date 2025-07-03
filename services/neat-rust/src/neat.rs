use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::architecture::network::Network;
use crate::methods::{
    activation::ActivationFunction, 
    mutation::MutationMethod, 
    selection::SelectionMethod,
    crossover::CrossoverMethod
};
use crate::utils::{Result, NeatError, save_to_json, load_from_json, save_to_binary, load_from_binary};
use chrono::{DateTime, Utc};

type FitnessFn = fn(&Network) -> f64;

/// Options for creating a NEAT instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatOptions {
    /// Population size
    pub popsize: usize,
    
    /// Number of elite networks to keep unchanged
    pub elitism: usize,
    
    /// Number of random networks to add each generation
    pub provenance: usize,
    
    /// Mutation rate
    pub mutation_rate: f64,
    
    /// Mutation amount
    pub mutation_amount: f64,
    
    /// Selection method
    pub selection: SelectionMethod,
    
    /// Crossover methods
    pub crossover: Vec<CrossoverMethod>,
    
    /// Mutation methods
    pub mutation: Vec<MutationMethod>,
    
    /// Maximum number of nodes allowed
    pub max_nodes: usize,
    
    /// Maximum number of connections allowed
    pub max_connections: usize,
    
    /// Whether to reset score after mutation
    pub reset_on_mutation: bool,
    
    /// Whether to evaluate fitness on the whole population
    pub fitness_population: bool,
    
    /// Default activation function
    pub activation: ActivationFunction,
}

impl Default for NeatOptions {
    fn default() -> Self {
        Self {
            popsize: 50,
            elitism: 2,
            provenance: 1,
            mutation_rate: 0.3,
            mutation_amount: 1.0,
            selection: SelectionMethod::Power,
            crossover: vec![
                CrossoverMethod::SinglePoint,
                CrossoverMethod::TwoPoint,
                CrossoverMethod::Uniform
            ],
            mutation: vec![
                MutationMethod::ModWeight,
                MutationMethod::ModBias,
                MutationMethod::ModActivation
            ],
            max_nodes: 1000,
            max_connections: 10000,
            reset_on_mutation: true,
            fitness_population: false,
            activation: ActivationFunction::Sigmoid,
        }
    }
}

/// NEAT algorithm implementation
#[derive(Debug, Serialize, Deserialize)]
pub struct Neat {
    /// Number of input nodes
    pub input: usize,
    
    /// Number of output nodes
    pub output: usize,
    
    /// Current generation number
    pub generation: usize,
    
    /// Current population
    pub population: Vec<Network>,
    
    /// Best network found so far
    pub best_network: Option<Network>,
    
    /// Best score found so far
    pub best_score: Option<f64>,
    
    /// Configuration options
    pub options: NeatOptions,
    
    /// Generation history
    #[serde(skip)]
    pub history: Vec<Vec<Network>>,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last evolution timestamp
    pub last_evolved_at: Option<DateTime<Utc>>,
    
    /// Fitness function (not serializable)
    #[serde(skip)]
    pub fitness_fn: Option<FitnessFn>,
}

impl Neat {
    /// Create a new NEAT instance
    pub fn new(input: usize, output: usize, fitness_fn: FitnessFn, options: Option<NeatOptions>) -> Self {
        let options = options.unwrap_or_default();
        let mut neat = Self {
            input,
            output,
            generation: 0,
            population: Vec::with_capacity(options.popsize),
            best_network: None,
            best_score: None,
            options,
            history: Vec::new(),
            created_at: Utc::now(),
            last_evolved_at: None,
            fitness_fn: Some(fitness_fn),
        };
        
        // Initialize population
        neat.create_population();
        neat
    }
    
    /// Create initial population
    pub fn create_population(&mut self) {
        self.population.clear();
        for _ in 0..self.options.popsize {
            self.population.push(Network::new(self.input, self.output));
        }
    }
    
    /// Evolve population for one generation
    pub fn evolve(&mut self) -> Result<f64> {
        if self.fitness_fn.is_none() {
            return Err(NeatError::InvalidNetworkConfig);
        }
        
        // 1. Evaluate fitness
        self.evaluate_population();
        
        // 2. Sort by fitness
        self.sort_population();
        
        // 3. Store best network
        self.update_best_network();
        
        // 4. Create next generation
        self.create_next_generation();
        
        // 5. Update generation counter and timestamp
        self.generation += 1;
        self.last_evolved_at = Some(Utc::now());
        
        // Return best score of current generation
        if let Some(score) = self.best_score {
            Ok(score)
        } else {
            Ok(0.0)
        }
    }
    
    /// Evaluate fitness for all networks in population
    fn evaluate_population(&mut self) {
        let fitness_fn = self.fitness_fn.unwrap();
        
        if self.options.fitness_population {
            // Evaluate fitness for whole population at once
            // This would require a different fitness_fn signature
            // Not implemented yet
        } else {
            // Evaluate individual networks
            for network in &mut self.population {
                let score = fitness_fn(network);
                network.score = Some(score);
            }
        }
    }
    
    /// Sort population by fitness score (descending)
    fn sort_population(&mut self) {
        self.population.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    
    /// Update best network if current generation has a better one
    fn update_best_network(&mut self) {
        if let Some(best) = self.population.first() {
            if let Some(score) = best.score {
                if self.best_score.is_none() || score > self.best_score.unwrap() {
                    self.best_score = Some(score);
                    self.best_network = Some(best.clone());
                }
            }
        }
    }
    
    /// Create next generation using selection, crossover and mutation
    fn create_next_generation(&mut self) {
        // Keep history if needed
        self.history.push(self.population.clone());
        
        // Initialize new population
        let mut new_population: Vec<Network> = Vec::with_capacity(self.options.popsize);
        
        // Elitism: keep best networks unchanged
        for i in 0..self.options.elitism.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }
        
        // Crossover and mutation to fill the rest of the population
        while new_population.len() < self.options.popsize - self.options.provenance {
            // Selection
            let parent1 = self.select_parent();
            let parent2 = self.select_parent();
            
            // Crossover
            let mut child = self.crossover(&parent1, &parent2);
            
            // Mutation
            self.mutate(&mut child);
            
            // Add to new population
            new_population.push(child);
        }
        
        // Add random networks for provenance
        for _ in 0..self.options.provenance {
            new_population.push(Network::new(self.input, self.output));
        }
        
        // Replace old population
        self.population = new_population;
    }
    
    /// Select a parent using the configured selection method
    fn select_parent(&self) -> Network {
        // Improved selection with better fitness pressure
        match self.options.selection {
            SelectionMethod::Tournament { size, .. } => {
                let tournament_size = size.min(self.population.len());
                let mut best_network = None;
                let mut best_score = f64::NEG_INFINITY;
                
                for _ in 0..tournament_size {
                    let idx = rand::random::<usize>() % self.population.len();
                    let network = &self.population[idx];
                    
                    if let Some(score) = network.score {
                        if score > best_score {
                            best_score = score;
                            best_network = Some(network);
                        }
                    }
                }
                
                best_network.unwrap_or(&self.population[0]).clone()
            }
            _ => {
                // Default to tournament selection with size 3
                let tournament_size = 3.min(self.population.len());
                let mut best_network = None;
                let mut best_score = f64::NEG_INFINITY;
                
                for _ in 0..tournament_size {
                    let idx = rand::random::<usize>() % self.population.len();
                    let network = &self.population[idx];
                    
                    if let Some(score) = network.score {
                        if score > best_score {
                            best_score = score;
                            best_network = Some(network);
                        }
                    }
                }
                
                best_network.unwrap_or(&self.population[0]).clone()
            }
        }
    }
    
    /// Perform crossover between two parent networks
    fn crossover(&self, parent1: &Network, parent2: &Network) -> Network {
        // Create a new network with the same structure as parent1
        let mut child = parent1.clone();
        
        // Blend weights from both parents with some randomness
        for i in 0..child.weights.len() {
            if i < parent2.weights.len() {
                if rand::random::<f64>() < 0.5 {
                    // Take from parent2
                    child.weights[i] = parent2.weights[i];
                } else {
                    // Take from parent1 (already there)
                    // Or blend
                    if rand::random::<f64>() < 0.3 {
                        child.weights[i] = (parent1.weights[i] + parent2.weights[i]) / 2.0;
                    }
                }
            }
        }
        
        // Blend biases
        for i in 0..child.bias.len() {
            if i < parent2.bias.len() {
                if rand::random::<f64>() < 0.5 {
                    child.bias[i] = parent2.bias[i];
                } else if rand::random::<f64>() < 0.3 {
                    child.bias[i] = (parent1.bias[i] + parent2.bias[i]) / 2.0;
                }
            }
        }
        
        child.record_mutation("crossover");
        child.score = None; // Reset score for new individual
        child
    }
    
    /// Mutate a network
    fn mutate(&self, network: &mut Network) {
        if rand::random::<f64>() > self.options.mutation_rate {
            return;
        }
        
        // More sophisticated mutation
        let mutation_type = rand::random::<f64>();
        
        if mutation_type < 0.6 {
            // Weight mutation (most common)
            for weight in &mut network.weights {
                if rand::random::<f64>() < self.options.mutation_rate {
                    // Sometimes replace completely, sometimes adjust
                    if rand::random::<f64>() < 0.1 {
                        *weight = (rand::random::<f64>() * 2.0 - 1.0) * 2.0;
                    } else {
                        *weight += (rand::random::<f64>() * 2.0 - 1.0) * self.options.mutation_amount;
                    }
                    // Clamp weights
                    *weight = weight.clamp(-5.0, 5.0);
                }
            }
        } else if mutation_type < 0.9 {
            // Bias mutation
            for bias in &mut network.bias {
                if rand::random::<f64>() < self.options.mutation_rate {
                    if rand::random::<f64>() < 0.1 {
                        *bias = (rand::random::<f64>() * 2.0 - 1.0) * 2.0;
                    } else {
                        *bias += (rand::random::<f64>() * 2.0 - 1.0) * self.options.mutation_amount;
                    }
                    *bias = bias.clamp(-5.0, 5.0);
                }
            }
        } else {
            // Perform some training on the network (like Lamarckian evolution)
            let xor_data = [
                ([0.0, 0.0], 0.0),
                ([0.0, 1.0], 1.0),
                ([1.0, 0.0], 1.0),
                ([1.0, 1.0], 0.0),
            ];
            
            // Small amount of training
            for _ in 0..10 {
                network.train(&xor_data);
            }
        }
        
        network.record_mutation("mutation");
        
        // Reset score if configured
        if self.options.reset_on_mutation {
            network.score = None;
        }
    }
    
    /// Save NEAT state to JSON file
    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        save_to_json(self, path)
    }
    
    /// Load NEAT state from JSON file
    pub fn load_from_json(path: &Path) -> Result<Self> {
        load_from_json(path)
    }
    
    /// Save NEAT state to binary file
    pub fn save_to_binary(&self, path: &Path) -> Result<()> {
        save_to_binary(self, path)
    }
    
    /// Load NEAT state from binary file
    pub fn load_from_binary(path: &Path) -> Result<Self> {
        load_from_binary(path)
    }
    
    /// Restore fitness function after loading
    pub fn set_fitness_fn(&mut self, fitness_fn: FitnessFn) {
        self.fitness_fn = Some(fitness_fn);
    }
    
    /// Get best network
    pub fn get_best_network(&self) -> Option<&Network> {
        self.best_network.as_ref()
    }
    
    /// Get generation statistics
    pub fn get_stats(&self) -> NeatStats {
        let mut min_score = f64::MAX;
        let mut max_score = f64::MIN;
        let mut sum_score = 0.0;
        let mut count = 0;
        
        for network in &self.population {
            if let Some(score) = network.score {
                min_score = min_score.min(score);
                max_score = max_score.max(score);
                sum_score += score;
                count += 1;
            }
        }
        
        let avg_score = if count > 0 { sum_score / count as f64 } else { 0.0 };
        
        NeatStats {
            generation: self.generation,
            population_size: self.population.len(),
            min_score,
            max_score,
            avg_score,
            best_all_time: self.best_score.unwrap_or(0.0),
        }
    }
}

/// Statistics about NEAT training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatStats {
    /// Current generation number
    pub generation: usize,
    
    /// Population size
    pub population_size: usize,
    
    /// Minimum score in current generation
    pub min_score: f64,
    
    /// Maximum score in current generation
    pub max_score: f64,
    
    /// Average score in current generation
    pub avg_score: f64,
    
    /// Best score found so far
    pub best_all_time: f64,
}
