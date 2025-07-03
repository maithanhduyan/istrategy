use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Selection methods for genetic algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    /// Select individuals based on their fitness proportionate to the total
    FitnessProportionate,
    
    /// Power selection with given power factor
    Power,
    
    /// Tournament selection with given size and probability
    Tournament {
        /// Number of individuals in each tournament
        size: usize,
        
        /// Probability of selecting the best individual
        probability: f64,
    },
}

impl Default for SelectionMethod {
    fn default() -> Self {
        SelectionMethod::Power
    }
}

/// Apply selection to a population
pub fn select<'a, T>(
    method: &SelectionMethod,
    population: &'a [T],
    get_score: impl Fn(&T) -> f64,
) -> Vec<&'a T> {
    match method {
        SelectionMethod::FitnessProportionate => fitness_proportionate(population, &get_score),
        SelectionMethod::Power => power_selection(population, &get_score),
        SelectionMethod::Tournament { size, probability } => {
            tournament_selection(population, &get_score, *size, *probability)
        }
    }
}

/// Fitness proportionate selection (roulette wheel)
fn fitness_proportionate<'a, T>(population: &'a [T], get_score: &impl Fn(&T) -> f64) -> Vec<&'a T> {
    let mut rng = thread_rng();
    let total_fitness: f64 = population.iter().map(get_score).sum();
    
    if total_fitness <= 0.0 {
        // If total fitness is non-positive, return random selection
        return (0..population.len())
            .map(|_| &population[rng.gen_range(0..population.len())])
            .collect();
    }
    
    (0..population.len())
        .map(|_| {
            let mut r = rng.gen::<f64>() * total_fitness;
            for individual in population {
                r -= get_score(individual);
                if r <= 0.0 {
                    return individual;
                }
            }
            // Default to last individual (should never happen)
            population.last().unwrap()
        })
        .collect()
}

/// Power selection - bias towards fitter individuals
fn power_selection<'a, T>(population: &'a [T], get_score: &impl Fn(&T) -> f64) -> Vec<&'a T> {
    // Sort population by fitness (highest first)
    let mut indexed_pop: Vec<(usize, f64)> = population
        .iter()
        .enumerate()
        .map(|(i, ind)| (i, get_score(ind)))
        .collect();
    
    indexed_pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Apply power law selection
    let mut rng = thread_rng();
    let power_factor = 4.0; // Higher values favor top individuals more
    
    (0..population.len())
        .map(|_| {
            let index = (rng.gen::<f64>().powf(power_factor) * population.len() as f64) as usize;
            let selected_idx = indexed_pop.get(index.min(population.len() - 1)).unwrap().0;
            &population[selected_idx]
        })
        .collect()
}

/// Tournament selection
fn tournament_selection<'a, T>(
    population: &'a [T],
    get_score: &impl Fn(&T) -> f64,
    size: usize,
    probability: f64,
) -> Vec<&'a T> {
    let mut rng = thread_rng();
    let actual_size = size.min(population.len());
    
    (0..population.len())
        .map(|_| {
            // Select tournament participants
            let mut tournament: Vec<usize> = (0..population.len()).collect();
            tournament.shuffle(&mut rng);
            tournament.truncate(actual_size);
            
            // Sort tournament by fitness
            tournament.sort_by(|&a, &b| {
                get_score(&population[b])
                    .partial_cmp(&get_score(&population[a]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Select based on probability
            let mut selected_idx = tournament[0];
            for i in 1..tournament.len() {
                if rng.gen::<f64>() < probability {
                    selected_idx = tournament[i];
                } else {
                    break;
                }
            }
            
            &population[selected_idx]
        })
        .collect()
}
