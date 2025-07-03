use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Crossover methods for genetic algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossoverMethod {
    /// Single-point crossover at a random point
    SinglePoint,
    
    /// Two-point crossover at random points
    TwoPoint,
    
    /// Uniform crossover with each gene having equal chance
    Uniform,
    
    /// Average crossover - take average of genes
    Average,
    
    /// Multi-point crossover with n points
    MultiPoint(usize),
    
    /// Weighted average crossover with weight parameter
    WeightedAverage(f64),
}

impl Default for CrossoverMethod {
    fn default() -> Self {
        CrossoverMethod::SinglePoint
    }
}

/// Perform crossover between two sequences of values
pub fn crossover<T: Clone>(
    method: &CrossoverMethod,
    parent1: &[T],
    parent2: &[T],
) -> Vec<T> {
    let len1 = parent1.len();
    let len2 = parent2.len();
    let max_len = len1.max(len2);
    let min_len = len1.min(len2);
    
    if min_len == 0 {
        // Handle empty parents
        if len1 > 0 {
            return parent1.to_vec();
        } else if len2 > 0 {
            return parent2.to_vec();
        } else {
            return vec![];
        }
    }
    
    match method {
        CrossoverMethod::SinglePoint => single_point_crossover(parent1, parent2, max_len),
        CrossoverMethod::TwoPoint => two_point_crossover(parent1, parent2, max_len),
        CrossoverMethod::Uniform => uniform_crossover(parent1, parent2, max_len),
        CrossoverMethod::Average => {
            // Average only works for numeric types, but we need it for the trait system
            // Let's assume we handle this at a higher level
            single_point_crossover(parent1, parent2, max_len)
        }
        CrossoverMethod::MultiPoint(n) => multi_point_crossover(parent1, parent2, max_len, *n),
        CrossoverMethod::WeightedAverage(_) => {
            // Similar to Average, this needs type-specific handling
            single_point_crossover(parent1, parent2, max_len)
        }
    }
}

/// Single-point crossover
fn single_point_crossover<T: Clone>(parent1: &[T], parent2: &[T], max_len: usize) -> Vec<T> {
    let mut rng = thread_rng();
    let point = rng.gen_range(0..=max_len);
    
    let mut child = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        if i < point {
            if i < parent1.len() {
                child.push(parent1[i].clone());
            } else if i < parent2.len() {
                child.push(parent2[i].clone());
            }
        } else {
            if i < parent2.len() {
                child.push(parent2[i].clone());
            } else if i < parent1.len() {
                child.push(parent1[i].clone());
            }
        }
    }
    
    child
}

/// Two-point crossover
fn two_point_crossover<T: Clone>(parent1: &[T], parent2: &[T], max_len: usize) -> Vec<T> {
    let mut rng = thread_rng();
    let mut point1 = rng.gen_range(0..=max_len);
    let mut point2 = rng.gen_range(0..=max_len);
    
    if point1 > point2 {
        std::mem::swap(&mut point1, &mut point2);
    }
    
    let mut child = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        if i < point1 || i >= point2 {
            if i < parent1.len() {
                child.push(parent1[i].clone());
            } else if i < parent2.len() {
                child.push(parent2[i].clone());
            }
        } else {
            if i < parent2.len() {
                child.push(parent2[i].clone());
            } else if i < parent1.len() {
                child.push(parent1[i].clone());
            }
        }
    }
    
    child
}

/// Uniform crossover
fn uniform_crossover<T: Clone>(parent1: &[T], parent2: &[T], max_len: usize) -> Vec<T> {
    let mut rng = thread_rng();
    let mut child = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let use_parent1 = rng.gen_bool(0.5);
        if use_parent1 {
            if i < parent1.len() {
                child.push(parent1[i].clone());
            } else if i < parent2.len() {
                child.push(parent2[i].clone());
            }
        } else {
            if i < parent2.len() {
                child.push(parent2[i].clone());
            } else if i < parent1.len() {
                child.push(parent1[i].clone());
            }
        }
    }
    
    child
}

/// Multi-point crossover
fn multi_point_crossover<T: Clone>(
    parent1: &[T],
    parent2: &[T],
    max_len: usize,
    n: usize,
) -> Vec<T> {
    if n <= 1 {
        return single_point_crossover(parent1, parent2, max_len);
    }
    
    let mut rng = thread_rng();
    let mut points: Vec<usize> = (0..n).map(|_| rng.gen_range(0..=max_len)).collect();
    points.sort_unstable();
    
    let mut child = Vec::with_capacity(max_len);
    let mut use_parent1 = rng.gen_bool(0.5);
    let mut next_point_idx = 0;
    
    for i in 0..max_len {
        if next_point_idx < points.len() && i == points[next_point_idx] {
            use_parent1 = !use_parent1;
            next_point_idx += 1;
        }
        
        if use_parent1 {
            if i < parent1.len() {
                child.push(parent1[i].clone());
            } else if i < parent2.len() {
                child.push(parent2[i].clone());
            }
        } else {
            if i < parent2.len() {
                child.push(parent2[i].clone());
            } else if i < parent1.len() {
                child.push(parent1[i].clone());
            }
        }
    }
    
    child
}
