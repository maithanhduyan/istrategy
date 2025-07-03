use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Mutation methods for NEAT networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationMethod {
    /// Add a new node between an existing connection
    AddNode,
    
    /// Remove a node
    RemoveNode,
    
    /// Add a new connection between nodes
    AddConnection,
    
    /// Remove a connection
    RemoveConnection,
    
    /// Modify connection weights
    ModWeight,
    
    /// Modify node bias
    ModBias,
    
    /// Modify node activation function
    ModActivation,
    
    /// Add self-connection to node
    AddSelfConnection,
    
    /// Remove self-connection
    RemoveSelfConnection,
    
    /// Add a gate to a connection
    AddGate,
    
    /// Remove a gate from connection
    RemoveGate,
    
    /// Add recurrent (backward) connection
    AddBackwardConnection,
    
    /// Remove recurrent connection
    RemoveBackwardConnection,
    
    /// Swap two nodes
    SwapNodes,
}

/// Default mutation methods with probabilities
pub fn default_mutation_methods() -> Vec<(MutationMethod, f64)> {
    vec![
        (MutationMethod::ModWeight, 0.4),
        (MutationMethod::ModBias, 0.3),
        (MutationMethod::ModActivation, 0.2),
        (MutationMethod::AddConnection, 0.05),
        (MutationMethod::AddNode, 0.02),
        (MutationMethod::RemoveConnection, 0.02),
        (MutationMethod::RemoveNode, 0.01),
    ]
}

/// Select a mutation method based on probabilities
pub fn select_mutation_method(methods: &[(MutationMethod, f64)]) -> MutationMethod {
    let mut rng = thread_rng();
    
    // Calculate sum of all probabilities
    let sum: f64 = methods.iter().map(|(_, prob)| prob).sum();
    
    // Generate random value between 0 and sum
    let mut r = rng.gen::<f64>() * sum;
    
    // Find the selected method
    for (method, prob) in methods {
        r -= prob;
        if r <= 0.0 {
            return method.clone();
        }
    }
    
    // Default to the first method
    methods.first().map(|(m, _)| m.clone()).unwrap_or(MutationMethod::ModWeight)
}
