// NEAT-Rust methods module
// Collection of all algorithm methods used in NEAT

pub mod activation;
pub mod connection;
pub mod cost;
pub mod crossover;
pub mod gating;
pub mod methods;
pub mod mutation;
pub mod rate;
pub mod selection;

/// Re-export all method types for convenient access
pub use activation::ActivationFunction;
pub use crossover::CrossoverMethod;
pub use mutation::MutationMethod;
pub use selection::SelectionMethod;

/// Export a combined methods structure
pub struct Methods;

impl Methods {
    /// Get all available methods
    pub fn all() -> &'static Methods {
        &Methods
    }
}
