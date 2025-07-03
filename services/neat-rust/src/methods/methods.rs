// Collection of all NEAT-RS methods
use serde::{Deserialize, Serialize};

pub use crate::methods::{
    activation::{self, ActivationFunction},
    mutation::{self, MutationMethod},
    selection::{self, SelectionMethod},
    crossover::{self, CrossoverMethod},
    cost, gating, connection, rate
};

/// Methods collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Methods {
    /// Available activation functions
    pub activation: Vec<ActivationFunction>,
    
    /// Available mutation methods
    pub mutation: Vec<MutationMethod>,
    
    /// Available selection methods
    pub selection: Vec<SelectionMethod>,
    
    /// Available crossover methods
    pub crossover: Vec<CrossoverMethod>,
}

impl Default for Methods {
    fn default() -> Self {
        Self {
            activation: vec![
                ActivationFunction::Sigmoid,
                ActivationFunction::Tanh,
                ActivationFunction::ReLU,
                ActivationFunction::LeakyReLU,
                ActivationFunction::Sinusoid,
                ActivationFunction::Gaussian,
                ActivationFunction::Softsign,
                ActivationFunction::BentIdentity,
                ActivationFunction::BipolarSigmoid,
            ],
            mutation: vec![
                MutationMethod::ModWeight,
                MutationMethod::ModBias,
                MutationMethod::ModActivation,
                MutationMethod::AddConnection,
                MutationMethod::AddNode,
            ],
            selection: vec![
                SelectionMethod::FitnessProportionate,
                SelectionMethod::Power,
                SelectionMethod::Tournament {
                    size: 3,
                    probability: 0.8,
                },
            ],
            crossover: vec![
                CrossoverMethod::SinglePoint,
                CrossoverMethod::TwoPoint,
                CrossoverMethod::Uniform,
                CrossoverMethod::Average,
            ],
        }
    }
}
