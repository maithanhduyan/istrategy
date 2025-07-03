// NEAT-RS: NeuroEvolution of Augmenting Topologies implementation in Rust
// Based on NEAT-TS project

// Required for lazy_static in config
extern crate lazy_static;

pub mod architecture;
pub mod methods;
pub mod config;
pub mod neat;
pub mod utils;

pub use crate::neat::Neat;
