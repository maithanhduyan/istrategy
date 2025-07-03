use std::sync::atomic::{AtomicUsize, Ordering};

/// Global configuration for NEAT-Rust implementation
#[derive(Debug)]
pub struct Config {
    /// Current network ID counter
    pub network_id_counter: AtomicUsize,
    
    /// Default neuron activation bias
    pub neuron_activation_bias: f64,
    
    /// Default neuron persistence value
    pub neuron_persistence: f64,
    
    /// Maximum layers in network
    pub max_layers: usize,
    
    /// Default network mutation rate
    pub mutation_rate: f64,
    
    /// Default network crossover probability
    pub crossover_rate: f64,
    
    /// Whether to use webworkers in WASM environment
    pub use_webworkers: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            network_id_counter: AtomicUsize::new(0),
            neuron_activation_bias: 0.0,
            neuron_persistence: 0.0,
            max_layers: 3,
            mutation_rate: 0.3,
            crossover_rate: 0.5,
            use_webworkers: false,
        }
    }
}

/// Get next unique network ID
pub fn get_next_network_id(_config: &Config) -> usize {
    GLOBAL_CONFIG.lock().unwrap().network_id_counter.fetch_add(1, Ordering::SeqCst)
}

/// Reset network ID counter
pub fn reset_network_id_counter(config: &mut Config) {
    config.network_id_counter.store(0, Ordering::SeqCst);
}

lazy_static::lazy_static! {
    /// Global config instance
    pub static ref GLOBAL_CONFIG: std::sync::Mutex<Config> = std::sync::Mutex::new(Config::default());
}

/// Get global configuration
pub fn get_config() -> Config {
    Config::default()
}

/// Update global configuration
pub fn update_global_config_mutex(f: impl FnOnce(&mut Config)) {
    let mut global = GLOBAL_CONFIG.lock().unwrap();
    f(&mut global);
}
