use neat_rust::{
    Neat,
    architecture::network::Network,
    neat::NeatOptions,
};
use std::path::Path;

/// Test fitness function for XOR problem
fn xor_fitness(network: &Network) -> f64 {
    let xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];
    
    let error = network.evaluate(&xor_data);
    (-error * 10.0).exp()
}

#[test]
fn test_neat_basic_functionality() {
    let mut neat = Neat::new(2, 1, xor_fitness, None);
    
    // Test initial state
    assert_eq!(neat.input, 2);
    assert_eq!(neat.output, 1);
    assert_eq!(neat.generation, 0);
    assert_eq!(neat.population.len(), 50); // default popsize
    
    // Test evolution
    let result = neat.evolve();
    assert!(result.is_ok());
    assert_eq!(neat.generation, 1);
    
    // Test stats
    let stats = neat.get_stats();
    assert_eq!(stats.generation, 1);
    assert_eq!(stats.population_size, 50);
    assert!(stats.avg_score >= 0.0);
    assert!(stats.max_score >= stats.min_score);
}

#[test]
fn test_neat_with_custom_options() {
    let mut options = NeatOptions::default();
    options.popsize = 20;
    options.elitism = 3;
    options.mutation_rate = 0.5;
    options.mutation_amount = 0.8;
    options.reset_on_mutation = false;
    
    let mut neat = Neat::new(2, 1, xor_fitness, Some(options));
    
    assert_eq!(neat.options.popsize, 20);
    assert_eq!(neat.options.elitism, 3);
    assert_eq!(neat.options.mutation_rate, 0.5);
    assert_eq!(neat.options.mutation_amount, 0.8);
    assert_eq!(neat.options.reset_on_mutation, false);
    
    // Test evolution with custom options
    let result = neat.evolve();
    assert!(result.is_ok());
    assert_eq!(neat.population.len(), 20);
}

#[test]
fn test_neat_convergence() {
    let mut options = NeatOptions::default();
    options.popsize = 30;
    options.elitism = 5;
    options.mutation_rate = 0.7;
    options.mutation_amount = 0.5;
    options.reset_on_mutation = false;
    
    let mut neat = Neat::new(2, 1, xor_fitness, Some(options));
    
    // Run for multiple generations
    let mut best_fitness = 0.0;
    for _gen in 0..10 {
        let result = neat.evolve();
        assert!(result.is_ok());
        
        if let Some(best) = neat.get_best_network() {
            if let Some(fitness) = best.score {
                if fitness > best_fitness {
                    best_fitness = fitness;
                }
            }
        }
    }
    
    // Should have made some progress
    assert!(best_fitness > 0.1);
}

#[test]
fn test_neat_serialization() {
    let mut neat = Neat::new(2, 1, xor_fitness, None);
    
    // Run a few generations
    for _ in 0..3 {
        let _ = neat.evolve();
    }
    
    // Save to JSON
    let path = Path::new("test_neat_state.json");
    let result = neat.save_to_json(path);
    assert!(result.is_ok());
    
    // Load from JSON
    let loaded_neat_result = Neat::load_from_json(path);
    assert!(loaded_neat_result.is_ok());
    
    let mut loaded_neat = loaded_neat_result.unwrap();
    loaded_neat.set_fitness_fn(xor_fitness);
    
    // Test that loaded state works
    assert_eq!(loaded_neat.input, neat.input);
    assert_eq!(loaded_neat.output, neat.output);
    assert_eq!(loaded_neat.generation, neat.generation);
    assert_eq!(loaded_neat.population.len(), neat.population.len());
    
    // Clean up
    std::fs::remove_file(path).ok();
}

#[test]
fn test_network_serialization() {
    let network = Network::new(2, 1);
    
    // Save to JSON
    let path = Path::new("test_network.json");
    let result = network.save_to_json(path);
    assert!(result.is_ok());
    
    // Load from JSON
    let loaded_network = Network::load_from_json(path);
    assert!(loaded_network.is_ok());
    
    let loaded = loaded_network.unwrap();
    assert_eq!(loaded.input, network.input);
    assert_eq!(loaded.output, network.output);
    assert_eq!(loaded.weights.len(), network.weights.len());
    assert_eq!(loaded.bias.len(), network.bias.len());
    
    // Clean up
    std::fs::remove_file(path).ok();
}

#[test]
fn test_activation_functions() {
    use neat_rust::methods::activation::activation;
    
    // Test all activation functions
    let x = 0.5;
    
    let sigmoid = activation::logistic(x, false);
    assert!(sigmoid > 0.0 && sigmoid < 1.0);
    
    let tanh = activation::tanh(x, false);
    assert!(tanh > -1.0 && tanh < 1.0);
    
    let relu = activation::relu(x, false);
    assert_eq!(relu, x);
    
    let identity = activation::identity(x, false);
    assert_eq!(identity, x);
    
    // Test derivatives
    let sigmoid_deriv = activation::logistic(x, true);
    assert!(sigmoid_deriv > 0.0);
    
    let tanh_deriv = activation::tanh(x, true);
    assert!(tanh_deriv > 0.0);
}

#[test]
fn test_mutation_methods() {
    use neat_rust::methods::mutation::{default_mutation_methods, select_mutation_method};
    
    let methods = default_mutation_methods();
    assert!(!methods.is_empty());
    
    // Test selection
    let selected = select_mutation_method(&methods);
    assert!(methods.iter().any(|(m, _)| std::mem::discriminant(m) == std::mem::discriminant(&selected)));
}

#[test]
fn test_crossover_methods() {
    use neat_rust::methods::crossover::{crossover, CrossoverMethod};
    
    let parent1 = vec![1.0, 2.0, 3.0, 4.0];
    let parent2 = vec![5.0, 6.0, 7.0, 8.0];
    
    let child = crossover(&CrossoverMethod::SinglePoint, &parent1, &parent2);
    assert_eq!(child.len(), 4);
    
    let child = crossover(&CrossoverMethod::TwoPoint, &parent1, &parent2);
    assert_eq!(child.len(), 4);
    
    let child = crossover(&CrossoverMethod::Uniform, &parent1, &parent2);
    assert_eq!(child.len(), 4);
}

#[test]
fn test_selection_methods() {
    use neat_rust::methods::selection::{select, SelectionMethod};
    
    let population = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let get_score = |x: &f64| *x;
    
    let selected = select(&SelectionMethod::FitnessProportionate, &population, get_score);
    assert_eq!(selected.len(), population.len());
    
    let selected = select(&SelectionMethod::Power, &population, get_score);
    assert_eq!(selected.len(), population.len());
    
    let selected = select(&SelectionMethod::Tournament { size: 3, probability: 0.7 }, &population, get_score);
    assert_eq!(selected.len(), population.len());
}

#[test]
fn test_network_training() {
    let mut network = Network::new(2, 1);
    
    let xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];
    
    let initial_error = network.evaluate(&xor_data);
    network.train(&xor_data);
    let final_error = network.evaluate(&xor_data);
    
    // Training should improve the network (reduce error)
    assert!(final_error <= initial_error);
}

#[test]
fn test_network_forward_pass() {
    let network = Network::new(2, 1);
    
    let input = vec![0.5, 0.5];
    let output = network.forward(&input);
    
    assert_eq!(output.len(), 1);
    assert!(output[0] >= 0.0 && output[0] <= 1.0); // Sigmoid output
}

#[test]
fn test_network_mutation() {
    let mut network = Network::new(2, 1);
    let original_weights = network.weights.clone();
    let original_bias = network.bias.clone();
    
    // Mutate multiple times to ensure something changes
    for _ in 0..100 {
        network.mutate();
    }
    
    // At least some weights or biases should have changed
    let weights_changed = original_weights != network.weights;
    let bias_changed = original_bias != network.bias;
    
    assert!(weights_changed || bias_changed);
}
