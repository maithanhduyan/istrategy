use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::utils::{Result, save_to_json, load_from_json, save_to_binary, load_from_binary};
use crate::config::get_config;
use chrono::{DateTime, Utc};

/// Network structure based on NEAT algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Unique ID for the network
    pub id: usize,
    
    /// Number of input nodes
    pub input: usize,
    
    /// Number of output nodes
    pub output: usize,
    
    /// Network weights
    pub weights: Vec<f64>,
    
    /// Network biases
    pub bias: Vec<f64>,
    
    /// Score from last evaluation (fitness)
    pub score: Option<f64>,
    
    /// Mutations history
    pub mutations: Vec<String>,
    
    /// Creation timestamp
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    
    /// Last modified timestamp
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
}

impl Network {
    /// Create a new network with random weights
    pub fn new(input: usize, output: usize) -> Self {
        let config = get_config();
        let weights = (0..(input * 2 + 2 * output))
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();
        let bias = (0..3).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        
        Network {
            id: crate::config::get_next_network_id(&config),
            input,
            output,
            weights,
            bias,
            score: None,
            mutations: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
    
    /// Save network to JSON file
    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        save_to_json(self, path)
    }
    
    /// Load network from JSON file
    pub fn load_from_json(path: &Path) -> Result<Self> {
        load_from_json(path)
    }
    
    /// Save network to binary file
    pub fn save_to_binary(&self, path: &Path) -> Result<()> {
        save_to_binary(self, path)
    }
    
    /// Load network from binary file
    pub fn load_from_binary(path: &Path) -> Result<Self> {
        load_from_binary(path)
    }
    
    /// Record a mutation
    pub fn record_mutation(&mut self, mutation_description: &str) {
        self.mutations.push(mutation_description.to_string());
        self.updated_at = Utc::now();
    }

    /// Forward propagate inputs through a 2-2-1 perceptron
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        // Hidden layer với sigmoid activation
        let h1 =
            Self::sigmoid(inputs[0] * self.weights[0] + inputs[1] * self.weights[1] + self.bias[0]);
        let h2 =
            Self::sigmoid(inputs[0] * self.weights[2] + inputs[1] * self.weights[3] + self.bias[1]);
        // Output layer với sigmoid activation
        let o = Self::sigmoid(h1 * self.weights[4] + h2 * self.weights[5] + self.bias[2]);
        vec![o]
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Sigmoid derivative
    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    /// Đánh giá lỗi MSE trên tập dữ liệu
    pub fn evaluate(&self, data: &[([f64; 2], f64)]) -> f64 {
        let mut error = 0.0;
        for (input, target) in data.iter() {
            let output = self.forward(&input[..]);
            let o = output[0];
            error += (target - o).powi(2);
        }
        error / data.len() as f64
    }

    /// Đột biến mạng với adaptive mutation rate
    pub fn mutate(&mut self) {
        let mutation_rate = 0.1;
        let mutation_strength = 0.5;

        for w in &mut self.weights {
            if rand::random::<f64>() < mutation_rate {
                *w += (rand::random::<f64>() - 0.5) * mutation_strength;
                // Clamp weights để tránh exploding gradients
                *w = w.clamp(-5.0, 5.0);
            }
        }
        for b in &mut self.bias {
            if rand::random::<f64>() < mutation_rate {
                *b += (rand::random::<f64>() - 0.5) * mutation_strength;
                *b = b.clamp(-5.0, 5.0);
            }
        }
    }

    /// Huấn luyện mạng với backpropagation thực tế
    pub fn train(&mut self, data: &[([f64; 2], f64)]) {
        let learning_rate = 0.5;
        let epochs = 100;

        for _epoch in 0..epochs {
            for (input, target) in data.iter() {
                // Forward pass
                let h1 = Self::sigmoid(
                    input[0] * self.weights[0] + input[1] * self.weights[1] + self.bias[0],
                );
                let h2 = Self::sigmoid(
                    input[0] * self.weights[2] + input[1] * self.weights[3] + self.bias[1],
                );
                let output =
                    Self::sigmoid(h1 * self.weights[4] + h2 * self.weights[5] + self.bias[2]);

                // Calculate error
                let error = target - output;

                // Backward pass
                // Output layer gradients
                let d_output = error * Self::sigmoid_derivative(output);

                // Hidden layer gradients
                let d_h1 = d_output * self.weights[4] * Self::sigmoid_derivative(h1);
                let d_h2 = d_output * self.weights[5] * Self::sigmoid_derivative(h2);

                // Update output layer weights and bias
                self.weights[4] += learning_rate * d_output * h1;
                self.weights[5] += learning_rate * d_output * h2;
                self.bias[2] += learning_rate * d_output;

                // Update hidden layer weights and biases
                self.weights[0] += learning_rate * d_h1 * input[0];
                self.weights[1] += learning_rate * d_h1 * input[1];
                self.weights[2] += learning_rate * d_h2 * input[0];
                self.weights[3] += learning_rate * d_h2 * input[1];
                self.bias[0] += learning_rate * d_h1;
                self.bias[1] += learning_rate * d_h2;
            }
        }
    }
}
