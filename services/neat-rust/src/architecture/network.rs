#[derive(Debug, Clone)]
pub struct Network {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl Network {
    pub fn new(input: usize, output: usize) -> Self {
        let weights = (0..(input * 2 + 2 * output))
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();
        let bias = (0..3).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        Network {
            input,
            output,
            weights,
            bias,
        }
    }

    /// Forward propagate inputs through a 2-2-1 perceptron
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        // Hidden layer với sigmoid activation
        let h1 = Self::sigmoid(inputs[0] * self.weights[0] + inputs[1] * self.weights[1] + self.bias[0]);
        let h2 = Self::sigmoid(inputs[0] * self.weights[2] + inputs[1] * self.weights[3] + self.bias[1]);
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
                let h1 = Self::sigmoid(input[0] * self.weights[0] + input[1] * self.weights[1] + self.bias[0]);
                let h2 = Self::sigmoid(input[0] * self.weights[2] + input[1] * self.weights[3] + self.bias[1]);
                let output = Self::sigmoid(h1 * self.weights[4] + h2 * self.weights[5] + self.bias[2]);
                
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
