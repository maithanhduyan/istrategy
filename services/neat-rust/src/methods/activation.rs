use serde::{Deserialize, Serialize};

/// Available activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Logistic sigmoid: 1/(1+e^(-x))
    Sigmoid,
    
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    
    /// Identity: f(x) = x
    Identity,
    
    /// Step function: x > 0 ? 1 : 0
    Step,
    
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    
    /// Leaky ReLU: x > 0 ? x : 0.01x
    LeakyReLU,
    
    /// Sinusoid: sin(x)
    Sinusoid,
    
    /// Gaussian: e^(-x^2)
    Gaussian,
    
    /// Softsign: x/(1+|x|)
    Softsign,
    
    /// Bent identity: (sqrt(x^2+1)-1)/2 + x
    BentIdentity,
    
    /// Bipolar sigmoid: 2/(1+e^(-x)) - 1
    BipolarSigmoid,
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::Sigmoid
    }
}

/// Activation functions for neural networks
pub mod activation {
    use super::ActivationFunction;
    
    /// Apply activation function to input value
    pub fn activate(func: ActivationFunction, x: f64, derivative: bool) -> f64 {
        match func {
            ActivationFunction::Sigmoid => logistic(x, derivative),
            ActivationFunction::Tanh => tanh(x, derivative),
            ActivationFunction::Identity => identity(x, derivative),
            ActivationFunction::Step => step(x, derivative),
            ActivationFunction::ReLU => relu(x, derivative),
            ActivationFunction::LeakyReLU => leaky_relu(x, derivative),
            ActivationFunction::Sinusoid => sinusoid(x, derivative),
            ActivationFunction::Gaussian => gaussian(x, derivative),
            ActivationFunction::Softsign => softsign(x, derivative),
            ActivationFunction::BentIdentity => bent_identity(x, derivative),
            ActivationFunction::BipolarSigmoid => bipolar_sigmoid(x, derivative),
        }
    }
    
    /// Logistic sigmoid activation function
    pub fn logistic(x: f64, derivative: bool) -> f64 {
        let fx = 1.0 / (1.0 + (-x).exp());
        if derivative {
            fx * (1.0 - fx)
        } else {
            fx
        }
    }
    
    /// Hyperbolic tangent activation function
    pub fn tanh(x: f64, derivative: bool) -> f64 {
        if derivative {
            1.0 - x.tanh().powi(2)
        } else {
            x.tanh()
        }
    }
    
    /// Identity activation function
    pub fn identity(x: f64, derivative: bool) -> f64 {
        if derivative {
            1.0
        } else {
            x
        }
    }
    
    /// Step activation function
    pub fn step(x: f64, derivative: bool) -> f64 {
        if derivative {
            0.0
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    
    /// Rectified Linear Unit activation function
    pub fn relu(x: f64, derivative: bool) -> f64 {
        if derivative {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else if x > 0.0 {
            x
        } else {
            0.0
        }
    }
    
    /// Leaky Rectified Linear Unit activation function
    pub fn leaky_relu(x: f64, derivative: bool) -> f64 {
        let alpha = 0.01;
        if derivative {
            if x > 0.0 {
                1.0
            } else {
                alpha
            }
        } else if x > 0.0 {
            x
        } else {
            alpha * x
        }
    }
    
    /// Sinusoid activation function
    pub fn sinusoid(x: f64, derivative: bool) -> f64 {
        if derivative {
            x.cos()
        } else {
            x.sin()
        }
    }
    
    /// Gaussian activation function
    pub fn gaussian(x: f64, derivative: bool) -> f64 {
        let fx = (-x.powi(2)).exp();
        if derivative {
            -2.0 * x * fx
        } else {
            fx
        }
    }
    
    /// Softsign activation function
    pub fn softsign(x: f64, derivative: bool) -> f64 {
        let d = 1.0 + x.abs();
        if derivative {
            1.0 / d.powi(2)
        } else {
            x / d
        }
    }
    
    /// Bent identity activation function
    pub fn bent_identity(x: f64, derivative: bool) -> f64 {
        if derivative {
            x / (2.0 * (x.powi(2) + 1.0).sqrt()) + 1.0
        } else {
            ((x.powi(2) + 1.0).sqrt() - 1.0) / 2.0 + x
        }
    }
    
    /// Bipolar sigmoid activation function
    pub fn bipolar_sigmoid(x: f64, derivative: bool) -> f64 {
        let fx = 2.0 / (1.0 + (-x).exp()) - 1.0;
        if derivative {
            0.5 * (1.0 - fx.powi(2))
        } else {
            fx
        }
    }
}
