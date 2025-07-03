//! Utility functions for NEAT-Rust

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeatError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    
    #[error("Binary serialization error: {0}")]
    BincodeSerde(#[from] bincode::Error),
    
    #[error("Invalid network configuration")]
    InvalidNetworkConfig,
    
    #[error("Invalid input size: expected {expected}, got {actual}")]
    InvalidInputSize { expected: usize, actual: usize },
    
    #[error("No solution found after {0} generations")]
    NoSolution(usize),
}

/// Result type for NEAT operations
pub type Result<T> = std::result::Result<T, NeatError>;

/// Helper function to save any serializable struct to JSON file
pub fn save_to_json<T: Serialize>(data: &T, path: &Path) -> Result<()> {
    let serialized = serde_json::to_string_pretty(data)?;
    let mut file = File::create(path)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

/// Helper function to load any deserializable struct from JSON file
pub fn load_from_json<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let deserialized = serde_json::from_str(&contents)?;
    Ok(deserialized)
}

/// Helper function to save any serializable struct to binary file
pub fn save_to_binary<T: Serialize>(data: &T, path: &Path) -> Result<()> {
    let serialized = bincode::serialize(data)?;
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    Ok(())
}

/// Helper function to load any deserializable struct from binary file
pub fn load_from_binary<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let deserialized = bincode::deserialize(&buffer)?;
    Ok(deserialized)
}

/// Check if file exists
pub fn file_exists(path: &Path) -> bool {
    path.exists() && path.is_file()
}

/// Create directory if not exists
pub fn ensure_dir_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get current timestamp as string (YYYYMMDD_HHMMSS)
pub fn get_timestamp() -> String {
    use chrono::Local;
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}
