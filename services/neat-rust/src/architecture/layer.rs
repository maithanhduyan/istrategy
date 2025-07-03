#[derive(Debug, Clone)]
pub struct Layer {
    pub size: usize,
    // TODO: Add nodes, groups, connections
}

impl Layer {
    pub fn new(size: usize) -> Self {
        Layer { size }
    }
}
