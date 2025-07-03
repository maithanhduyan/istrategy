#[derive(Debug, Clone)]
pub struct Group {
    pub size: usize,
    // TODO: Add nodes and connections
}

impl Group {
    pub fn new(size: usize) -> Self {
        Group { size }
    }
}
