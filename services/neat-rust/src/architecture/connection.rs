use rand;

#[derive(Debug, Clone)]
pub struct Connection {
    pub from: Option<usize>, // Node index or reference
    pub to: Option<usize>,   // Node index or reference
    pub gain: f64,
    pub weight: f64,
    pub gater: Option<usize>,
    pub elegibility: f64,
    pub previous_delta_weight: f64,
    pub total_delta_weight: f64,
    pub xtrace: XTrace,
}

#[derive(Debug, Clone)]
pub struct XTrace {
    pub nodes: Vec<usize>,
    pub values: Vec<f64>,
}

impl Connection {
    pub fn new(from: Option<usize>, to: Option<usize>, weight: Option<f64>) -> Self {
        Connection {
            from,
            to,
            gain: 1.0,
            weight: weight.unwrap_or_else(|| rand::random::<f64>() * 0.2 - 0.1),
            gater: None,
            elegibility: 0.0,
            previous_delta_weight: 0.0,
            total_delta_weight: 0.0,
            xtrace: XTrace {
                nodes: vec![],
                values: vec![],
            },
        }
    }
    pub fn new_self() -> Self {
        Connection::new(None, None, Some(0.0))
    }
}
