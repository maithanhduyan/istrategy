use crate::architecture::connection::Connection;
use crate::methods::activation::activation::logistic;
use rand;

#[derive(Debug, Clone)]
pub struct Node {
    pub bias: f64,
    pub squash: fn(f64, bool) -> f64,
    pub node_type: String,
    pub activation: f64,
    pub state: f64,
    pub old: f64,
    pub mask: f64,
    pub previous_delta_bias: f64,
    pub total_delta_bias: f64,
    pub connections: NodeConnections,
    pub error: NodeError,
    pub derivative: Option<f64>,
    pub index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct NodeConnections {
    pub in_conns: Vec<Connection>,
    pub out_conns: Vec<Connection>,
    pub gated: Vec<Connection>,
    pub self_conn: Connection,
}

#[derive(Debug, Clone)]
pub struct NodeError {
    pub responsibility: f64,
    pub projected: f64,
    pub gated: f64,
}

impl Node {
    /// Creates a new node
    pub fn new(node_type: Option<&str>) -> Self {
        let bias = if node_type == Some("input") {
            0.0
        } else {
            rand::random::<f64>() * 0.2 - 0.1
        };
        let squash = logistic;
        let node_type = node_type.unwrap_or("hidden").to_string();
        let self_conn = Connection::new_self();
        Node {
            bias,
            squash,
            node_type,
            activation: 0.0,
            state: 0.0,
            old: 0.0,
            mask: 1.0,
            previous_delta_bias: 0.0,
            total_delta_bias: 0.0,
            connections: NodeConnections {
                in_conns: vec![],
                out_conns: vec![],
                gated: vec![],
                self_conn,
            },
            error: NodeError {
                responsibility: 0.0,
                projected: 0.0,
                gated: 0.0,
            },
            derivative: None,
            index: None,
        }
    }
    // TODO: Implement activate, no_trace_activate, propagate, connect, disconnect, gate, ungate, clear, mutate, is_projecting_to, is_projected_by, to_json, from_json
}
