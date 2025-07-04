/// MCP (Model Context Protocol) Library
/// A reusable library for creating MCP servers with JSON-RPC 2.0 support
pub mod mcp_core;
pub mod mcp_tools;

pub use mcp_core::*;
pub use mcp_tools::*;

/// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::mcp_core::{
        ErrorObject, JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpResult, McpServer, McpTool,
    };
    pub use crate::mcp_tools::{CalculatorTool, EchoTool, Tool};
}
