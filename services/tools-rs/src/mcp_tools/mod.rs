// Built-in tools module (moved from src/tools/mod.rs)
pub mod calculator;
pub mod critical;
pub mod echo;
pub mod lateral;
pub mod memory;
pub mod root_cause;
pub mod sequential;
pub mod six_hats;
pub mod systems;

pub use calculator::CalculatorTool;
pub use critical::CriticalThinkingTool;
pub use echo::EchoTool;
pub use lateral::LateralThinkingTool;
pub use memory::{
    MemoryAddObservationsTool, MemoryCreateEntitesTool, MemoryCreateRelationsTool,
    MemoryDeleteEntitiesTool, MemoryDeleteObservationsTool, MemoryDeleteRelationsTool,
    MemoryManagementTool, MemoryOpenNodesTool, MemoryReadGraphTool, MemorySearchNodesTool,
};
pub use root_cause::RootCauseAnalysisTool;
pub use sequential::SequentialThinkingTool;
pub use six_hats::SixThinkingHatsTool;
pub use systems::SystemsThinkingTool;

use crate::mcp_core::{McpResult, McpTool};

/// Trait for implementing MCP tools
pub trait Tool: Send + Sync {
    /// Returns the tool definition
    fn definition(&self) -> McpTool;
    /// Executes the tool with given parameters
    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value>;
}
