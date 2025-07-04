use thinking_tools::mcp_tools::{
    CalculatorTool, CriticalThinkingTool, EchoTool, LateralThinkingTool, MemoryAddObservationsTool,
    MemoryCreateEntitesTool, MemoryCreateRelationsTool, MemoryDeleteEntitiesTool,
    MemoryDeleteObservationsTool, MemoryDeleteRelationsTool, MemoryManagementTool,
    MemoryOpenNodesTool, MemoryReadGraphTool, MemorySearchNodesTool, RootCauseAnalysisTool,
    SequentialThinkingTool, SixThinkingHatsTool, SystemsThinkingTool,
};
use thinking_tools::prelude::*;
use thinking_tools::ServerInfo;

fn main() -> McpResult<()> {
    let server_info = ServerInfo {
        name: "thinking-tool-server".to_string(),
        version: "1.0.0".to_string(),
        description: Some("A comprehensive thinking tools MCP server".to_string()),
    };
    let mut server = McpServer::with_info(server_info);

    // Register all thinking tools (18 total)
    server.register_tool(Box::new(EchoTool::new())); // 1
    server.register_tool(Box::new(CalculatorTool::new())); // 2
    server.register_tool(Box::new(SequentialThinkingTool::new())); // 3
    server.register_tool(Box::new(CriticalThinkingTool::new())); // 4
    server.register_tool(Box::new(SystemsThinkingTool::new())); // 5
    server.register_tool(Box::new(LateralThinkingTool::new())); // 6
    server.register_tool(Box::new(RootCauseAnalysisTool::new())); // 7
    server.register_tool(Box::new(SixThinkingHatsTool::new())); // 8
    server.register_tool(Box::new(MemoryManagementTool::new())); // 9
    server.register_tool(Box::new(MemoryCreateEntitesTool::new())); // 10
    server.register_tool(Box::new(MemoryCreateRelationsTool::new())); // 11
    server.register_tool(Box::new(MemoryAddObservationsTool::new())); // 12
    server.register_tool(Box::new(MemoryDeleteEntitiesTool::new())); // 13
    server.register_tool(Box::new(MemoryDeleteObservationsTool::new())); // 14
    server.register_tool(Box::new(MemoryDeleteRelationsTool::new())); // 15
    server.register_tool(Box::new(MemoryReadGraphTool::new())); // 16
    server.register_tool(Box::new(MemorySearchNodesTool::new())); // 17
    server.register_tool(Box::new(MemoryOpenNodesTool::new())); // 18

    server.run()
}
