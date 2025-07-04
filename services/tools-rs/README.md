# MCP Tools Library

A reusable Rust library for creating Model Context Protocol (MCP) servers with JSON-RPC 2.0 support.

## Features

- **JSON-RPC 2.0 compliant**: Full support for JSON-RPC 2.0 specification
- **MCP Protocol**: Complete implementation of MCP protocol for tool servers
- **Extensible**: Easy to add new tools with the `Tool` trait
- **Type-safe**: Leverages Rust's type system for safety and performance
- **Built-in tools**: Includes common tools like Echo and Calculator
- **Comprehensive testing**: Full test coverage for all components

## Architecture

```
src/
├── lib.rs          # Library entry point and re-exports
├── mcp.rs          # Core MCP protocol implementation
├── tools/
│   ├── mod.rs      # Tools module
│   ├── echo.rs     # Echo tool implementation
│   └── calculator.rs # Calculator tool implementation
├── echo_lib.rs     # Single-tool server example
└── multi_tool.rs   # Multi-tool server example
```

## Usage

### Creating a Simple Server

```rust
use tools_rs::prelude::*;

fn main() -> McpResult<()> {
    let mut server = McpServer::new();
    server.register_tool(Box::new(EchoTool::new()));
    server.run()
}
```

### Creating a Multi-Tool Server

```rust
use tools_rs::prelude::*;

fn main() -> McpResult<()> {
    let server_info = tools_rs::mcp::ServerInfo {
        name: "my-mcp-server".to_string(),
        version: "1.0.0".to_string(),
        description: Some("My custom MCP server".to_string()),
    };

    let mut server = McpServer::with_info(server_info);
    
    server
        .register_tool(Box::new(EchoTool::new()))
        .register_tool(Box::new(CalculatorTool::new()));
    
    server.run()
}
```

### Implementing Custom Tools

```rust
use tools_rs::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct MyToolParams {
    input: String,
}

pub struct MyTool;

impl Tool for MyTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "my_tool".to_string(),
            description: "My custom tool".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let my_params: MyToolParams = serde_json::from_value(params)?;
        
        Ok(serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": format!("Processed: {}", my_params.input)
                }
            ]
        }))
    }
}
```

## Built-in Tools

### Echo Tool
- **Name**: `echo`
- **Description**: Echo back the input message
- **Parameters**: `message` (string)

### Calculator Tool
- **Name**: `calculator`
- **Description**: Perform basic arithmetic operations
- **Parameters**: 
  - `operation` (string): add, subtract, multiply, divide
  - `a` (number): First number
  - `b` (number): Second number

## Available Binaries

- `echo_lib`: Single echo tool server
- `multi_tool`: Multi-tool server with echo and calculator
- `echo_mcp`: Standalone echo server (legacy)

## Testing

Run all tests:
```bash
cargo test --lib
```

## Configuration

Add to `.vscode/mcp.json`:
```json
{
    "servers": {
        "multi-tool-server": {
            "type": "stdio",
            "command": "${workspaceFolder}/services/tools-rs/target/debug/multi_tool.exe",
            "args": []
        }
    }
}
```

## Standards Compliance

- ✅ JSON-RPC 2.0 specification
- ✅ MCP Protocol 2024-11-05
- ✅ Rust coding standards
- ✅ Comprehensive error handling
- ✅ Full documentation
- ✅ Unit testing coverage
