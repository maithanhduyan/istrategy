use crate::mcp_core::{McpResult, McpTool};
use super::Tool;
use serde::Deserialize;

#[derive(Deserialize)]
struct EchoParams {
    message: String,
}

pub struct EchoTool;

impl EchoTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EchoTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "echo".to_string(),
            description: "Echo back the input message".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let echo_params: EchoParams =
            serde_json::from_value(params).map_err(|e| format!("Invalid parameters: {}", e))?;
        let result = serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": echo_params.message
                }
            ]
        });
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_echo_tool_definition() {
        let tool = EchoTool::new();
        let definition = tool.definition();
        assert_eq!(definition.name, "echo");
        assert_eq!(definition.description, "Echo back the input message");
    }
    #[test]
    fn test_echo_tool_execute() {
        let tool = EchoTool::new();
        let params = serde_json::json!({"message": "Hello, Test!"});
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text_content = content[0].get("text").unwrap().as_str().unwrap();
        assert_eq!(text_content, "Hello, Test!");
    }
    #[test]
    fn test_echo_tool_invalid_params() {
        let tool = EchoTool::new();
        let params = serde_json::json!({"invalid": "parameter"});
        let result = tool.execute(params);
        assert!(result.is_err());
    }
}
