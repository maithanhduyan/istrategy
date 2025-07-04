use crate::mcp_core::{McpResult, McpTool};
use super::Tool;
use serde::Deserialize;

#[derive(Deserialize)]
struct CalculatorParams {
    operation: String,
    a: f64,
    b: f64,
}

pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CalculatorTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "calculator".to_string(),
            description: "Perform basic arithmetic operations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform (add, subtract, multiply, divide)",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let calc_params: CalculatorParams =
            serde_json::from_value(params).map_err(|e| format!("Invalid parameters: {}", e))?;
        let result = match calc_params.operation.as_str() {
            "add" => calc_params.a + calc_params.b,
            "subtract" => calc_params.a - calc_params.b,
            "multiply" => calc_params.a * calc_params.b,
            "divide" => {
                if calc_params.b == 0.0 {
                    return Err("Division by zero is not allowed".into());
                }
                calc_params.a / calc_params.b
            }
            _ => return Err("Invalid operation. Use: add, subtract, multiply, or divide".into()),
        };
        let operation_symbol = match calc_params.operation.as_str() {
            "add" => "+",
            "subtract" => "-",
            "multiply" => "×",
            "divide" => "÷",
            _ => "?",
        };
        let response = serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": format!("{} {} {} = {}", calc_params.a, operation_symbol, calc_params.b, result)
                }
            ]
        });
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calculator_tool_definition() {
        let tool = CalculatorTool::new();
        let definition = tool.definition();
        assert_eq!(definition.name, "calculator");
        assert_eq!(
            definition.description,
            "Perform basic arithmetic operations"
        );
    }
    #[test]
    fn test_calculator_add() {
        let tool = CalculatorTool::new();
        let params = serde_json::json!({
            "operation": "add",
            "a": 5.0,
            "b": 3.0
        });
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text_content = content[0].get("text").unwrap().as_str().unwrap();
        assert_eq!(text_content, "5 + 3 = 8");
    }
    #[test]
    fn test_calculator_divide_by_zero() {
        let tool = CalculatorTool::new();
        let params = serde_json::json!({
            "operation": "divide",
            "a": 5.0,
            "b": 0.0
        });
        let result = tool.execute(params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Division by zero"));
    }
    #[test]
    fn test_calculator_invalid_operation() {
        let tool = CalculatorTool::new();
        let params = serde_json::json!({
            "operation": "invalid",
            "a": 5.0,
            "b": 3.0
        });
        let result = tool.execute(params);
        assert!(result.is_err());
    }
}
