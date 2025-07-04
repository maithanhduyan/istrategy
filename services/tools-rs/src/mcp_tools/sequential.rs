use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Represents a single thought in the sequential thinking process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtData {
    pub thought: String,
    #[serde(rename = "thoughtNumber")]
    pub thought_number: u32,
    #[serde(rename = "totalThoughts")]
    pub total_thoughts: u32,
    #[serde(rename = "nextThoughtNeeded")]
    pub next_thought_needed: bool,
    #[serde(rename = "isRevision")]
    pub is_revision: Option<bool>,
    #[serde(rename = "revisesThought")]
    pub revises_thought: Option<u32>,
    #[serde(rename = "branchFromThought")]
    pub branch_from_thought: Option<u32>,
    #[serde(rename = "branchId")]
    pub branch_id: Option<String>,
    #[serde(rename = "needsMoreThoughts")]
    pub needs_more_thoughts: Option<bool>,
}

/// Sequential Thinking tool input parameters
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SequentialThinkingParams {
    #[serde(rename = "thought")]
    pub thought: String,
    #[serde(rename = "thoughtNumber")]
    pub thought_number: u32,
    #[serde(rename = "totalThoughts")]
    pub total_thoughts: u32,
    #[serde(rename = "nextThoughtNeeded")]
    pub next_thought_needed: bool,
    #[serde(rename = "thinkingMethod")]
    pub thinking_method: String,
    #[serde(rename = "isRevision")]
    pub is_revision: Option<bool>,
    #[serde(rename = "revisesThought")]
    pub revises_thought: Option<u32>,
    #[serde(rename = "branchFromThought")]
    pub branch_from_thought: Option<u32>,
    #[serde(rename = "branchId")]
    pub branch_id: Option<String>,
    #[serde(rename = "needsMoreThoughts")]
    pub needs_more_thoughts: Option<bool>,
}

/// Internal state manager for sequential thinking
#[derive(Debug, Default)]
pub struct SequentialThinkingState {
    pub thought_history: Vec<ThoughtData>,
    pub branches: HashMap<String, Vec<ThoughtData>>,
}

/// Sequential Thinking tool implementation with state management
pub struct SequentialThinkingTool {
    state: Arc<Mutex<SequentialThinkingState>>,
}

impl SequentialThinkingTool {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SequentialThinkingState::default())),
        }
    }

    /// Validates thought data input parameters
    fn validate_thought_data(&self, params: &SequentialThinkingParams) -> McpResult<()> {
        let mut missing_fields = Vec::new();
        if params.thinking_method != "sequential" {
            return Err("Invalid thinkingMethod: must be 'sequential'".into());
        }
        if params.thought.trim().is_empty() {
            missing_fields.push("thought");
        }
        if params.thought_number < 1 {
            missing_fields.push("thoughtNumber");
        }
        if params.total_thoughts < 1 {
            missing_fields.push("totalThoughts");
        }
        // next_thought_needed is required, but bool so always present
        if !missing_fields.is_empty() {
            return Err(format!(
                "Missing or invalid required fields: {}",
                missing_fields.join(", ")
            )
            .into());
        }
        if let Some(revises_thought) = params.revises_thought {
            if revises_thought < 1 {
                return Err("Invalid revisesThought: must be at least 1".into());
            }
        }
        if let Some(branch_from_thought) = params.branch_from_thought {
            if branch_from_thought < 1 {
                return Err("Invalid branchFromThought: must be at least 1".into());
            }
        }
        Ok(())
    }

    /// Processes a thought and updates internal state
    pub fn process_thought(
        &self,
        params: SequentialThinkingParams,
    ) -> McpResult<serde_json::Value> {
        // Validate input
        self.validate_thought_data(&params)?;

        let mut total_thoughts = params.total_thoughts;
        if params.thought_number > total_thoughts {
            total_thoughts = params.thought_number;
        }

        // Create thought data
        let thought_data = ThoughtData {
            thought: params.thought.clone(),
            thought_number: params.thought_number,
            total_thoughts,
            next_thought_needed: params.next_thought_needed,
            is_revision: params.is_revision,
            revises_thought: params.revises_thought,
            branch_from_thought: params.branch_from_thought,
            branch_id: params.branch_id.clone(),
            needs_more_thoughts: params.needs_more_thoughts,
        };

        // Update state
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("State lock error: {}", e))?;
        state.thought_history.push(thought_data.clone());

        // Handle branching
        if let (Some(_branch_from_thought), Some(ref branch_id)) =
            (params.branch_from_thought, &params.branch_id)
        {
            state
                .branches
                .entry(branch_id.clone())
                .or_insert_with(Vec::new)
                .push(thought_data.clone());
        }

        // Prepare response similar to TypeScript version
        let response = serde_json::json!({
            "thoughtNumber": thought_data.thought_number,
            "totalThoughts": thought_data.total_thoughts,
            "nextThoughtNeeded": thought_data.next_thought_needed,
            "branches": state.branches.keys().collect::<Vec<_>>(),
            "thoughtHistoryLength": state.thought_history.len(),
            "isRevision": thought_data.is_revision.unwrap_or(false),
            "branchId": thought_data.branch_id,
            "needsMoreThoughts": thought_data.needs_more_thoughts,
        });

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": response.to_string()
            }]
        }))
    }
}

impl Default for SequentialThinkingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SequentialThinkingTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "sequentialthinking".to_string(),
            description: r#"A detailed tool for dynamic and reflective problem-solving through sequential thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out

Key features:
- You can adjust totalThoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts even after reaching what seemed like the end
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Generates a solution hypothesis
- Verifies the hypothesis based on the Chain of Thought steps
- Repeats the process until satisfied
- Provides a correct answer"#.to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "thought": { 
                        "type": "string", 
                        "description": "Your current thinking step" 
                    },
                    "nextThoughtNeeded": { 
                        "type": "boolean", 
                        "description": "Whether another thought step is needed" 
                    },
                    "thoughtNumber": { 
                        "type": "integer", 
                        "minimum": 1, 
                        "description": "Current thought number" 
                    },
                    "totalThoughts": { 
                        "type": "integer", 
                        "minimum": 1, 
                        "description": "Estimated total thoughts needed" 
                    },
                    "thinkingMethod": {
                        "type": "string",
                        "enum": ["sequential"],
                        "description": "Thinking method type"
                    },
                    "isRevision": { 
                        "type": "boolean", 
                        "description": "Whether this revises previous thinking" 
                    },
                    "revisesThought": { 
                        "type": "integer", 
                        "minimum": 1, 
                        "description": "Which thought is being reconsidered" 
                    },
                    "branchFromThought": { 
                        "type": "integer", 
                        "minimum": 1, 
                        "description": "Branching point thought number" 
                    },
                    "branchId": { 
                        "type": "string", 
                        "description": "Branch identifier" 
                    },
                    "needsMoreThoughts": { 
                        "type": "boolean", 
                        "description": "If more thoughts are needed" 
                    }
                },
                "required": ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts", "thinkingMethod"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let input: SequentialThinkingParams =
            serde_json::from_value(params).map_err(|e| format!("Invalid parameters: {}", e))?;

        self.process_thought(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_tool_definition() {
        let tool = SequentialThinkingTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "sequentialthinking");
        assert!(def.description.contains("sequential thoughts"));
    }

    #[test]
    fn test_sequential_tool_execute_valid() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "Bước 1: xác định mục tiêu",
            "thoughtNumber": 1,
            "totalThoughts": 3,
            "nextThoughtNeeded": true,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("\"thoughtNumber\":1"));
        assert!(text.contains("\"totalThoughts\":3"));
    }

    #[test]
    fn test_sequential_tool_with_branching() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "Nhánh thay thế",
            "thoughtNumber": 2,
            "totalThoughts": 5,
            "nextThoughtNeeded": true,
            "branchFromThought": 1,
            "branchId": "alternative-approach",
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("\"branchId\":\"alternative-approach\""));
        assert!(text.contains("\"branches\":[\"alternative-approach\"]"));
    }

    #[test]
    fn test_sequential_tool_with_revision() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "Chỉnh sửa lại bước 1",
            "thoughtNumber": 2,
            "totalThoughts": 3,
            "nextThoughtNeeded": true,
            "isRevision": true,
            "revisesThought": 1,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("\"isRevision\":true"));
    }

    #[test]
    fn test_sequential_tool_validation_empty_thought() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "",
            "thoughtNumber": 1,
            "totalThoughts": 1,
            "nextThoughtNeeded": false,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        // Chấp nhận thông báo mới: "Missing or invalid required fields: thought"
        assert!(error_msg.contains("thought"));
    }

    #[test]
    fn test_sequential_tool_validation_invalid_numbers() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "Valid thought",
            "thoughtNumber": 0,
            "totalThoughts": 1,
            "nextThoughtNeeded": false,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequential_tool_auto_adjust_total_thoughts() {
        let tool = SequentialThinkingTool::new();
        let params = serde_json::json!({
            "thought": "Bước 5 của 3",
            "thoughtNumber": 5,
            "totalThoughts": 3,
            "nextThoughtNeeded": false,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params).unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("\"totalThoughts\":5"));
    }

    #[test]
    fn test_sequential_tool_missing_required_fields() {
        let tool = SequentialThinkingTool::new();
        // Thiếu thought
        let params = serde_json::json!({
            "thought": "",
            "thoughtNumber": 1,
            "totalThoughts": 1,
            "nextThoughtNeeded": true,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("thought"));

        // Thiếu thoughtNumber
        let params = serde_json::json!({
            "thought": "abc",
            "thoughtNumber": 0,
            "totalThoughts": 1,
            "nextThoughtNeeded": true,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("thoughtNumber"));

        // Thiếu totalThoughts
        let params = serde_json::json!({
            "thought": "abc",
            "thoughtNumber": 1,
            "totalThoughts": 0,
            "nextThoughtNeeded": true,
            "thinkingMethod": "sequential"
        });
        let result = tool.execute(params);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("totalThoughts"));
    }
}
