use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Mutex;

/// System Component definition
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemComponent {
    pub name: String,
    pub component_type: String, // 'input' | 'process' | 'output' | 'feedback' | 'environment'
    pub description: String,
    pub relationships: Vec<String>,
}

/// Systems Analysis parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemsAnalysisParams {
    pub system_name: String,
    pub purpose: String,
    pub components: Vec<SystemComponent>,
    pub feedback_loops: Vec<String>,
    pub constraints: Vec<String>,
    pub emergent_properties: Vec<String>,
    pub leverage_points: Vec<String>,
    pub systemic_issues: Vec<String>,
    pub interventions: Vec<String>,
    pub next_analysis_needed: bool,
}

/// Systems Analysis response
#[derive(Debug, Serialize)]
pub struct SystemsAnalysisResponse {
    pub analysis_complete: bool,
    pub next_analysis_needed: bool,
    pub total_analyses: usize,
    pub system_name: String,
    pub leverage_points_count: usize,
}

/// Systems Thinking tool implementation
pub struct SystemsThinkingTool {
    analyses: Mutex<Vec<SystemsAnalysisParams>>,
    disable_thought_logging: bool,
}

impl SystemsThinkingTool {
    pub fn new() -> Self {
        let disable_thought_logging = env::var("DISABLE_THOUGHT_LOGGING")
            .unwrap_or_default()
            .to_lowercase()
            == "true";

        Self {
            analyses: Mutex::new(Vec::new()),
            disable_thought_logging,
        }
    }

    fn validate_params(&self, params: &SystemsAnalysisParams) -> McpResult<()> {
        if params.system_name.is_empty() {
            return Err("Invalid systemName: must not be empty".into());
        }
        if params.purpose.is_empty() {
            return Err("Invalid purpose: must not be empty".into());
        }
        // Validate component types
        for component in &params.components {
            match component.component_type.as_str() {
                "input" | "process" | "output" | "feedback" | "environment" => {},
                _ => return Err("Invalid component type: must be input, process, output, feedback, or environment".into()),
            }
        }
        Ok(())
    }

    fn format_systems_analysis(&self, data: &SystemsAnalysisParams) -> String {
        let components_summary = data
            .components
            .iter()
            .map(|c| format!("{}: {}", c.name, c.component_type))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "🔄 Systems Thinking Analysis\n\
             ┌─────────────────────────────────────────┐\n\
             │ System: {}\n\
             │ Purpose: {}\n\
             │ Components: {}\n\
             │ Feedback Loops: {}\n\
             │ Constraints: {}\n\
             │ Emergent Props: {}\n\
             │ Leverage Points: {}\n\
             │ Issues: {}\n\
             │ Interventions: {}\n\
             └─────────────────────────────────────────┘",
            data.system_name,
            data.purpose,
            components_summary,
            data.feedback_loops.join(", "),
            data.constraints.join(", "),
            data.emergent_properties.join(", "),
            data.leverage_points.join(", "),
            data.systemic_issues.join(", "),
            data.interventions.join(", ")
        )
    }
}

impl Default for SystemsThinkingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SystemsThinkingTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "systems_thinking".to_string(),
            description: "A tool for analyzing complex systems, identifying leverage points and systemic issues".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "system_name": { "type": "string", "description": "Name of the system being analyzed" },
                    "purpose": { "type": "string", "description": "The purpose or goal of the system" },
                    "components": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "component_type": { "type": "string", "enum": ["input", "process", "output", "feedback", "environment"] },
                                "description": { "type": "string" },
                                "relationships": { "type": "array", "items": { "type": "string" } }
                            },
                            "required": ["name", "component_type", "description", "relationships"]
                        },
                        "description": "System components"
                    },
                    "feedback_loops": { "type": "array", "items": { "type": "string" }, "description": "Feedback loops in the system" },
                    "constraints": { "type": "array", "items": { "type": "string" }, "description": "System constraints" },
                    "emergent_properties": { "type": "array", "items": { "type": "string" }, "description": "Emergent properties" },
                    "leverage_points": { "type": "array", "items": { "type": "string" }, "description": "Points of maximum impact" },
                    "systemic_issues": { "type": "array", "items": { "type": "string" }, "description": "System-wide problems" },
                    "interventions": { "type": "array", "items": { "type": "string" }, "description": "Proposed interventions" },
                    "next_analysis_needed": { "type": "boolean", "description": "Whether further analysis is needed" }
                },
                "required": ["system_name", "purpose", "components", "feedback_loops", "constraints", "emergent_properties", "leverage_points", "systemic_issues", "interventions", "next_analysis_needed"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let analysis_params: SystemsAnalysisParams = serde_json::from_value(params)?;

        self.validate_params(&analysis_params)?;

        if !self.disable_thought_logging {
            let formatted_analysis = self.format_systems_analysis(&analysis_params);
            eprintln!("{}", formatted_analysis);
        }

        let mut analyses = self.analyses.lock().unwrap();
        analyses.push(analysis_params.clone());
        let total_analyses = analyses.len();
        drop(analyses);

        let response = SystemsAnalysisResponse {
            analysis_complete: true,
            next_analysis_needed: analysis_params.next_analysis_needed,
            total_analyses,
            system_name: analysis_params.system_name,
            leverage_points_count: analysis_params.leverage_points.len(),
        };

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&response)?
            }]
        }))
    }
}
