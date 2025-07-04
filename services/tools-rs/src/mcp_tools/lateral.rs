use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Mutex;

/// Lateral Thinking technique types
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum LateralTechnique {
    RandomWord,
    Provocation,
    Alternative,
    Reversal,
    Metaphor,
    AssumptionChallenge,
}

impl LateralTechnique {
    fn emoji(&self) -> &'static str {
        match self {
            LateralTechnique::RandomWord => "🎲",
            LateralTechnique::Provocation => "🚀",
            LateralTechnique::Alternative => "🔄",
            LateralTechnique::Reversal => "↩️",
            LateralTechnique::Metaphor => "🎭",
            LateralTechnique::AssumptionChallenge => "❓",
        }
    }
}

/// Lateral Thinking parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LateralThoughtParams {
    pub technique: LateralTechnique,
    pub stimulus: String,
    pub connection: String,
    pub idea: String,
    pub evaluation: String,
    pub next_technique_needed: bool,
}

/// Lateral Thinking response
#[derive(Debug, Serialize)]
pub struct LateralThoughtResponse {
    pub idea_generated: bool,
    pub technique_used: String,
    pub next_technique_needed: bool,
    pub total_ideas: usize,
    pub evaluation: String,
}

/// Lateral Thinking tool implementation
pub struct LateralThinkingTool {
    techniques_used: Mutex<Vec<String>>,
    ideas: Mutex<Vec<LateralThoughtParams>>,
    disable_thought_logging: bool,
}

impl LateralThinkingTool {
    pub fn new() -> Self {
        let disable_thought_logging = env::var("DISABLE_THOUGHT_LOGGING")
            .unwrap_or_default()
            .to_lowercase()
            == "true";

        Self {
            techniques_used: Mutex::new(Vec::new()),
            ideas: Mutex::new(Vec::new()),
            disable_thought_logging,
        }
    }

    fn validate_params(&self, params: &LateralThoughtParams) -> McpResult<()> {
        if params.stimulus.is_empty() {
            return Err("Invalid stimulus: must not be empty".into());
        }
        if params.connection.is_empty() {
            return Err("Invalid connection: must not be empty".into());
        }
        if params.idea.is_empty() {
            return Err("Invalid idea: must not be empty".into());
        }
        if params.evaluation.is_empty() {
            return Err("Invalid evaluation: must not be empty".into());
        }
        Ok(())
    }

    fn format_lateral_thought(&self, data: &LateralThoughtParams) -> String {
        let emoji = data.technique.emoji();
        let technique_name = format!("{:?}", data.technique).to_uppercase();

        format!(
            "{} Lateral Thinking: {}\n\
             ┌─────────────────────────────────────────┐\n\
             │ Stimulus: {}\n\
             │ Connection: {}\n\
             │ Idea: {}\n\
             │ Evaluation: {}\n\
             └─────────────────────────────────────────┘",
            emoji, technique_name, data.stimulus, data.connection, data.idea, data.evaluation
        )
    }
}

impl Default for LateralThinkingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for LateralThinkingTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "lateral_thinking".to_string(),
            description:
                "A tool for creative problem-solving using various lateral thinking techniques"
                    .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "technique": {
                        "type": "string",
                        "enum": ["random_word", "provocation", "alternative", "reversal", "metaphor", "assumption_challenge"],
                        "description": "The lateral thinking technique to use"
                    },
                    "stimulus": { "type": "string", "description": "The stimulus or starting point" },
                    "connection": { "type": "string", "description": "How the stimulus connects to the problem" },
                    "idea": { "type": "string", "description": "The generated idea" },
                    "evaluation": { "type": "string", "description": "Evaluation of the idea" },
                    "next_technique_needed": { "type": "boolean", "description": "Whether another technique is needed" }
                },
                "required": ["technique", "stimulus", "connection", "idea", "evaluation", "next_technique_needed"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let thought_params: LateralThoughtParams = serde_json::from_value(params)?;

        self.validate_params(&thought_params)?;

        if !self.disable_thought_logging {
            let formatted_thought = self.format_lateral_thought(&thought_params);
            eprintln!("{}", formatted_thought);
        }

        let technique_name = format!("{:?}", thought_params.technique).to_lowercase();

        let mut techniques_used = self.techniques_used.lock().unwrap();
        techniques_used.push(technique_name.clone());
        drop(techniques_used);

        let mut ideas = self.ideas.lock().unwrap();
        ideas.push(thought_params.clone());
        let total_ideas = ideas.len();
        drop(ideas);

        let response = LateralThoughtResponse {
            idea_generated: true,
            technique_used: technique_name,
            next_technique_needed: thought_params.next_technique_needed,
            total_ideas,
            evaluation: thought_params.evaluation,
        };

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&response)?
            }]
        }))
    }
}
