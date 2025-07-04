use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Mutex;

/// Six Thinking Hats colors
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum HatColor {
    White,  // Facts and information
    Red,    // Emotions and feelings
    Black,  // Critical judgment
    Yellow, // Positive assessment
    Green,  // Creativity and alternatives
    Blue,   // Process control
}

impl HatColor {
    fn description(&self) -> &'static str {
        match self {
            HatColor::White => "Facts and Information",
            HatColor::Red => "Emotions and Feelings",
            HatColor::Black => "Critical Judgment",
            HatColor::Yellow => "Positive Assessment",
            HatColor::Green => "Creativity and Alternatives",
            HatColor::Blue => "Process Control",
        }
    }

    fn emoji(&self) -> &'static str {
        match self {
            HatColor::White => "⚪",
            HatColor::Red => "🔴",
            HatColor::Black => "⚫",
            HatColor::Yellow => "🟡",
            HatColor::Green => "🟢",
            HatColor::Blue => "🔵",
        }
    }
}

/// Six Hats Thinking parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SixHatsParams {
    pub hat_color: HatColor,
    pub perspective: String,
    pub insights: Vec<String>,
    pub questions: Vec<String>,
    pub next_hat_needed: bool,
    pub session_complete: bool,
}

/// Six Hats Thinking response
#[derive(Debug, Serialize)]
pub struct SixHatsResponse {
    pub hat_processed: String,
    pub insights_count: usize,
    pub next_hat_needed: bool,
    pub session_complete: bool,
    pub total_hats_used: usize,
}

/// Six Thinking Hats tool implementation
pub struct SixThinkingHatsTool {
    sessions: Mutex<Vec<SixHatsParams>>,
    hats_used: Mutex<Vec<String>>,
    disable_thought_logging: bool,
}

impl SixThinkingHatsTool {
    pub fn new() -> Self {
        let disable_thought_logging = env::var("DISABLE_THOUGHT_LOGGING")
            .unwrap_or_default()
            .to_lowercase()
            == "true";

        Self {
            sessions: Mutex::new(Vec::new()),
            hats_used: Mutex::new(Vec::new()),
            disable_thought_logging,
        }
    }

    fn validate_params(&self, params: &SixHatsParams) -> McpResult<()> {
        if params.perspective.is_empty() {
            return Err("Invalid perspective: must not be empty".into());
        }
        if params.insights.is_empty() {
            return Err("Invalid insights: must have at least one insight".into());
        }
        Ok(())
    }

    fn format_six_hats_thinking(&self, data: &SixHatsParams) -> String {
        let emoji = data.hat_color.emoji();
        let description = data.hat_color.description();
        let hat_name = format!("{:?}", data.hat_color);

        format!(
            "{} {} Hat - {}\n\
             ┌─────────────────────────────────────────┐\n\
             │ Perspective: {}\n\
             │ Insights: {}\n\
             │ Questions: {}\n\
             │ Session Complete: {}\n\
             └─────────────────────────────────────────┘",
            emoji,
            hat_name,
            description,
            data.perspective,
            data.insights.join(", "),
            data.questions.join(", "),
            data.session_complete
        )
    }
}

impl Default for SixThinkingHatsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SixThinkingHatsTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "six_thinking_hats".to_string(),
            description:
                "A tool for structured thinking using Edward de Bono's Six Thinking Hats method"
                    .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "hat_color": {
                        "type": "string",
                        "enum": ["white", "red", "black", "yellow", "green", "blue"],
                        "description": "The thinking hat color to use"
                    },
                    "perspective": { "type": "string", "description": "The perspective from this hat's viewpoint" },
                    "insights": { "type": "array", "items": { "type": "string" }, "description": "Key insights from this perspective" },
                    "questions": { "type": "array", "items": { "type": "string" }, "description": "Questions raised from this perspective" },
                    "next_hat_needed": { "type": "boolean", "description": "Whether another hat perspective is needed" },
                    "session_complete": { "type": "boolean", "description": "Whether the thinking session is complete" }
                },
                "required": ["hat_color", "perspective", "insights", "questions", "next_hat_needed", "session_complete"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let hat_params: SixHatsParams = serde_json::from_value(params)?;

        self.validate_params(&hat_params)?;

        if !self.disable_thought_logging {
            let formatted_thinking = self.format_six_hats_thinking(&hat_params);
            eprintln!("{}", formatted_thinking);
        }

        let hat_name = format!("{:?}", hat_params.hat_color).to_lowercase();

        let mut hats_used = self.hats_used.lock().unwrap();
        hats_used.push(hat_name.clone());
        let total_hats_used = hats_used.len();
        drop(hats_used);

        let mut sessions = self.sessions.lock().unwrap();
        sessions.push(hat_params.clone());
        drop(sessions);

        let response = SixHatsResponse {
            hat_processed: hat_name,
            insights_count: hat_params.insights.len(),
            next_hat_needed: hat_params.next_hat_needed,
            session_complete: hat_params.session_complete,
            total_hats_used,
        };

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&response)?
            }]
        }))
    }
}
