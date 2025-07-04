use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Mutex;

/// Critical Analysis data structure
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CriticalAnalysisParams {
    pub claim: String,
    pub evidence: Vec<String>,
    pub assumptions: Vec<String>,
    pub counterarguments: Vec<String>,
    pub logical_fallacies: Vec<String>,
    pub credibility_assessment: String,
    pub conclusion: String,
    pub confidence_level: u8,
    pub next_analysis_needed: bool,
}

/// Critical Analysis response
#[derive(Debug, Serialize)]
pub struct CriticalAnalysisResponse {
    pub analysis_complete: bool,
    pub confidence_level: u8,
    pub next_analysis_needed: bool,
    pub total_analyses: usize,
    pub conclusion: String,
}

/// Critical Thinking tool implementation
pub struct CriticalThinkingTool {
    analyses: Mutex<Vec<CriticalAnalysisParams>>,
    disable_thought_logging: bool,
}

impl CriticalThinkingTool {
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

    fn validate_params(&self, params: &CriticalAnalysisParams) -> McpResult<()> {
        if params.claim.is_empty() {
            return Err("Invalid claim: must not be empty".into());
        }
        if params.credibility_assessment.is_empty() {
            return Err("Invalid credibilityAssessment: must not be empty".into());
        }
        if params.conclusion.is_empty() {
            return Err("Invalid conclusion: must not be empty".into());
        }
        if params.confidence_level > 100 {
            return Err("Invalid confidenceLevel: must be between 0-100".into());
        }
        Ok(())
    }

    fn format_critical_analysis(&self, data: &CriticalAnalysisParams) -> String {
        let confidence_indicator = match data.confidence_level {
            80..=100 => "🟢",
            60..=79 => "🟡",
            _ => "🔴",
        };

        format!(
            "🔍 Critical Analysis\n\
             ┌─────────────────────────────────────────┐\n\
             │ Claim: {}\n\
             │ Evidence: {}\n\
             │ Assumptions: {}\n\
             │ Counter-args: {}\n\
             │ Fallacies: {}\n\
             │ Credibility: {}\n\
             │ Confidence: {} {}%\n\
             │ Conclusion: {}\n\
             └─────────────────────────────────────────┘",
            data.claim,
            data.evidence.join(", "),
            data.assumptions.join(", "),
            data.counterarguments.join(", "),
            data.logical_fallacies.join(", "),
            data.credibility_assessment,
            confidence_indicator,
            data.confidence_level,
            data.conclusion
        )
    }
}

impl Default for CriticalThinkingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CriticalThinkingTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "critical_thinking".to_string(),
            description: "A tool for structured critical analysis and evaluation of claims"
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "claim": { "type": "string", "description": "The main claim or statement to analyze" },
                    "evidence": { "type": "array", "items": { "type": "string" }, "description": "Supporting evidence for the claim" },
                    "assumptions": { "type": "array", "items": { "type": "string" }, "description": "Underlying assumptions" },
                    "counterarguments": { "type": "array", "items": { "type": "string" }, "description": "Arguments against the claim" },
                    "logical_fallacies": { "type": "array", "items": { "type": "string" }, "description": "Identified logical fallacies" },
                    "credibility_assessment": { "type": "string", "description": "Assessment of source credibility" },
                    "conclusion": { "type": "string", "description": "Final conclusion" },
                    "confidence_level": { "type": "number", "minimum": 0, "maximum": 100, "description": "Confidence level (0-100)" },
                    "next_analysis_needed": { "type": "boolean", "description": "Whether further analysis is needed" }
                },
                "required": ["claim", "evidence", "assumptions", "counterarguments", "logical_fallacies", "credibility_assessment", "conclusion", "confidence_level", "next_analysis_needed"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let analysis_params: CriticalAnalysisParams = serde_json::from_value(params)?;

        self.validate_params(&analysis_params)?;

        if !self.disable_thought_logging {
            let formatted_analysis = self.format_critical_analysis(&analysis_params);
            eprintln!("{}", formatted_analysis);
        }

        let mut analyses = self.analyses.lock().unwrap();
        analyses.push(analysis_params.clone());
        let total_analyses = analyses.len();
        drop(analyses);

        let response = CriticalAnalysisResponse {
            analysis_complete: true,
            confidence_level: analysis_params.confidence_level,
            next_analysis_needed: analysis_params.next_analysis_needed,
            total_analyses,
            conclusion: analysis_params.conclusion,
        };

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&response)?
            }]
        }))
    }
}
