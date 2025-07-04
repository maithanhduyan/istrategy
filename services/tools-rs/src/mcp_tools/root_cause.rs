use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Mutex;

/// Root Cause Analysis parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RootCauseAnalysisParams {
    pub problem_statement: String,
    pub symptoms: Vec<String>,
    pub immediate_causes: Vec<String>,
    pub root_causes: Vec<String>,
    pub contributing_factors: Vec<String>,
    pub evidence: Vec<String>,
    pub verification_methods: Vec<String>,
    pub preventive_actions: Vec<String>,
    pub corrective_actions: Vec<String>,
    pub next_analysis_needed: bool,
}

/// Root Cause Analysis response
#[derive(Debug, Serialize)]
pub struct RootCauseAnalysisResponse {
    pub analysis_complete: bool,
    pub root_causes_identified: usize,
    pub next_analysis_needed: bool,
    pub total_analyses: usize,
    pub preventive_actions_count: usize,
}

/// Root Cause Analysis tool implementation
pub struct RootCauseAnalysisTool {
    analyses: Mutex<Vec<RootCauseAnalysisParams>>,
    disable_thought_logging: bool,
}

impl RootCauseAnalysisTool {
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

    fn validate_params(&self, params: &RootCauseAnalysisParams) -> McpResult<()> {
        if params.problem_statement.is_empty() {
            return Err("Invalid problem_statement: must not be empty".into());
        }
        if params.symptoms.is_empty() {
            return Err("Invalid symptoms: must have at least one symptom".into());
        }
        if params.root_causes.is_empty() {
            return Err("Invalid root_causes: must have at least one root cause".into());
        }
        Ok(())
    }

    fn format_root_cause_analysis(&self, data: &RootCauseAnalysisParams) -> String {
        format!(
            "🔍 Root Cause Analysis\n\
             ┌─────────────────────────────────────────┐\n\
             │ Problem: {}\n\
             │ Symptoms: {}\n\
             │ Immediate Causes: {}\n\
             │ Root Causes: {}\n\
             │ Contributing Factors: {}\n\
             │ Evidence: {}\n\
             │ Verification: {}\n\
             │ Preventive Actions: {}\n\
             │ Corrective Actions: {}\n\
             └─────────────────────────────────────────┘",
            data.problem_statement,
            data.symptoms.join(", "),
            data.immediate_causes.join(", "),
            data.root_causes.join(", "),
            data.contributing_factors.join(", "),
            data.evidence.join(", "),
            data.verification_methods.join(", "),
            data.preventive_actions.join(", "),
            data.corrective_actions.join(", ")
        )
    }
}

impl Default for RootCauseAnalysisTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for RootCauseAnalysisTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "root_cause_analysis".to_string(),
            description: "A tool for systematic root cause analysis to identify underlying causes and preventive actions".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "problem_statement": { "type": "string", "description": "Clear statement of the problem" },
                    "symptoms": { "type": "array", "items": { "type": "string" }, "description": "Observable symptoms" },
                    "immediate_causes": { "type": "array", "items": { "type": "string" }, "description": "Direct causes of symptoms" },
                    "root_causes": { "type": "array", "items": { "type": "string" }, "description": "Fundamental root causes" },
                    "contributing_factors": { "type": "array", "items": { "type": "string" }, "description": "Factors that contribute to the problem" },
                    "evidence": { "type": "array", "items": { "type": "string" }, "description": "Evidence supporting the analysis" },
                    "verification_methods": { "type": "array", "items": { "type": "string" }, "description": "Methods to verify root causes" },
                    "preventive_actions": { "type": "array", "items": { "type": "string" }, "description": "Actions to prevent recurrence" },
                    "corrective_actions": { "type": "array", "items": { "type": "string" }, "description": "Actions to fix current issues" },
                    "next_analysis_needed": { "type": "boolean", "description": "Whether further analysis is needed" }
                },
                "required": ["problem_statement", "symptoms", "immediate_causes", "root_causes", "contributing_factors", "evidence", "verification_methods", "preventive_actions", "corrective_actions", "next_analysis_needed"]
            }),
        }
    }

    fn execute(&self, params: serde_json::Value) -> McpResult<serde_json::Value> {
        let analysis_params: RootCauseAnalysisParams = serde_json::from_value(params)?;

        self.validate_params(&analysis_params)?;

        if !self.disable_thought_logging {
            let formatted_analysis = self.format_root_cause_analysis(&analysis_params);
            eprintln!("{}", formatted_analysis);
        }

        let mut analyses = self.analyses.lock().unwrap();
        analyses.push(analysis_params.clone());
        let total_analyses = analyses.len();
        drop(analyses);

        let response = RootCauseAnalysisResponse {
            analysis_complete: true,
            root_causes_identified: analysis_params.root_causes.len(),
            next_analysis_needed: analysis_params.next_analysis_needed,
            total_analyses,
            preventive_actions_count: analysis_params.preventive_actions.len(),
        };

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&response)?
            }]
        }))
    }
}
