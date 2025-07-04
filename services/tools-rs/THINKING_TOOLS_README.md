# Thinking Tools MCP Server

A comprehensive MCP (Model Context Protocol) server implementing various thinking methodologies to enhance problem-solving, decision-making, and creative thinking.

## Overview

This Rust-based MCP server provides 8 powerful thinking tools based on proven methodologies:

### 🧠 Thinking Tools Available

1. **Sequential Thinking** - Step-by-step problem-solving with branching and revision capabilities
2. **Critical Thinking** - Systematic analysis of claims, evidence, and logical reasoning  
3. **Systems Thinking** - Holistic analysis of complex systems and their interactions
4. **Lateral Thinking** - Creative problem-solving using unconventional approaches
5. **Root Cause Analysis** - Deep investigation to identify fundamental causes
6. **Six Thinking Hats** - Structured thinking using Edward de Bono's methodology
7. **Memory Management** - Persistent storage and retrieval of insights and patterns
8. **Echo Tool** - Simple utility for testing and validation

## Features

- **Thread-Safe**: All tools use `Mutex` for safe concurrent access
- **Configurable Logging**: Environment variable control for thought logging
- **Rich Validation**: Comprehensive input validation for all parameters
- **Structured Output**: Consistent JSON responses with detailed feedback
- **Memory Persistence**: Store and retrieve thinking patterns across sessions

## Installation

```bash
git clone <repository>
cd tools-rs
cargo build --release
```

## Usage

### Starting the Server

```bash
cargo run
```

The server will start and listen for MCP connections.

### Tool Descriptions

#### Sequential Thinking
```json
{
  "name": "sequentialthinking",
  "description": "Step-by-step problem-solving with revision capabilities",
  "required": ["thought", "next_step_needed", "step_number", "total_steps", "thinking_method"]
}
```

#### Critical Thinking
```json
{
  "name": "critical_thinking", 
  "description": "Systematic analysis of claims and evidence",
  "required": ["claim", "evidence", "assumptions", "counterarguments", "logical_fallacies", "credibility_assessment", "conclusion", "confidence_level", "next_analysis_needed"]
}
```

#### Systems Thinking
```json
{
  "name": "systems_thinking",
  "description": "Holistic analysis of complex systems",
  "required": ["system_name", "purpose", "components", "feedback_loops", "constraints", "emergent_properties", "leverage_points", "systemic_issues", "interventions", "next_analysis_needed"]
}
```

#### Lateral Thinking
```json
{
  "name": "lateral_thinking",
  "description": "Creative problem-solving using various techniques",
  "required": ["technique", "stimulus", "connection", "idea", "evaluation", "next_technique_needed"],
  "techniques": ["random_word", "provocation", "alternative", "reversal", "metaphor", "assumption_challenge"]
}
```

#### Root Cause Analysis
```json
{
  "name": "root_cause_analysis",
  "description": "Systematic identification of fundamental causes",
  "required": ["problem_statement", "symptoms", "immediate_causes", "root_causes", "contributing_factors", "evidence", "verification_methods", "preventive_actions", "corrective_actions", "next_analysis_needed"]
}
```

#### Six Thinking Hats
```json
{
  "name": "six_thinking_hats",
  "description": "Structured thinking using colored hat perspectives",
  "required": ["hat_color", "perspective", "insights", "questions", "next_hat_needed", "session_complete"],
  "hat_colors": ["white", "red", "black", "yellow", "green", "blue"]
}
```

#### Memory Management
```json
{
  "name": "memory_management",
  "description": "Persistent storage and retrieval of insights",
  "required": ["operation"],
  "operations": ["store", "retrieve", "search", "update", "delete", "list_by_tags", "get_high_importance"]
}
```

## Configuration

### Environment Variables

- `DISABLE_THOUGHT_LOGGING=true` - Disable console output of thinking processes

## Development

### Running Tests
```bash
cargo test
```

### Building Documentation
```bash
cargo doc --open
```

### Code Structure

```
src/
├── lib.rs              # Library entry point
├── main.rs             # MCP server entry point
├── mcp_core/           # Core MCP implementation
│   ├── mod.rs
│   ├── server.rs
│   └── types.rs
└── mcp_tools/          # Thinking tools implementation
    ├── mod.rs
    ├── sequential.rs   # Sequential thinking
    ├── critical.rs     # Critical thinking  
    ├── systems.rs      # Systems thinking
    ├── lateral.rs      # Lateral thinking
    ├── root_cause.rs   # Root cause analysis
    ├── six_hats.rs     # Six thinking hats
    ├── memory.rs       # Memory management
    ├── echo.rs         # Echo utility
    └── calculator.rs   # Calculator utility
```

## Examples

### Sequential Thinking Example
```json
{
  "thought": "Analyzing the customer retention problem",
  "step_number": 1,
  "total_steps": 5, 
  "next_step_needed": true,
  "thinking_method": "sequential"
}
```

### Critical Thinking Example
```json
{
  "claim": "Remote work increases productivity",
  "evidence": ["Study shows 20% productivity increase", "Employee surveys indicate higher satisfaction"],
  "assumptions": ["Productivity can be measured accurately", "Home environment is conducive to work"],
  "counterarguments": ["Collaboration may decrease", "Management oversight is reduced"],
  "logical_fallacies": [],
  "credibility_assessment": "Mixed evidence from reputable sources",
  "conclusion": "Remote work can increase productivity under right conditions",
  "confidence_level": 75,
  "next_analysis_needed": false
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `cargo test`
5. Submit a pull request

## License

See LICENSE file for details.
