import chalk from 'chalk';
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export interface SystemComponent {
  name: string;
  type: 'input' | 'process' | 'output' | 'feedback' | 'environment';
  description: string;
  relationships: string[];
}

export interface SystemsAnalysis {
  systemName: string;
  purpose: string;
  components: SystemComponent[];
  feedbackLoops: string[];
  constraints: string[];
  emergentProperties: string[];
  leverage_points: string[];
  systemicIssues: string[];
  interventions: string[];
  nextAnalysisNeeded: boolean;
}

export class SystemsThinkingServer {
  private analyses: SystemsAnalysis[] = [];

  private validateSystemsData(input: unknown): SystemsAnalysis {
    const data = input as Record<string, unknown>;
    
    if (!data.systemName || typeof data.systemName !== 'string') {
      throw new Error('Invalid systemName: must be a string');
    }
    if (!data.purpose || typeof data.purpose !== 'string') {
      throw new Error('Invalid purpose: must be a string');
    }
    if (!Array.isArray(data.components)) {
      throw new Error('Invalid components: must be an array');
    }
    if (!Array.isArray(data.feedbackLoops)) {
      throw new Error('Invalid feedbackLoops: must be an array');
    }
    if (!Array.isArray(data.constraints)) {
      throw new Error('Invalid constraints: must be an array');
    }
    if (!Array.isArray(data.emergentProperties)) {
      throw new Error('Invalid emergentProperties: must be an array');
    }
    if (!Array.isArray(data.leverage_points)) {
      throw new Error('Invalid leverage_points: must be an array');
    }
    if (!Array.isArray(data.systemicIssues)) {
      throw new Error('Invalid systemicIssues: must be an array');
    }
    if (!Array.isArray(data.interventions)) {
      throw new Error('Invalid interventions: must be an array');
    }
    if (typeof data.nextAnalysisNeeded !== 'boolean') {
      throw new Error('Invalid nextAnalysisNeeded: must be a boolean');
    }

    return {
      systemName: data.systemName as string,
      purpose: data.purpose as string,
      components: data.components as SystemComponent[],
      feedbackLoops: data.feedbackLoops as string[],
      constraints: data.constraints as string[],
      emergentProperties: data.emergentProperties as string[],
      leverage_points: data.leverage_points as string[],
      systemicIssues: data.systemicIssues as string[],
      interventions: data.interventions as string[],
      nextAnalysisNeeded: data.nextAnalysisNeeded as boolean
    };
  }

  private formatSystemsAnalysis(data: SystemsAnalysis): string {
    const header = chalk.cyan('🔄 Systems Thinking Analysis');
    
    return `
┌─────────────────────────────────────────┐
│ ${header} │
├─────────────────────────────────────────┤
│ System: ${data.systemName} │
│ Purpose: ${data.purpose} │
│ Components: ${data.components.length} identified │
│ Feedback Loops: ${data.feedbackLoops.length} │
│ Constraints: ${data.constraints.length} │
│ Emergent Properties: ${data.emergentProperties.length} │
│ Leverage Points: ${data.leverage_points.length} │
│ Issues: ${data.systemicIssues.length} │
│ Interventions: ${data.interventions.length} │
└─────────────────────────────────────────┘`;
  }

  public processSystemsAnalysis(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateSystemsData(input);
      
      this.analyses.push(validatedInput);

      const formattedAnalysis = this.formatSystemsAnalysis(validatedInput);
      console.error(formattedAnalysis);

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            systemAnalyzed: validatedInput.systemName,
            componentsCount: validatedInput.components.length,
            leveragePoints: validatedInput.leverage_points,
            nextAnalysisNeeded: validatedInput.nextAnalysisNeeded,
            totalAnalyses: this.analyses.length
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }
}

const SYSTEMS_THINKING_TOOL: Tool = {
  name: "systemsthinking",
  description: `A tool for holistic analysis of complex systems, identifying relationships, patterns, and leverage points.
Helps understand how components interact and influence the whole system.

Key concepts:
- System components and their relationships
- Feedback loops (reinforcing and balancing)
- Constraints and bottlenecks
- Emergent properties
- Leverage points for intervention
- Systemic issues vs symptoms
- Root cause analysis

When to use:
- Complex organizational problems
- Process improvement
- Understanding interconnected issues
- Strategic planning
- Root cause analysis
- Change management
- Policy design
- Business ecosystem analysis`,
  inputSchema: {
    type: "object",
    properties: {
      systemName: {
        type: "string",
        description: "Name of the system being analyzed"
      },
      purpose: {
        type: "string",
        description: "Main purpose or function of the system"
      },
      components: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string" },
            type: { 
              type: "string",
              enum: ["input", "process", "output", "feedback", "environment"]
            },
            description: { type: "string" },
            relationships: {
              type: "array",
              items: { type: "string" }
            }
          },
          required: ["name", "type", "description", "relationships"]
        },
        description: "System components and their relationships"
      },
      feedbackLoops: {
        type: "array",
        items: { type: "string" },
        description: "Feedback loops identified in the system"
      },
      constraints: {
        type: "array",
        items: { type: "string" },
        description: "Constraints limiting system performance"
      },
      emergentProperties: {
        type: "array",
        items: { type: "string" },
        description: "Properties that emerge from system interactions"
      },
      leverage_points: {
        type: "array",
        items: { type: "string" },
        description: "High-impact intervention points"
      },
      systemicIssues: {
        type: "array",
        items: { type: "string" },
        description: "Systemic issues vs surface symptoms"
      },
      interventions: {
        type: "array",
        items: { type: "string" },
        description: "Proposed system interventions"
      },
      nextAnalysisNeeded: {
        type: "boolean",
        description: "Whether deeper analysis is needed"
      }
    },
    required: ["systemName", "purpose", "components", "feedbackLoops", "constraints", "emergentProperties", "leverage_points", "systemicIssues", "interventions", "nextAnalysisNeeded"]
  }
};

export { SYSTEMS_THINKING_TOOL };