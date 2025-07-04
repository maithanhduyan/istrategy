import chalk from 'chalk';
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export interface CriticalAnalysis {
  claim: string;
  evidence: string[];
  assumptions: string[];
  counterarguments: string[];
  logicalFallacies: string[];
  credibilityAssessment: string;
  conclusion: string;
  confidenceLevel: number;
  nextAnalysisNeeded: boolean;
}

export class CriticalThinkingServer {
  private analyses: CriticalAnalysis[] = [];

  private validateCriticalData(input: unknown): CriticalAnalysis {
    const data = input as Record<string, unknown>;
    
    if (!data.claim || typeof data.claim !== 'string') {
      throw new Error('Invalid claim: must be a string');
    }
    if (!Array.isArray(data.evidence)) {
      throw new Error('Invalid evidence: must be an array');
    }
    if (!Array.isArray(data.assumptions)) {
      throw new Error('Invalid assumptions: must be an array');
    }
    if (!Array.isArray(data.counterarguments)) {
      throw new Error('Invalid counterarguments: must be an array');
    }
    if (!Array.isArray(data.logicalFallacies)) {
      throw new Error('Invalid logicalFallacies: must be an array');
    }
    if (!data.credibilityAssessment || typeof data.credibilityAssessment !== 'string') {
      throw new Error('Invalid credibilityAssessment: must be a string');
    }
    if (!data.conclusion || typeof data.conclusion !== 'string') {
      throw new Error('Invalid conclusion: must be a string');
    }
    if (typeof data.confidenceLevel !== 'number' || data.confidenceLevel < 0 || data.confidenceLevel > 100) {
      throw new Error('Invalid confidenceLevel: must be a number between 0-100');
    }
    if (typeof data.nextAnalysisNeeded !== 'boolean') {
      throw new Error('Invalid nextAnalysisNeeded: must be a boolean');
    }

    return {
      claim: data.claim as string,
      evidence: data.evidence as string[],
      assumptions: data.assumptions as string[],
      counterarguments: data.counterarguments as string[],
      logicalFallacies: data.logicalFallacies as string[],
      credibilityAssessment: data.credibilityAssessment as string,
      conclusion: data.conclusion as string,
      confidenceLevel: data.confidenceLevel as number,
      nextAnalysisNeeded: data.nextAnalysisNeeded as boolean
    };
  }

  private formatCriticalAnalysis(data: CriticalAnalysis): string {
    const header = chalk.red('🔍 Critical Analysis');
    const confidenceColor = data.confidenceLevel >= 80 ? 'green' : 
                           data.confidenceLevel >= 60 ? 'yellow' : 'red';
    
    return `
┌─────────────────────────────────────────┐
│ ${header} │
├─────────────────────────────────────────┤
│ Claim: ${data.claim} │
│ Evidence: ${data.evidence.join(', ')} │
│ Assumptions: ${data.assumptions.join(', ')} │
│ Counter-args: ${data.counterarguments.join(', ')} │
│ Fallacies: ${data.logicalFallacies.join(', ')} │
│ Credibility: ${data.credibilityAssessment} │
│ Confidence: ${chalk[confidenceColor](data.confidenceLevel + '%')} │
│ Conclusion: ${data.conclusion} │
└─────────────────────────────────────────┘`;
  }

  public processCriticalAnalysis(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateCriticalData(input);
      
      this.analyses.push(validatedInput);

      const formattedAnalysis = this.formatCriticalAnalysis(validatedInput);
      console.error(formattedAnalysis);

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            analysisComplete: true,
            confidenceLevel: validatedInput.confidenceLevel,
            nextAnalysisNeeded: validatedInput.nextAnalysisNeeded,
            totalAnalyses: this.analyses.length,
            conclusion: validatedInput.conclusion
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

const CRITICAL_THINKING_TOOL: Tool = {
  name: "criticalthinking",
  description: `A tool for systematic critical analysis and evaluation of claims, arguments, and information.
Helps identify logical fallacies, assess evidence quality, and reach well-reasoned conclusions.

Key components:
- Claim identification and clarification
- Evidence evaluation
- Assumption analysis
- Counter-argument consideration
- Logical fallacy detection
- Source credibility assessment
- Confidence level estimation

When to use:
- Evaluating arguments or claims
- Fact-checking and verification
- Decision-making with uncertain information
- Detecting bias and logical errors
- Academic or research analysis
- Debate preparation
- News and media analysis`,
  inputSchema: {
    type: "object",
    properties: {
      claim: {
        type: "string",
        description: "The main claim or argument being analyzed"
      },
      evidence: {
        type: "array",
        items: { type: "string" },
        description: "Evidence supporting the claim"
      },
      assumptions: {
        type: "array",
        items: { type: "string" },
        description: "Underlying assumptions identified"
      },
      counterarguments: {
        type: "array",
        items: { type: "string" },
        description: "Arguments against the claim"
      },
      logicalFallacies: {
        type: "array",
        items: { type: "string" },
        description: "Logical fallacies identified"
      },
      credibilityAssessment: {
        type: "string",
        description: "Assessment of source credibility"
      },
      conclusion: {
        type: "string",
        description: "Final reasoned conclusion"
      },
      confidenceLevel: {
        type: "number",
        minimum: 0,
        maximum: 100,
        description: "Confidence level in conclusion (0-100%)"
      },
      nextAnalysisNeeded: {
        type: "boolean",
        description: "Whether further analysis is needed"
      }
    },
    required: ["claim", "evidence", "assumptions", "counterarguments", "logicalFallacies", "credibilityAssessment", "conclusion", "confidenceLevel", "nextAnalysisNeeded"]
  }
};

export { CRITICAL_THINKING_TOOL };