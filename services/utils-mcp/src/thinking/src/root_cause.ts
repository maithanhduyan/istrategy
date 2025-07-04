/**
 * Root Cause Analysis (RCA) is a systematic process for identifying the underlying causes of problems or events.
 * It aims to address the root cause rather than just treating the symptoms, leading to more effective and sustainable solutions.
 * Common techniques include the 5 Whys, Fishbone Diagram, and Fault Tree Analysis.
 * RCA is widely used in various fields such as engineering, healthcare, and business to improve processes and prevent recurrence of issues.
 */

import chalk from 'chalk';
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export interface RootCauseAnalysis {
  problemStatement: string;
  technique: '5_whys' | 'fishbone' | 'fault_tree' | 'timeline' | 'barrier_analysis';
  symptoms: string[];
  immediateActions: string[];
  rootCauses: string[];
  contributingFactors: string[];
  preventiveActions: string[];
  verification: string[];
  nextAnalysisNeeded: boolean;
}

export class RootCauseServer {
  private analyses: RootCauseAnalysis[] = [];

  private validateRootCauseData(input: unknown): RootCauseAnalysis {
    const data = input as Record<string, unknown>;
    
    if (!data.problemStatement || typeof data.problemStatement !== 'string') {
      throw new Error('Invalid problemStatement: must be a string');
    }
    if (!data.technique || typeof data.technique !== 'string') {
      throw new Error('Invalid technique: must be a string');
    }
    if (!Array.isArray(data.symptoms)) {
      throw new Error('Invalid symptoms: must be an array');
    }
    if (!Array.isArray(data.immediateActions)) {
      throw new Error('Invalid immediateActions: must be an array');
    }
    if (!Array.isArray(data.rootCauses)) {
      throw new Error('Invalid rootCauses: must be an array');
    }
    if (!Array.isArray(data.contributingFactors)) {
      throw new Error('Invalid contributingFactors: must be an array');
    }
    if (!Array.isArray(data.preventiveActions)) {
      throw new Error('Invalid preventiveActions: must be an array');
    }
    if (!Array.isArray(data.verification)) {
      throw new Error('Invalid verification: must be an array');
    }
    if (typeof data.nextAnalysisNeeded !== 'boolean') {
      throw new Error('Invalid nextAnalysisNeeded: must be a boolean');
    }

    return {
      problemStatement: data.problemStatement as string,
      technique: data.technique as RootCauseAnalysis['technique'],
      symptoms: data.symptoms as string[],
      immediateActions: data.immediateActions as string[],
      rootCauses: data.rootCauses as string[],
      contributingFactors: data.contributingFactors as string[],
      preventiveActions: data.preventiveActions as string[],
      verification: data.verification as string[],
      nextAnalysisNeeded: data.nextAnalysisNeeded as boolean
    };
  }

  private formatRootCauseAnalysis(data: RootCauseAnalysis): string {
    const techniqueEmojis: Record<string, string> = {
      '5_whys': '❓',
      'fishbone': '🐟',
      'fault_tree': '🌳',
      'timeline': '📅',
      'barrier_analysis': '🚧'
    };

    const emoji = techniqueEmojis[data.technique] || '🔍';
    const header = chalk.yellow(`${emoji} Root Cause Analysis - ${data.technique.toUpperCase()}`);
    
    return `
┌─────────────────────────────────────────┐
│ ${header} │
├─────────────────────────────────────────┤
│ Problem: ${data.problemStatement} │
│ Symptoms: ${data.symptoms.length} identified │
│ Root Causes: ${data.rootCauses.length} found │
│ Preventive Actions: ${data.preventiveActions.length} │
│ Verification Steps: ${data.verification.length} │
└─────────────────────────────────────────┘`;
  }

  public processRootCauseAnalysis(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateRootCauseData(input);
      
      this.analyses.push(validatedInput);

      const formattedAnalysis = this.formatRootCauseAnalysis(validatedInput);
      console.error(formattedAnalysis);

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            analysisComplete: true,
            technique: validatedInput.technique,
            rootCausesCount: validatedInput.rootCauses.length,
            preventiveActionsCount: validatedInput.preventiveActions.length,
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

const ROOT_CAUSE_TOOL: Tool = {
  name: "rootcauseanalysis",
  description: `A systematic tool for identifying root causes of problems using various RCA techniques.

Techniques available:
- 5_whys: Ask "why" repeatedly to drill down to root causes
- fishbone: Ishikawa diagram to categorize potential causes
- fault_tree: Top-down deductive failure analysis
- timeline: Chronological analysis of events leading to problem
- barrier_analysis: Analyze what barriers failed to prevent the problem

When to use:
- Problem solving and troubleshooting
- Quality improvement initiatives
- Incident investigation
- Process improvement
- Failure analysis
- Risk assessment`,
  inputSchema: {
    type: "object",
    properties: {
      problemStatement: {
        type: "string",
        description: "Clear description of the problem to analyze"
      },
      technique: {
        type: "string",
        enum: ["5_whys", "fishbone", "fault_tree", "timeline", "barrier_analysis"],
        description: "RCA technique to use"
      },
      symptoms: {
        type: "array",
        items: { type: "string" },
        description: "Observable symptoms of the problem"
      },
      immediateActions: {
        type: "array",
        items: { type: "string" },
        description: "Immediate actions taken to contain the problem"
      },
      rootCauses: {
        type: "array",
        items: { type: "string" },
        description: "Identified root causes"
      },
      contributingFactors: {
        type: "array",
        items: { type: "string" },
        description: "Factors that contributed to the problem"
      },
      preventiveActions: {
        type: "array",
        items: { type: "string" },
        description: "Actions to prevent recurrence"
      },
      verification: {
        type: "array",
        items: { type: "string" },
        description: "Steps to verify the root cause and effectiveness of solutions"
      },
      nextAnalysisNeeded: {
        type: "boolean",
        description: "Whether additional analysis is needed"
      }
    },
    required: ["problemStatement", "technique", "symptoms", "immediateActions", "rootCauses", "contributingFactors", "preventiveActions", "verification", "nextAnalysisNeeded"]
  }
};

export { ROOT_CAUSE_TOOL };