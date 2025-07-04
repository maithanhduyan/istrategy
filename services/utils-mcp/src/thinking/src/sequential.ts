import { Tool } from "@modelcontextprotocol/sdk/types.js";

// Sequential Thinking interface
export interface SequentialThinkingData {
  thought: string;
  stepNumber: number;
  totalSteps: number;
  nextStepNeeded: boolean;
  thinkingMethod: 'sequential';
  isRevision?: boolean;
  revisesThought?: number;
  branchFromThought?: number;
  branchId?: string;
  needsMoreThoughts?: boolean;
}

// Sequential Thinking Server Class
export class SequentialThinkingServer {
  private sequentialHistory: SequentialThinkingData[] = [];
  private disableThoughtLogging: boolean;

  constructor() {
    this.disableThoughtLogging = (process.env.DISABLE_THOUGHT_LOGGING || "").toLowerCase() === "true";
  }

  private validateSequentialData(input: unknown): SequentialThinkingData {
    const data = input as Record<string, unknown>;

    if (!data.thought || typeof data.thought !== 'string') {
      throw new Error('Invalid thought: must be a string');
    }
    if (!data.stepNumber || typeof data.stepNumber !== 'number') {
      throw new Error('Invalid stepNumber: must be a number');
    }
    if (!data.totalSteps || typeof data.totalSteps !== 'number') {
      throw new Error('Invalid totalSteps: must be a number');
    }
    if (typeof data.nextStepNeeded !== 'boolean') {
      throw new Error('Invalid nextStepNeeded: must be a boolean');
    }
    if (data.thinkingMethod !== 'sequential') {
      throw new Error('Invalid thinkingMethod: must be sequential');
    }

    return {
      thought: data.thought as string,
      stepNumber: data.stepNumber as number,
      totalSteps: data.totalSteps as number,
      nextStepNeeded: data.nextStepNeeded as boolean,
      thinkingMethod: 'sequential',
      isRevision: data.isRevision as boolean | undefined,
      revisesThought: data.revisesThought as number | undefined,
      branchFromThought: data.branchFromThought as number | undefined,
      branchId: data.branchId as string | undefined,
      needsMoreThoughts: data.needsMoreThoughts as boolean | undefined
    };
  }

  public processSequentialThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateSequentialData(input);

      if (validatedInput.stepNumber > validatedInput.totalSteps) {
        validatedInput.totalSteps = validatedInput.stepNumber;
      }

      this.sequentialHistory.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const { stepNumber, totalSteps, thought } = validatedInput;
        console.error(`💭 SEQUENTIAL Thinking ${stepNumber}/${totalSteps}: ${thought}`);
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            thoughtProcessed: true,
            stepNumber: validatedInput.stepNumber,
            totalSteps: validatedInput.totalSteps,
            nextStepNeeded: validatedInput.nextStepNeeded,
            isRevision: validatedInput.isRevision || false,
            branchId: validatedInput.branchId,
            totalThoughts: this.sequentialHistory.length
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

  public getHistory(): SequentialThinkingData[] {
    return [...this.sequentialHistory];
  }

  public clear(): void {
    this.sequentialHistory = [];
  }
}

// Tool definition for Sequential Thinking
export const SEQUENTIAL_TOOL: Tool = {
  name: "sequentialthinking",
  description: "A detailed tool for dynamic and reflective problem-solving through sequential thoughts",
  inputSchema: {
    type: "object",
    properties: {
      thought: { type: "string", description: "Your current thinking step" },
      nextStepNeeded: { type: "boolean", description: "Whether another step is needed" },
      stepNumber: { type: "number", minimum: 1, description: "Current step number" },
      totalSteps: { type: "number", minimum: 1, description: "Estimated total steps needed" },
      thinkingMethod: { type: "string", enum: ["sequential"], description: "Thinking method type" },
      isRevision: { type: "boolean", description: "Whether this revises previous thinking" },
      revisesThought: { type: "number", minimum: 1, description: "Which thought is being reconsidered" },
      branchFromThought: { type: "number", minimum: 1, description: "Branching point thought number" },
      branchId: { type: "string", description: "Branch identifier" },
      needsMoreThoughts: { type: "boolean", description: "If more thoughts are needed" }
    },
    required: ["thought", "nextStepNeeded", "stepNumber", "totalSteps", "thinkingMethod"]
  }
};
