
/**
Các dạng tư duy chính:

1. Linear Thinking (Tư duy tuyến tính)
Tư duy theo trình tự logic từ A → B → C
Phù hợp cho các vấn đề có bước rõ ràng

2. Lateral Thinking (Tư duy bên)
Tìm giải pháp sáng tạo, phi truyền thống
Edward de Bono's Six Thinking Hats
Brainstorming, tạo ý tưởng mới

3. Critical Thinking (Tư duy phản biện)
Phân tích, đánh giá thông tin
Tìm lỗ hổng trong lập luận
Kiểm tra tính hợp lý

4. Systems Thinking (Tư duy hệ thống)
Nhìn nhận toàn cục, mối quan hệ
Hiểu các thành phần tương tác
Root cause analysis

5. Dialectical Thinking (Tư duy biện chứng)
Thesis → Antithesis → Synthesis
Xem xét mâu thuẫn để tìm giải pháp

6. Parallel Thinking (Tư duy song song)
Six Thinking Hats method
Mỗi người cùng góc nhìn

7. Divergent vs Convergent Thinking
Divergent: Tạo nhiều ý tưởng
Convergent: Thu hẹp về giải pháp tối ưu

8. Analogical Thinking (Tư duy so sánh)
Sử dụng phép tương tự
Học từ trường hợp tương tự

9. Inductive vs Deductive Thinking
Inductive: Từ cụ thể → tổng quát
Deductive: Từ tổng quát → cụ thể

10. Design Thinking
Empathize → Define → Ideate → Prototype → Test
Tập trung vào người dùng

*/ 


import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import chalk from 'chalk';

// Base interface for all thinking methods
interface BaseThinkingData {
  thought: string;
  stepNumber: number;
  totalSteps: number;
  nextStepNeeded: boolean;
  thinkingMethod: 'sequential' | 'lateral' | 'critical' | 'systems' | 'design';
}

interface SequentialThinkingData extends BaseThinkingData {
  thinkingMethod: 'sequential';
  isRevision?: boolean;
  revisesThought?: number;
  branchFromThought?: number;
  branchId?: string;
  needsMoreThoughts?: boolean;
}

interface LateralThinkingData extends BaseThinkingData {
  thinkingMethod: 'lateral';
  technique: 'random_word' | 'provocation' | 'alternative' | 'reversal' | 'metaphor' | 'assumption_challenge';
  stimulus: string;
  connection: string;
  idea: string;
  evaluation: string;
}

interface CriticalThinkingData extends BaseThinkingData {
  thinkingMethod: 'critical';
  claim: string;
  evidence: string[];
  assumptions: string[];
  counterarguments: string[];
  logicalFallacies: string[];
  credibilityAssessment: string;
  conclusion: string;
  confidenceLevel: number;
}

interface SystemsThinkingData extends BaseThinkingData {
  thinkingMethod: 'systems';
  systemName: string;
  purpose: string;
  components: Array<{
    name: string;
    type: 'input' | 'process' | 'output' | 'feedback' | 'environment';
    description: string;
    relationships: string[];
  }>;
  feedbackLoops: string[];
  constraints: string[];
  emergentProperties: string[];
  leveragePoints: string[];
  systemicIssues: string[];
  interventions: string[];
}

interface DesignThinkingData extends BaseThinkingData {
  thinkingMethod: 'design';
  phase: 'empathize' | 'define' | 'ideate' | 'prototype' | 'test';
  userInsights: string[];
  problemStatement: string;
  solutions: string[];
  prototypes: string[];
  testResults: string[];
  iterations: string[];
}

class ThinkingServer {
  private sequentialHistory: SequentialThinkingData[] = [];
  private lateralIdeas: LateralThinkingData[] = [];
  private criticalAnalyses: CriticalThinkingData[] = [];
  private systemsAnalyses: SystemsThinkingData[] = [];
  private designProcesses: DesignThinkingData[] = [];
  private disableThoughtLogging: boolean;

  constructor() {
    this.disableThoughtLogging = (process.env.DISABLE_THOUGHT_LOGGING || "").toLowerCase() === "true";
  }

  private validateThoughtData(input: unknown, method: string): BaseThinkingData {
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
    if (data.thinkingMethod !== method) {
      throw new Error(`Invalid thinkingMethod: must be ${method}`);
    }

    return {
      thought: data.thought as string,
      stepNumber: data.stepNumber as number,
      totalSteps: data.totalSteps as number,
      nextStepNeeded: data.nextStepNeeded as boolean,
      thinkingMethod: data.thinkingMethod as BaseThinkingData['thinkingMethod']
    };
  }

  private formatThought(thoughtData: BaseThinkingData): string {
    const methodEmojis: Record<string, string> = {
      'sequential': '💭',
      'lateral': '🎲',
      'critical': '🔍',
      'systems': '🔄',
      'design': '🎨'
    };

    const { stepNumber, totalSteps, thought, thinkingMethod } = thoughtData;
    const emoji = methodEmojis[thinkingMethod] || '💡';
    const methodName = String(thinkingMethod).toUpperCase();
    
    let coloredHeader = '';
    switch (thinkingMethod) {
      case 'sequential':
        coloredHeader = chalk.blue(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
        break;
      case 'lateral':
        coloredHeader = chalk.magenta(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
        break;
      case 'critical':
        coloredHeader = chalk.red(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
        break;
      case 'systems':
        coloredHeader = chalk.cyan(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
        break;
      case 'design':
        coloredHeader = chalk.green(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
        break;
      default:
        coloredHeader = chalk.white(`${emoji} ${methodName} Thinking ${stepNumber}/${totalSteps}`);
    }
    
    const border = '─'.repeat(Math.max(coloredHeader.length, thought.length) + 4);

    return `
┌${border}┐
│ ${coloredHeader} │
├${border}┤
│ ${thought.padEnd(border.length - 2)} │
└${border}┘`;
  }

  public processSequentialThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const baseData = this.validateThoughtData(input, 'sequential');
      const validatedInput = input as SequentialThinkingData;

      if (validatedInput.stepNumber > validatedInput.totalSteps) {
        validatedInput.totalSteps = validatedInput.stepNumber;
      }

      this.sequentialHistory.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const formattedThought = this.formatThought(validatedInput);
        console.error(formattedThought);
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

  public processLateralThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const baseData = this.validateThoughtData(input, 'lateral');
      const validatedInput = input as LateralThinkingData;

      this.lateralIdeas.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const formattedThought = this.formatThought(validatedInput);
        console.error(formattedThought);
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            ideaGenerated: true,
            technique: validatedInput.technique,
            nextStepNeeded: validatedInput.nextStepNeeded,
            totalIdeas: this.lateralIdeas.length
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

  public processCriticalThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const baseData = this.validateThoughtData(input, 'critical');
      const validatedInput = input as CriticalThinkingData;

      this.criticalAnalyses.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const formattedThought = this.formatThought(validatedInput);
        console.error(formattedThought);
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            analysisComplete: true,
            confidenceLevel: validatedInput.confidenceLevel,
            nextStepNeeded: validatedInput.nextStepNeeded,
            totalAnalyses: this.criticalAnalyses.length
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

  public processSystemsThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const baseData = this.validateThoughtData(input, 'systems');
      const validatedInput = input as SystemsThinkingData;

      this.systemsAnalyses.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const formattedThought = this.formatThought(validatedInput);
        console.error(formattedThought);
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            systemAnalyzed: validatedInput.systemName,
            componentsCount: validatedInput.components.length,
            leveragePoints: validatedInput.leveragePoints,
            nextStepNeeded: validatedInput.nextStepNeeded,
            totalAnalyses: this.systemsAnalyses.length
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

  public processDesignThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const baseData = this.validateThoughtData(input, 'design');
      const validatedInput = input as DesignThinkingData;

      this.designProcesses.push(validatedInput);

      if (!this.disableThoughtLogging) {
        const formattedThought = this.formatThought(validatedInput);
        console.error(formattedThought);
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            phaseComplete: validatedInput.phase,
            nextStepNeeded: validatedInput.nextStepNeeded,
            totalProcesses: this.designProcesses.length
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

// Define all thinking tools
const SEQUENTIAL_THINKING_TOOL: Tool = {
  name: "mcp_sequentialthinking_sequentialthinking",
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

const LATERAL_THINKING_TOOL: Tool = {
  name: "mcp_lateralthinking_lateralthinking",
  description: "A tool for creative problem-solving using lateral thinking techniques",
  inputSchema: {
    type: "object",
    properties: {
      thought: { type: "string", description: "Your current thinking step" },
      nextStepNeeded: { type: "boolean", description: "Whether another step is needed" },
      stepNumber: { type: "number", minimum: 1, description: "Current step number" },
      totalSteps: { type: "number", minimum: 1, description: "Estimated total steps needed" },
      thinkingMethod: { type: "string", enum: ["lateral"], description: "Thinking method type" },
      technique: {
        type: "string",
        enum: ["random_word", "provocation", "alternative", "reversal", "metaphor", "assumption_challenge"],
        description: "Lateral thinking technique to use"
      },
      stimulus: { type: "string", description: "The stimulus or prompt used" },
      connection: { type: "string", description: "How the stimulus connects to the problem" },
      idea: { type: "string", description: "The creative idea generated" },
      evaluation: { type: "string", description: "Brief evaluation of the idea's potential" }
    },
    required: ["thought", "nextStepNeeded", "stepNumber", "totalSteps", "thinkingMethod", "technique", "stimulus", "connection", "idea", "evaluation"]
  }
};

const CRITICAL_THINKING_TOOL: Tool = {
  name: "mcp_criticalthinking_criticalthinking", 
  description: "A tool for systematic critical analysis and evaluation",
  inputSchema: {
    type: "object",
    properties: {
      thought: { type: "string", description: "Your current thinking step" },
      nextStepNeeded: { type: "boolean", description: "Whether another step is needed" },
      stepNumber: { type: "number", minimum: 1, description: "Current step number" },
      totalSteps: { type: "number", minimum: 1, description: "Estimated total steps needed" },
      thinkingMethod: { type: "string", enum: ["critical"], description: "Thinking method type" },
      claim: { type: "string", description: "The main claim being analyzed" },
      evidence: { type: "array", items: { type: "string" }, description: "Evidence supporting the claim" },
      assumptions: { type: "array", items: { type: "string" }, description: "Underlying assumptions" },
      counterarguments: { type: "array", items: { type: "string" }, description: "Arguments against the claim" },
      logicalFallacies: { type: "array", items: { type: "string" }, description: "Logical fallacies identified" },
      credibilityAssessment: { type: "string", description: "Assessment of source credibility" },
      conclusion: { type: "string", description: "Final reasoned conclusion" },
      confidenceLevel: { type: "number", minimum: 0, maximum: 100, description: "Confidence level (0-100%)" }
    },
    required: ["thought", "nextStepNeeded", "stepNumber", "totalSteps", "thinkingMethod", "claim", "evidence", "assumptions", "counterarguments", "logicalFallacies", "credibilityAssessment", "conclusion", "confidenceLevel"]
  }
};

const SYSTEMS_THINKING_TOOL: Tool = {
  name: "mcp_systemsthinking_systemsthinking",
  description: "A tool for holistic analysis of complex systems",
  inputSchema: {
    type: "object",
    properties: {
      thought: { type: "string", description: "Your current thinking step" },
      nextStepNeeded: { type: "boolean", description: "Whether another step is needed" },
      stepNumber: { type: "number", minimum: 1, description: "Current step number" },
      totalSteps: { type: "number", minimum: 1, description: "Estimated total steps needed" },
      thinkingMethod: { type: "string", enum: ["systems"], description: "Thinking method type" },
      systemName: { type: "string", description: "Name of the system being analyzed" },
      purpose: { type: "string", description: "Main purpose of the system" },
      components: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string" },
            type: { type: "string", enum: ["input", "process", "output", "feedback", "environment"] },
            description: { type: "string" },
            relationships: { type: "array", items: { type: "string" } }
          },
          required: ["name", "type", "description", "relationships"]
        },
        description: "System components"
      },
      feedbackLoops: { type: "array", items: { type: "string" }, description: "Feedback loops" },
      constraints: { type: "array", items: { type: "string" }, description: "System constraints" },
      emergentProperties: { type: "array", items: { type: "string" }, description: "Emergent properties" },
      leveragePoints: { type: "array", items: { type: "string" }, description: "High-impact intervention points" },
      systemicIssues: { type: "array", items: { type: "string" }, description: "Systemic issues" },
      interventions: { type: "array", items: { type: "string" }, description: "Proposed interventions" }
    },
    required: ["thought", "nextStepNeeded", "stepNumber", "totalSteps", "thinkingMethod", "systemName", "purpose", "components", "feedbackLoops", "constraints", "emergentProperties", "leveragePoints", "systemicIssues", "interventions"]
  }
};

const DESIGN_THINKING_TOOL: Tool = {
  name: "mcp_designthinking_designthinking",
  description: "A tool for human-centered design thinking process",
  inputSchema: {
    type: "object",
    properties: {
      thought: { type: "string", description: "Your current thinking step" },
      nextStepNeeded: { type: "boolean", description: "Whether another step is needed" },
      stepNumber: { type: "number", minimum: 1, description: "Current step number" },
      totalSteps: { type: "number", minimum: 1, description: "Estimated total steps needed" },
      thinkingMethod: { type: "string", enum: ["design"], description: "Thinking method type" },
      phase: { type: "string", enum: ["empathize", "define", "ideate", "prototype", "test"], description: "Design thinking phase" },
      userInsights: { type: "array", items: { type: "string" }, description: "User insights gathered" },
      problemStatement: { type: "string", description: "Defined problem statement" },
      solutions: { type: "array", items: { type: "string" }, description: "Generated solutions" },
      prototypes: { type: "array", items: { type: "string" }, description: "Prototypes created" },
      testResults: { type: "array", items: { type: "string" }, description: "Test results" },
      iterations: { type: "array", items: { type: "string" }, description: "Iterations made" }
    },
    required: ["thought", "nextStepNeeded", "stepNumber", "totalSteps", "thinkingMethod", "phase", "userInsights", "problemStatement", "solutions", "prototypes", "testResults", "iterations"]
  }
};

// Main server setup
const server = new Server(
  {
    name: "thinking-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

const thinkingServer = new ThinkingServer();

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      SEQUENTIAL_THINKING_TOOL,
      LATERAL_THINKING_TOOL,
      CRITICAL_THINKING_TOOL,
      SYSTEMS_THINKING_TOOL,
      DESIGN_THINKING_TOOL
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "mcp_sequentialthinking_sequentialthinking":
      return thinkingServer.processSequentialThought(request.params.arguments);
    case "mcp_lateralthinking_lateralthinking":
      return thinkingServer.processLateralThought(request.params.arguments);
    case "mcp_criticalthinking_criticalthinking":
      return thinkingServer.processCriticalThought(request.params.arguments);
    case "mcp_systemsthinking_systemsthinking":
      return thinkingServer.processSystemsThought(request.params.arguments);
    case "mcp_designthinking_designthinking":
      return thinkingServer.processDesignThought(request.params.arguments);
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Multi-method Thinking MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});