
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

// Import thinking modules
import { LateralThinkingServer, LateralThoughtData, LATERAL_THINKING_TOOL } from './src/lateral.js';
import { CriticalThinkingServer, CriticalAnalysis, CRITICAL_THINKING_TOOL } from './src/critical.js';
import { SystemsThinkingServer, SystemsAnalysis, SYSTEMS_THINKING_TOOL } from './src/systems.js';

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

class ThinkingServer {
  private sequentialHistory: SequentialThinkingData[] = [];
  private lateralServer: LateralThinkingServer;
  private criticalServer: CriticalThinkingServer;
  private systemsServer: SystemsThinkingServer;
  private disableThoughtLogging: boolean;

  constructor() {
    this.disableThoughtLogging = (process.env.DISABLE_THOUGHT_LOGGING || "").toLowerCase() === "true";
    this.lateralServer = new LateralThinkingServer();
    this.criticalServer = new CriticalThinkingServer();
    this.systemsServer = new SystemsThinkingServer();
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
      thought: data.thought,
      stepNumber: data.stepNumber,
      totalSteps: data.totalSteps,
      nextStepNeeded: data.nextStepNeeded,
      thinkingMethod: 'sequential',
      isRevision: data.isRevision as boolean | undefined,
      revisesThought: data.revisesThought as number | undefined,
      branchFromThought: data.branchFromThought as number | undefined,
      branchId: data.branchId as string | undefined,
      needsMoreThoughts: data.needsMoreThoughts as boolean | undefined,
    };
  }

  private formatThought(thoughtData: SequentialThinkingData): string {
    const { stepNumber, totalSteps, thought, isRevision, revisesThought, branchFromThought, branchId } = thoughtData;

    let prefix = '';
    let context = '';

    if (isRevision) {
      prefix = chalk.yellow('🔄 Revision');
      context = ` (revising thought ${revisesThought})`;
    } else if (branchFromThought) {
      prefix = chalk.green('🌿 Branch');
      context = ` (from thought ${branchFromThought}, ID: ${branchId})`;
    } else {
      prefix = chalk.blue('💭 Thought');
      context = '';
    }

    const header = `${prefix} ${stepNumber}/${totalSteps}${context}`;
    const border = '─'.repeat(Math.max(header.length, thought.length) + 4);

    return `
┌${border}┐
│ ${header} │
├${border}┤
│ ${thought.padEnd(border.length - 2)} │
└${border}┘`;
  }

  public processSequentialThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateSequentialData(input);

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
    return this.lateralServer.processLateralThought(input);
  }

  public processCriticalThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    return this.criticalServer.processCriticalAnalysis(input);
  }

  public processSystemsThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    return this.systemsServer.processSystemsAnalysis(input);
  }
}

// Define thinking tools
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
      SYSTEMS_THINKING_TOOL
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "mcp_sequentialthinking_sequentialthinking":
      return thinkingServer.processSequentialThought(request.params.arguments);
    case "lateralthinking":
      return thinkingServer.processLateralThought(request.params.arguments);
    case "criticalthinking":
      return thinkingServer.processCriticalThought(request.params.arguments);
    case "systemsthinking":
      return thinkingServer.processSystemsThought(request.params.arguments);
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Multi-Method Thinking MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});