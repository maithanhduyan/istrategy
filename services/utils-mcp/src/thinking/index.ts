
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
import { SequentialThinkingServer, SequentialThinkingData, SEQUENTIAL_TOOL } from './src/sequential.js';
import { LateralThinkingServer, LateralThoughtData, LATERAL_THINKING_TOOL } from './src/lateral.js';
import { CriticalThinkingServer, CriticalAnalysis, CRITICAL_THINKING_TOOL } from './src/critical.js';
import { SystemsThinkingServer, SystemsAnalysis, SYSTEMS_THINKING_TOOL } from './src/systems.js';
import { RootCauseServer, RootCauseAnalysis, ROOT_CAUSE_TOOL } from './src/root_cause.js';

// Base interface for all thinking methods
interface BaseThinkingData {
  thought: string;
  stepNumber: number;
  totalSteps: number;
  nextStepNeeded: boolean;
  thinkingMethod: 'sequential' | 'lateral' | 'critical' | 'systems' | 'design';
}

class ThinkingServer {
  private sequentialServer: SequentialThinkingServer;
  private lateralServer: LateralThinkingServer;
  private criticalServer: CriticalThinkingServer;
  private systemsServer: SystemsThinkingServer;
  private rootCauseServer: RootCauseServer;
  private disableThoughtLogging: boolean;

  constructor() {
    this.disableThoughtLogging = (process.env.DISABLE_THOUGHT_LOGGING || "").toLowerCase() === "true";
    this.sequentialServer = new SequentialThinkingServer();
    this.lateralServer = new LateralThinkingServer();
    this.criticalServer = new CriticalThinkingServer();
    this.systemsServer = new SystemsThinkingServer();
    this.rootCauseServer = new RootCauseServer();
  }

  public processSequentialThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    return this.sequentialServer.processSequentialThought(input);
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

  public processRootCauseAnalysis(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    return this.rootCauseServer.processRootCauseAnalysis(input);
  }
}

// Define thinking tools
// Tool definitions are imported from modules

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
      SEQUENTIAL_TOOL,
      LATERAL_THINKING_TOOL,
      CRITICAL_THINKING_TOOL,
      SYSTEMS_THINKING_TOOL,
      ROOT_CAUSE_TOOL
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "sequentialthinking":
      return thinkingServer.processSequentialThought(request.params.arguments);
    case "lateralthinking":
      return thinkingServer.processLateralThought(request.params.arguments);
    case "criticalthinking":
      return thinkingServer.processCriticalThought(request.params.arguments);
    case "systemsthinking":
      return thinkingServer.processSystemsThought(request.params.arguments);
    case "rootcauseanalysis":
      return thinkingServer.processRootCauseAnalysis(request.params.arguments);
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