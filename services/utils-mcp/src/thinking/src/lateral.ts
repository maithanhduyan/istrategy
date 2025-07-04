import chalk from 'chalk';
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export interface LateralThoughtData {
  technique: 'random_word' | 'provocation' | 'alternative' | 'reversal' | 'metaphor' | 'assumption_challenge';
  stimulus: string;
  connection: string;
  idea: string;
  evaluation: string;
  nextTechniqueNeeded: boolean;
}

export class LateralThinkingServer {
  private techniques: string[] = [];
  private ideas: LateralThoughtData[] = [];

  private validateLateralData(input: unknown): LateralThoughtData {
    const data = input as Record<string, unknown>;
    
    if (!data.technique || typeof data.technique !== 'string') {
      throw new Error('Invalid technique: must be a string');
    }
    if (!data.stimulus || typeof data.stimulus !== 'string') {
      throw new Error('Invalid stimulus: must be a string');
    }
    if (!data.connection || typeof data.connection !== 'string') {
      throw new Error('Invalid connection: must be a string');
    }
    if (!data.idea || typeof data.idea !== 'string') {
      throw new Error('Invalid idea: must be a string');
    }
    if (!data.evaluation || typeof data.evaluation !== 'string') {
      throw new Error('Invalid evaluation: must be a string');
    }
    if (typeof data.nextTechniqueNeeded !== 'boolean') {
      throw new Error('Invalid nextTechniqueNeeded: must be a boolean');
    }

    return {
      technique: data.technique as LateralThoughtData['technique'],
      stimulus: data.stimulus as string,
      connection: data.connection as string,
      idea: data.idea as string,
      evaluation: data.evaluation as string,
      nextTechniqueNeeded: data.nextTechniqueNeeded as boolean
    };
  }

  private formatLateralThought(data: LateralThoughtData): string {
    const techniqueEmojis: Record<string, string> = {
      'random_word': '🎲',
      'provocation': '🚀',
      'alternative': '🔄',
      'reversal': '↩️',
      'metaphor': '🎭',
      'assumption_challenge': '❓'
    };

    const emoji = techniqueEmojis[data.technique] || '💡';
    const header = chalk.magenta(`${emoji} Lateral Thinking: ${data.technique.toUpperCase()}`);
    
    return `
┌─────────────────────────────────────────┐
│ ${header} │
├─────────────────────────────────────────┤
│ Stimulus: ${data.stimulus} │
│ Connection: ${data.connection} │
│ Idea: ${data.idea} │
│ Evaluation: ${data.evaluation} │
└─────────────────────────────────────────┘`;
  }

  public processLateralThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
    try {
      const validatedInput = this.validateLateralData(input);
      
      this.ideas.push(validatedInput);
      this.techniques.push(validatedInput.technique);

      const formattedThought = this.formatLateralThought(validatedInput);
      console.error(formattedThought);

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            technique: validatedInput.technique,
            nextTechniqueNeeded: validatedInput.nextTechniqueNeeded,
            techniquesUsed: this.techniques,
            totalIdeas: this.ideas.length
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

const LATERAL_THINKING_TOOL: Tool = {
  name: "lateralthinking",
  description: `A tool for creative problem-solving using Edward de Bono's lateral thinking techniques.
Generates unconventional solutions by breaking normal thought patterns.

Techniques available:
- random_word: Use random stimulus to generate new connections
- provocation: Create deliberately unreasonable statements to provoke new ideas
- alternative: Generate multiple alternative approaches
- reversal: Reverse the problem or assumptions
- metaphor: Use analogies and metaphors for new perspectives
- assumption_challenge: Challenge fundamental assumptions

When to use:
- Need creative, innovative solutions
- Stuck in conventional thinking
- Want to generate multiple alternatives
- Breaking through mental blocks
- Brainstorming sessions
- Innovation challenges`,
  inputSchema: {
    type: "object",
    properties: {
      technique: {
        type: "string",
        enum: ["random_word", "provocation", "alternative", "reversal", "metaphor", "assumption_challenge"],
        description: "Lateral thinking technique to use"
      },
      stimulus: {
        type: "string",
        description: "The stimulus or prompt used for the technique"
      },
      connection: {
        type: "string", 
        description: "How the stimulus connects to the problem"
      },
      idea: {
        type: "string",
        description: "The creative idea generated"
      },
      evaluation: {
        type: "string",
        description: "Brief evaluation of the idea's potential"
      },
      nextTechniqueNeeded: {
        type: "boolean",
        description: "Whether to try another technique"
      }
    },
    required: ["technique", "stimulus", "connection", "idea", "evaluation", "nextTechniqueNeeded"]
  }
};

export { LATERAL_THINKING_TOOL };