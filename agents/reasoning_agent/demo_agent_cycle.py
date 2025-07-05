#!/usr/bin/env python3
"""
Demo script to illustrate Agent Cycle operation in the reasoning agent
This script shows step-by-step how the ReAct pattern works
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import ReasoningAgent
from src.tools import ToolExecutor
import time


def analyze_agent_cycle():
    """Phân tích chi tiết Agent Cycle architecture"""

    print("=" * 80)
    print("🧠 REASONING AGENT - AGENT CYCLE ANALYSIS")
    print("=" * 80)

    print("\n📋 1. AGENT CYCLE COMPONENTS:")
    print("-" * 40)
    print("✅ Backend Selection: Auto-detect (Together.xyz → Ollama fallback)")
    print("✅ ReAct Pattern: Thought → Action → Observation → Repeat")
    print("✅ Tool Executor: 7 local tools available")
    print("✅ Response Parser: Extract thoughts, actions, and arguments")
    print("✅ Completion Detector: Detect when reasoning is complete")
    print("✅ Answer Extractor: Extract final answer from response")

    print("\n🔄 2. AGENT CYCLE FLOW:")
    print("-" * 40)
    print("Step 1: Initialize agent with AI backend (Together.xyz or Ollama)")
    print("Step 2: Create system prompt with tool descriptions")
    print("Step 3: Start conversation with user question")
    print("Step 4: Enter ReAct Loop (max iterations: 20)")
    print("Step 5: AI generates response with Thought/Action pattern")
    print("Step 6: Parse response to extract action and arguments")
    print("Step 7: Execute tool with arguments")
    print("Step 8: Add observation to conversation context")
    print("Step 9: Check if reasoning is complete (Answer: found)")
    print("Step 10: Return final answer or continue loop")

    print("\n🛠️ 3. AVAILABLE TOOLS:")
    tool_executor = ToolExecutor()
    tools_info = {
        "date_diff": "Calculate days between two ISO dates",
        "math_calc": "Evaluate mathematical expressions with functions",
        "run_python": "Execute Python code safely",
        "read_file": "Read content from file",
        "write_file": "Write content to file",
        "search_text": "Search for text in file",
        "run_shell": "Execute shell commands (with security checks)",
    }

    for tool, desc in tools_info.items():
        print(f"   • {tool}: {desc}")

    print("\n🤖 4. AI BACKEND SELECTION:")
    print("-" * 40)
    print("Priority 1: Together.xyz (DeepSeek-V3) - Fast cloud API")
    print("Priority 2: Ollama (Local) - DeepSeek-R1:8B fallback")
    print("Auto-detection: Try Together first, fallback to Ollama")

    return True


def demonstrate_agent_cycle():
    """Minh họa Agent Cycle với một ví dụ cụ thể"""

    print("\n" + "=" * 80)
    print("🎯 AGENT CYCLE DEMONSTRATION")
    print("=" * 80)

    # Initialize agent
    print("\n🚀 Initializing Reasoning Agent...")
    agent = ReasoningAgent(backend="auto")

    if not agent.ai_client:
        print("❌ Failed to initialize AI client")
        return False

    # Demo question
    demo_question = "What is the square root of 144 plus 5?"
    print(f"\n📝 Demo Question: {demo_question}")
    print("\n" + "-" * 50)

    # Analyze the solve process step by step
    print("🔍 TRACING AGENT CYCLE EXECUTION:")
    print("-" * 40)

    # Show system prompt
    system_prompt = agent.create_system_prompt()
    print("📋 System Prompt Created:")
    print("   • Tool descriptions provided")
    print("   • ReAct format specified")
    print("   • Concise response rules set")

    print(f"\n💭 Starting ReAct reasoning cycle...")
    print(f"   Question: {demo_question}")

    # Call solve with tracing
    try:
        answer = agent.solve(demo_question)
        print(f"\n✅ Final Answer Obtained: {answer}")

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        return False

    return True


def show_parsing_logic():
    """Minh họa logic parsing của agent"""

    print("\n" + "=" * 80)
    print("🔧 RESPONSE PARSING LOGIC")
    print("=" * 80)

    agent = ReasoningAgent()

    # Example AI responses to parse
    example_responses = [
        """Thought 1: I need to calculate the square root of 144 first.
Action 1: math_calc(sqrt(144))""",
        """Thought 2: Now I need to add 5 to the result.
Action 2: math_calc(12 + 5)""",
        """Thought 3: I have the final result.
Answer: 17""",
    ]

    print("\n📝 Example Response Parsing:")
    print("-" * 40)

    for i, response in enumerate(example_responses, 1):
        print(f"\nExample {i}:")
        print(f"Response: {response}")

        thought, action, args = agent.parse_response(response)
        print(f"Parsed:")
        print(f"   • Thought: {thought}")
        print(f"   • Action: {action}")
        print(f"   • Args: {args}")

        is_complete = agent.is_complete(response)
        print(f"   • Complete: {is_complete}")

    return True


def main():
    """Main demo function"""

    try:
        # Part 1: Architecture Analysis
        analyze_agent_cycle()

        # Part 2: Live Demonstration
        demonstrate_agent_cycle()

        # Part 3: Parsing Logic
        show_parsing_logic()

        print("\n" + "=" * 80)
        print("✅ AGENT CYCLE ANALYSIS COMPLETE")
        print("=" * 80)

        print("\n📊 KEY INSIGHTS:")
        print("• ReAct pattern enables systematic reasoning with tool usage")
        print("• Multi-backend support provides reliability and performance")
        print("• Local tool execution ensures security and control")
        print("• Structured parsing enables precise action extraction")
        print("• Iterative approach handles complex multi-step problems")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    main()
