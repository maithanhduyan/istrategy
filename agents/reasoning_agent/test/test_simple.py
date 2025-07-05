import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ReasoningAgent

# Create a simple test
agent = ReasoningAgent()

# Test just the solve logic with debug prints
question = "What is 2 + 2?"

# Mock the ollama response to test tool execution
test_response = """Question: What is 2 + 2?
Thought 1: I need to calculate this simple math expression.
Action 1: math_calc("2 + 2")
"""

print("Testing tool execution:")
thought, action, args = agent.parse_response(test_response)
print(f"Parsed - Action: {action}, Args: {args}")

if action and args is not None:
    result = agent.tool_executor.execute(action, args)
    print(f"Tool result: {result}")
else:
    print("No action found!")
