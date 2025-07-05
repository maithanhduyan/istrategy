import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ReasoningAgent

agent = ReasoningAgent()
response = 'Action 1: date_diff("2022-01-01", "2025-07-05")'
thought, action, args = agent.parse_response(response)
print("Action:", action)
print("Args:", args)

# Test tool execution
from tools import ToolExecutor

te = ToolExecutor()
result = te.execute(action, args)
print("Tool result:", result)
