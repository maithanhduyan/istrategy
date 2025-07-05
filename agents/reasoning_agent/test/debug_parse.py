import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ReasoningAgent
import re

response = """Question: How many days between 2022-01-01 and 2025-07-05?     
Thought 1: I need to calculate the number of days between two s
specific dates, both provided in ISO format (YYYY-MM-DD). The da
ate_diff tool is designed for this purpose.
Action 1: date_diff("2022-01-01", "2025-07-05")
Observation 1: The result from the date_diff function will be a
a number representing the days between these two dates. Since no
o other tools are needed, I can proceed to use this directly."""

agent = ReasoningAgent()
thought, action, args = agent.parse_response(response)
print("Thought:", thought)
print("Action:", action)
print("Args:", args)
