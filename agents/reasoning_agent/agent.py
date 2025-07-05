"""ReAct-style reasoning agent using DeepSeek-R1"""

import re
from typing import List, Tuple, Optional
from ollama_client import OllamaClient
from tools import ToolExecutor
from config import MAX_ITERATIONS


class ReasoningAgent:
    """ReAct-style reasoning agent"""

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.tool_executor = ToolExecutor()
        self.conversation_history = []

    def create_system_prompt(self) -> str:
        """Create system prompt for ReAct reasoning"""
        return """You are a concise reasoning agent. Answer directly using tools.

Available tools:
- date_diff(date1, date2): Calculate days between ISO dates
- run_python(code): Execute Python code
- read_file(filepath): Read file content
- write_file(filepath, content): Write content to file
- math_calc(expression): Evaluate math expression
- search_text(filepath, search_term): Search text in file
- run_shell(command): Execute shell commands

Format (be concise):
Question: [question]
Thought 1: [brief reasoning]
Action 1: [tool_name(args)]
Observation 1: [result]
Answer: [final answer]

Rules:
- Keep thoughts brief (1-2 sentences max)
- Execute tools when possible rather than reasoning manually
- End with Answer: when complete
- No verbose explanations

Begin!
"""

    def parse_response(
        self, response: str
    ) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
        """Parse LLM response to extract thought, action, and arguments"""

        # Extract the last thought
        thought_pattern = r"Thought \d+: (.+?)(?=Action \d+:|Answer:|$)"
        thought_matches = re.findall(thought_pattern, response, re.DOTALL)
        last_thought = thought_matches[-1].strip() if thought_matches else None

        # Extract the last action
        action_pattern = r"Action \d+: (\w+)\((.*?)\)"
        action_matches = re.findall(action_pattern, response)

        if action_matches:
            action_name, args_str = action_matches[-1]

            # Parse arguments (simple CSV parsing)
            args = []
            if args_str.strip():
                # Handle quoted strings and simple arguments
                import csv
                import io

                try:
                    reader = csv.reader(io.StringIO(args_str))
                    args = next(reader)
                    # Remove quotes from arguments
                    args = [arg.strip().strip("\"'") for arg in args]
                except:
                    # Fallback: split by comma and strip
                    args = [arg.strip().strip("\"'") for arg in args_str.split(",")]

            return last_thought, action_name, args

        return last_thought, None, None

    def is_complete(self, response: str) -> bool:
        """Check if the reasoning is complete (has Answer:)"""
        return "Answer:" in response

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        answer_pattern = r"Answer: (.+?)(?:\n|$)"
        match = re.search(answer_pattern, response, re.DOTALL)
        return match.group(1).strip() if match else "No answer found"

    def solve(self, question: str) -> str:
        """Solve a question using ReAct reasoning"""

        if not self.ollama_client.is_available():
            return "Error: Ollama is not available. Please start Ollama and ensure DeepSeek-R1:8B is installed."

        # Initialize conversation
        system_prompt = self.create_system_prompt()
        conversation = f"{system_prompt}\n\nQuestion: {question}\n"

        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Get LLM response
            print("Calling LLM...")
            response = self.ollama_client.generate(conversation)

            if not response or "Error" in response:
                return f"LLM Error: {response}"

            print(f"LLM Response:\n{response}")

            # Check if reasoning is complete
            if self.is_complete(response):
                answer = self.extract_answer(response)
                print(f"\nFinal Answer: {answer}")
                return answer

            # Parse response to get action
            thought, action, args = self.parse_response(response)

            if action and args is not None:
                print(f"Executing: {action}({args})")

                # Execute tool
                observation = self.tool_executor.execute(action, args)
                print(f"Observation: {observation}")

                # Update conversation with observation
                conversation += (
                    f"{response}\nObservation {iteration + 1}: {observation}\n"
                )
            else:
                # No valid action found, continue with LLM response
                conversation += f"{response}\n"

        return "Error: Maximum iterations reached without finding an answer"

    def chat(self):
        """Interactive chat mode"""
        print("Reasoning Agent Chat Mode")
        print("Type 'quit' to exit, 'tools' to list available tools")
        print("-" * 50)

        while True:
            try:
                question = input("\nYour question: ").strip()

                if question.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                elif question.lower() == "tools":
                    print(self.tool_executor.list_tools())
                    continue
                elif not question:
                    continue

                print("\nSolving...")
                answer = self.solve(question)
                print(f"\nFinal Answer: {answer}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    agent = ReasoningAgent()
    agent.chat()
