"""ReAct-style reasoning agent using DeepSeek-R1"""

import re
import os
from typing import List, Tuple, Optional, Union

# Use absolute imports to avoid relative import issues
try:
    from .ollama_client import OllamaClient
    from .together_client import TogetherAIClient
    from .tools import ToolExecutor
    from .advanced_tools import AdvancedToolExecutor
    from .config import MAX_ITERATIONS
except ImportError:
    # Fallback to direct imports when used as script
    from ollama_client import OllamaClient
    from together_client import TogetherAIClient
    from tools import ToolExecutor
    from advanced_tools import AdvancedToolExecutor
    from config import MAX_ITERATIONS


class ReasoningAgent:
    """ReAct-style reasoning agent with multiple AI backend support"""

    def __init__(self, backend: str = "auto", use_advanced_tools: bool = False):
        """
        Initialize reasoning agent with specified backend

        Args:
            backend: "ollama", "together", or "auto" (try together first, fallback to ollama)
            use_advanced_tools: Whether to use advanced tools (RAG, thinking, inference)
        """
        self.backend = backend
        self.use_advanced_tools = use_advanced_tools
        self.ai_client = None

        # Initialize tool executor based on advanced tools preference
        if use_advanced_tools:
            try:
                self.tool_executor = AdvancedToolExecutor(enable_async=True)
                print("✅ Advanced tools enabled (RAG, Thinking, Inference)")
            except Exception as e:
                print(f"⚠️ Failed to load advanced tools, using basic tools: {e}")
                self.tool_executor = ToolExecutor()
        else:
            self.tool_executor = ToolExecutor()

        self.conversation_history = []

        # Initialize AI client based on backend preference
        self._initialize_ai_client()

    def _initialize_ai_client(self):
        """Initialize AI client based on backend preference"""
        if self.backend == "together":
            self._try_together_client()
        elif self.backend == "ollama":
            self._try_ollama_client()
        elif self.backend == "auto":
            # Try together first (faster), fallback to ollama
            if not self._try_together_client():
                self._try_ollama_client()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _try_together_client(self) -> bool:
        """Try to initialize Together.xyz client"""
        try:
            client = TogetherAIClient()
            if client.is_available():
                self.ai_client = client
                print(f"✅ Using Together.xyz - {client.model}")
                return True
            else:
                print("❌ Together.xyz API not available")
                return False
        except Exception as e:
            print(f"❌ Together.xyz initialization failed: {str(e)}")
            return False

    def _try_ollama_client(self) -> bool:
        """Try to initialize Ollama client"""
        try:
            client = OllamaClient()
            if client.is_available():
                self.ai_client = client
                print(f"✅ Using Ollama - {client.model}")
                return True
            else:
                print("❌ Ollama not available")
                return False
        except Exception as e:
            print(f"❌ Ollama initialization failed: {str(e)}")
            return False

    def create_system_prompt(self) -> str:
        """Create system prompt for ReAct reasoning"""
        basic_tools = """Available tools:
- date_diff(date1, date2): Calculate days between ISO dates
- run_python(code): Execute Python code
- read_file(filepath): Read file content
- write_file(filepath, content): Write content to file
- math_calc(expression): Evaluate math expression
- search_text(filepath, search_term): Search text in file
- run_shell(command): Execute shell commands"""

        advanced_tools = """
ADVANCED REASONING TOOLS:

RAG (Knowledge) Tools:
- rag_add_knowledge(text, source): Add knowledge to system
- rag_search(query, max_results): Search knowledge base
- rag_augmented_query(question): Create context-enhanced query

Thinking Tools:
- think_sequential(problem, num_thoughts): Step-by-step analysis
- think_systems(system_name, component1, component2, ...): Systems analysis
- think_critical(claim, evidence1, evidence2, ...): Critical evaluation
- think_lateral(challenge, technique): Creative problem solving
- think_root_cause(problem, symptom1, symptom2, ...): Root cause analysis
- think_six_hats(topic, hat_color): Six thinking hats methodology
- get_thinking_summary(): Summary of thinking processes

Inference Tools:
- logical_add_rule(premise, conclusion, confidence): Add logical rule
- logical_add_fact(fact, confidence): Add logical fact
- logical_infer(method): Perform logical inference (forward/prove:goal)
- pattern_analyze_numeric(num1, num2, num3, ...): Analyze numeric patterns
- pattern_analyze_text(text1, text2, text3, ...): Analyze text patterns
- inference_status(): Get inference engine status"""

        tools_section = basic_tools
        if self.use_advanced_tools:
            tools_section += advanced_tools

        return f"""You are an advanced reasoning agent with sophisticated analytical capabilities.

{tools_section}

Format (be concise):
Question: [question]
Thought 1: [brief reasoning]
Action 1: [tool_name(args)]
Observation 1: [result]
Answer: [final answer]

Advanced Reasoning Guidelines:
- Use thinking tools for complex analysis before taking action
- Use RAG tools when knowledge lookup would be helpful
- Use inference tools for logical reasoning and pattern analysis
- Combine multiple tools for sophisticated problem solving
- Keep thoughts brief but leverage advanced capabilities

Rules:
- Keep thoughts brief (1-2 sentences max)
- Execute tools when possible rather than reasoning manually
- End with Answer: when complete
- Use advanced tools for complex reasoning tasks

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
        # Check for multiple Answer: patterns to catch completion
        answer_patterns = ["Answer:", "Final Answer:", "The answer is", "The result is"]

        for pattern in answer_patterns:
            if pattern in response:
                return True

        # Also check if response directly gives a numerical answer without action
        if re.search(r"\b\d+\b", response) and len(response.split("\n")) <= 3:
            return True

        return False

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        # Look for Answer: pattern first
        answer_pattern = r"Answer: (.+?)(?:\n|$)"
        match = re.search(answer_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: look for just the last line if it contains numbers/results
        lines = response.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith(
                ("Thought", "Action", "Observation")
            ):
                return last_line

        return "No answer found"

    def solve(self, question: str) -> str:
        """Solve a question using ReAct reasoning"""

        if not self.ai_client:
            return "Error: No AI backend available. Please check your configuration."

        if not self.ai_client.is_available():
            return f"Error: AI backend not available. Current backend: {type(self.ai_client).__name__}"

        # Initialize conversation
        system_prompt = self.create_system_prompt()
        conversation = f"{system_prompt}\n\nQuestion: {question}\n"

        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Get LLM response
            print("Calling AI...")
            response = self.ai_client.generate(conversation)

            if not response or "Error" in response:
                return f"AI Error: {response}"

            print(f"AI Response:\n{response}")

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
