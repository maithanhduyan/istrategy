"""
Advanced Tool Executor
Extends the basic ToolExecutor with RAG, Thinking, and Inference capabilities
"""

import asyncio
from typing import Dict, List, Any, Optional
from .tools import ToolExecutor
from .rag_engine import get_rag_tools
from .thinking_bridge import get_thinking_tools
from .inference_engine import get_inference_tools


class AdvancedToolExecutor(ToolExecutor):
    """Extended ToolExecutor with advanced reasoning capabilities"""

    def __init__(self, enable_async: bool = True):
        """
        Initialize AdvancedToolExecutor

        Args:
            enable_async: Whether to enable async tool execution
        """
        super().__init__()
        self.enable_async = enable_async
        self.advanced_tools = {}
        self.tool_categories = {
            "basic": [],
            "rag": [],
            "thinking": [],
            "inference": [],
            "async": [],
        }

        # Load advanced tools
        self._load_advanced_tools()
        self._categorize_tools()

    def _load_advanced_tools(self):
        """Load all advanced tool categories"""
        try:
            # RAG tools
            rag_tools = get_rag_tools()
            self.advanced_tools.update(rag_tools)

            # Thinking tools
            thinking_tools = get_thinking_tools()
            self.advanced_tools.update(thinking_tools)

            # Inference tools
            inference_tools = get_inference_tools()
            self.advanced_tools.update(inference_tools)

            # Tree-of-Thought planner tools
            from .tree_of_thought_planner import get_tree_of_thought_tools

            tot_tools = get_tree_of_thought_tools()
            self.advanced_tools.update(tot_tools)

            # AutoGPT planner tools
            from .autogpt_planner import get_autogpt_planner_tools

            autogpt_tools = get_autogpt_planner_tools()
            self.advanced_tools.update(autogpt_tools)

            # Prompt optimizer tools
            from .prompt_optimizer import get_prompt_optimizer_tools

            prompt_tools = get_prompt_optimizer_tools()
            self.advanced_tools.update(prompt_tools)

            # Merge with basic tools
            self.tools.update(self.advanced_tools)

        except Exception as e:
            print(f"Warning: Failed to load some advanced tools: {e}")

    def _categorize_tools(self):
        """Categorize tools by type for better organization"""
        # Basic tools (from parent class)
        basic_tools = [
            "date_diff",
            "run_python",
            "read_file",
            "write_file",
            "run_shell",
            "math_calc",
            "search_text",
        ]
        self.tool_categories["basic"] = basic_tools

        # RAG tools
        rag_tools = ["rag_add_knowledge", "rag_search", "rag_augmented_query"]
        self.tool_categories["rag"] = rag_tools

        # Thinking tools
        thinking_tools = [
            "think_sequential",
            "think_systems",
            "think_critical",
            "think_lateral",
            "think_root_cause",
            "think_six_hats",
            "get_thinking_summary",
        ]
        self.tool_categories["thinking"] = thinking_tools

        # Inference tools
        inference_tools = [
            "logical_add_rule",
            "logical_add_fact",
            "logical_infer",
            "pattern_analyze_numeric",
            "pattern_analyze_text",
            "inference_status",
        ]
        self.tool_categories["inference"] = inference_tools

        # Planning tools
        planning_tools = [
            "tree_of_thought_planning",
            "evaluate_thought_paths",
            "autonomous_plan_execute",
            "analyze_goal_complexity",
        ]
        self.tool_categories["planning"] = planning_tools

        # Optimization tools
        optimization_tools = ["optimize_prompt", "adapt_prompt_dynamically"]
        self.tool_categories["optimization"] = optimization_tools

        # Async capable tools
        async_tools = rag_tools + thinking_tools + planning_tools + optimization_tools
        self.tool_categories["async"] = async_tools

    def execute(self, action: str, args: List[str]) -> str:
        """Execute tool with async support for advanced tools"""
        if action not in self.tools:
            return f"Error: Unknown tool '{action}'"

        try:
            # Check if tool supports async execution
            if self.enable_async and action in self.tool_categories["async"]:
                return self._execute_async_tool(action, args)
            else:
                # Use parent class execution for basic tools
                return super().execute(action, args)

        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def _execute_async_tool(self, action: str, args: List[str]) -> str:
        """Execute async tools using asyncio"""
        try:
            # Get the async tool function
            tool_func = self.tools[action]

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tool_func(args))
            loop.close()

            return result

        except Exception as e:
            return f"Error in async execution of {action}: {str(e)}"

    def list_tools(self) -> str:
        """List all available tools with categories"""
        result = "Available Tools by Category:\n"
        result += "=" * 40 + "\n"

        for category, tools in self.tool_categories.items():
            if tools:  # Only show categories that have tools
                result += f"\n{category.upper()} TOOLS:\n"
                result += "-" * 20 + "\n"

                for tool in tools:
                    if tool in self.tools:
                        description = self._get_tool_description(tool)
                        async_marker = (
                            " [ASYNC]" if tool in self.tool_categories["async"] else ""
                        )
                        result += f"  • {tool}{async_marker}: {description}\n"

        result += f"\nTotal Tools Available: {len(self.tools)}\n"
        result += f"Advanced Tools: {len(self.advanced_tools)}\n"
        result += f"Async Support: {'Enabled' if self.enable_async else 'Disabled'}\n"

        return result

    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool"""
        descriptions = {
            # Basic tools
            "date_diff": "Calculate days between two dates",
            "run_python": "Execute Python code safely",
            "read_file": "Read content from file",
            "write_file": "Write content to file",
            "run_shell": "Execute shell commands",
            "math_calc": "Evaluate mathematical expressions",
            "search_text": "Search for text in file",
            # RAG tools
            "rag_add_knowledge": "Add knowledge to RAG system",
            "rag_search": "Search knowledge base",
            "rag_augmented_query": "Create context-augmented query",
            # Thinking tools
            "think_sequential": "Step-by-step sequential reasoning",
            "think_systems": "Systems analysis and thinking",
            "think_critical": "Critical evaluation of claims",
            "think_lateral": "Creative lateral thinking",
            "think_root_cause": "Root cause analysis",
            "think_six_hats": "Six thinking hats methodology",
            "get_thinking_summary": "Summary of thinking processes",
            # Inference tools
            "logical_add_rule": "Add logical inference rule",
            "logical_add_fact": "Add logical fact",
            "logical_infer": "Perform logical inference",
            "pattern_analyze_numeric": "Analyze numeric patterns",
            "pattern_analyze_text": "Analyze text patterns",
            "inference_status": "Get inference engine status",
        }

        return descriptions.get(tool_name, "Tool description not available")

    def get_tool_category(self, tool_name: str) -> Optional[str]:
        """Get the category of a tool"""
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return None

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a specific category"""
        return self.tool_categories.get(category, [])

    def execute_workflow(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a workflow of multiple tool calls"""
        results = []

        for step in workflow:
            action = step.get("action")
            args = step.get("args", [])
            step_name = step.get("name", f"Step {len(results) + 1}")

            if not action:
                results.append({"step": step_name, "error": "No action specified"})
                continue

            try:
                result = self.execute(action, args)
                results.append(
                    {
                        "step": step_name,
                        "action": action,
                        "args": args,
                        "result": result,
                        "success": "Error" not in result,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "step": step_name,
                        "action": action,
                        "args": args,
                        "error": str(e),
                        "success": False,
                    }
                )

        return results

    def analyze_problem_complexity(self, problem_description: str) -> Dict[str, Any]:
        """Analyze problem complexity and suggest appropriate tools"""
        problem_lower = problem_description.lower()

        suggested_tools = []
        complexity_score = 0

        # Check for different types of reasoning needed
        if any(word in problem_lower for word in ["pattern", "sequence", "trend"]):
            suggested_tools.extend(["pattern_analyze_numeric", "pattern_analyze_text"])
            complexity_score += 2

        if any(
            word in problem_lower for word in ["logic", "prove", "infer", "conclude"]
        ):
            suggested_tools.extend(["logical_add_fact", "logical_infer"])
            complexity_score += 3

        if any(
            word in problem_lower
            for word in ["research", "knowledge", "search", "information"]
        ):
            suggested_tools.extend(["rag_search", "rag_augmented_query"])
            complexity_score += 2

        if any(
            word in problem_lower
            for word in ["analyze", "think", "reasoning", "consider"]
        ):
            suggested_tools.extend(
                ["think_sequential", "think_critical", "think_systems"]
            )
            complexity_score += 3

        if any(word in problem_lower for word in ["calculate", "math", "compute"]):
            suggested_tools.extend(["math_calc"])
            complexity_score += 1

        if any(word in problem_lower for word in ["file", "document", "text"]):
            suggested_tools.extend(["read_file", "write_file", "search_text"])
            complexity_score += 1

        # Determine complexity level
        if complexity_score <= 2:
            complexity_level = "Simple"
        elif complexity_score <= 5:
            complexity_level = "Moderate"
        elif complexity_score <= 8:
            complexity_level = "Complex"
        else:
            complexity_level = "Very Complex"

        return {
            "problem": problem_description,
            "complexity_level": complexity_level,
            "complexity_score": complexity_score,
            "suggested_tools": list(set(suggested_tools)),
            "estimated_steps": len(set(suggested_tools)),
            "requires_async": any(
                tool in self.tool_categories["async"] for tool in suggested_tools
            ),
        }

    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get a summary of all advanced capabilities"""
        return {
            "total_tools": len(self.tools),
            "basic_tools": len(self.tool_categories["basic"]),
            "advanced_tools": len(self.advanced_tools),
            "categories": {
                "rag_capabilities": len(self.tool_categories["rag"]),
                "thinking_capabilities": len(self.tool_categories["thinking"]),
                "inference_capabilities": len(self.tool_categories["inference"]),
            },
            "async_support": self.enable_async,
            "async_tools": len(self.tool_categories["async"]),
            "workflow_support": True,
            "complexity_analysis": True,
        }


# Factory function for easy instantiation
def create_advanced_tool_executor(enable_async: bool = True) -> AdvancedToolExecutor:
    """Create an AdvancedToolExecutor instance"""
    return AdvancedToolExecutor(enable_async=enable_async)
