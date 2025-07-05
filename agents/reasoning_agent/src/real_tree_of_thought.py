"""
Real Tree-of-Thought Planner với Agent Integration
================================================================================
Phiên bản thực tế sử dụng ReasoningAgent thay vì mock data
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class RealThoughtNode:
    """Node thực tế trong Tree-of-Thought với agent reasoning"""

    id: str
    content: str
    parent_id: Optional[str] = None
    children: List[str] = None
    depth: int = 0
    confidence: float = 0.0
    reasoning_method: str = "general"
    agent_response_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class RealTreeOfThoughtPlanner:
    """
    Tree-of-Thought Planner thực tế với ReasoningAgent integration

    KHÔNG SỬ DỤNG MOCK DATA - chỉ sử dụng agent reasoning thực tế
    """

    def __init__(self, agent=None, max_depth: int = 3, branching_factor: int = 2):
        """
        Initialize với agent thực tế

        Args:
            agent: ReasoningAgent instance (REQUIRED for real reasoning)
            max_depth: Độ sâu tối đa của tree
            branching_factor: Số branches mỗi node
        """
        if agent is None:
            raise ValueError(
                "Agent is REQUIRED for real Tree-of-Thought reasoning. "
                "This planner does NOT use mock data."
            )

        if not hasattr(agent, "solve"):
            raise ValueError(
                "Agent must have 'solve' method for reasoning. "
                "Please provide a valid ReasoningAgent instance."
            )

        self.agent = agent
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.nodes: Dict[str, RealThoughtNode] = {}
        self.execution_log: List[Dict] = []

        print(
            f"✅ Real Tree-of-Thought initialized with agent: {agent.__class__.__name__}"
        )

    def plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Tree-of-Thought planning với agent thực tế

        Args:
            goal: Mục tiêu cần giải quyết
            context: Context bổ sung

        Returns:
            Kết quả planning với real reasoning
        """
        print(f"🌳 Starting REAL Tree-of-Thought planning...")
        print(f"🎯 Goal: {goal}")

        start_time = time.time()

        try:
            # Step 1: Tạo root node với agent reasoning
            root_node = self._create_root_node_real(goal, context or {})

            # Step 2: Explore tree với agent
            exploration_result = self._explore_tree_real()

            # Step 3: Synthesize final solution
            final_solution = self._synthesize_solution_real()

            total_time = time.time() - start_time

            result = {
                "goal": goal,
                "final_solution": final_solution,
                "total_nodes": len(self.nodes),
                "max_depth_reached": max(node.depth for node in self.nodes.values()),
                "execution_time": total_time,
                "confidence": self._calculate_overall_confidence(),
                "agent_calls": len(self.execution_log),
                "reasoning_method": "real_agent_integration",
            }

            print(f"✅ REAL Tree-of-Thought completed in {total_time:.2f}s")
            print(f"📊 Agent calls: {len(self.execution_log)}")
            print(f"📈 Nodes explored: {len(self.nodes)}")

            return result

        except Exception as e:
            print(f"❌ Real Tree-of-Thought error: {e}")
            raise RuntimeError(f"Tree-of-Thought planning failed: {e}")

    def _create_root_node_real(self, goal: str, context: Dict) -> RealThoughtNode:
        """Tạo root node với agent reasoning thực tế"""

        root_prompt = f"""
        Analyze this problem and provide initial reasoning approach:
        
        Goal: {goal}
        Context: {json.dumps(context, indent=2)}
        
        Provide your initial analysis and key considerations for solving this problem.
        Focus on the most important aspects and potential approaches.
        """

        print(f"🔍 Creating root node with agent reasoning...")
        start_time = time.time()

        # Sử dụng agent thực tế
        response = self.agent.solve(root_prompt)
        response_time = time.time() - start_time

        root_node = RealThoughtNode(
            id="root",
            content=response,
            depth=0,
            reasoning_method="root_analysis",
            agent_response_time=response_time,
            metadata={"goal": goal, "context": context, "node_type": "root"},
        )

        # Evaluate confidence với agent
        root_node.confidence = self._evaluate_node_real(root_node)

        self.nodes["root"] = root_node

        # Log execution
        self.execution_log.append(
            {
                "node_id": "root",
                "action": "create_root",
                "response_time": response_time,
                "content_length": len(response),
            }
        )

        print(f"✅ Root node created - Response time: {response_time:.2f}s")
        return root_node

    def _explore_tree_real(self) -> Dict[str, Any]:
        """Explore tree với agent reasoning thực tế"""

        nodes_to_explore = ["root"]
        current_depth = 0

        while current_depth < self.max_depth and nodes_to_explore:
            print(f"🔍 Exploring depth {current_depth + 1}...")

            next_level_nodes = []

            for node_id in nodes_to_explore:
                if self.nodes[node_id].depth == current_depth:
                    # Generate children với agent
                    children = self._generate_children_real(node_id)
                    next_level_nodes.extend(children)

            nodes_to_explore = next_level_nodes
            current_depth += 1

        return {
            "levels_explored": current_depth,
            "total_nodes": len(self.nodes),
            "agent_calls": len(self.execution_log),
        }

    def _generate_children_real(self, parent_id: str) -> List[str]:
        """Generate children nodes với agent reasoning"""

        parent_node = self.nodes[parent_id]
        children_ids = []

        # Các phương pháp reasoning khác nhau
        reasoning_methods = [
            "analytical_breakdown",
            "creative_approach",
            "systematic_analysis",
            "alternative_perspective",
        ]

        for i, method in enumerate(reasoning_methods[: self.branching_factor]):
            child_id = f"{parent_id}_child_{i}"

            # Tạo prompt cho method cụ thể
            child_content = self._generate_child_content_real(parent_node, method)

            child_node = RealThoughtNode(
                id=child_id,
                content=child_content,
                parent_id=parent_id,
                depth=parent_node.depth + 1,
                reasoning_method=method,
            )

            # Evaluate với agent
            child_node.confidence = self._evaluate_node_real(child_node)

            self.nodes[child_id] = child_node
            parent_node.children.append(child_id)
            children_ids.append(child_id)

            print(
                f"   ✅ Generated child {i+1}: {method} (confidence: {child_node.confidence:.2f})"
            )

        return children_ids

    def _generate_child_content_real(
        self, parent_node: RealThoughtNode, method: str
    ) -> str:
        """Generate child content với agent reasoning cụ thể"""

        goal = parent_node.metadata.get("goal", "Problem solving")

        method_prompts = {
            "analytical_breakdown": f"""
            Break down this problem analytically based on the previous reasoning:
            
            Goal: {goal}
            Previous reasoning: {parent_node.content}
            
            Provide a detailed analytical breakdown of one key component or aspect.
            Focus on logical decomposition and systematic analysis.
            """,
            "creative_approach": f"""
            Think creatively about this problem from the previous analysis:
            
            Goal: {goal} 
            Previous reasoning: {parent_node.content}
            
            Propose an innovative, unconventional approach or solution.
            Think outside the box and challenge assumptions.
            """,
            "systematic_analysis": f"""
            Apply systematic methodology based on previous reasoning:
            
            Goal: {goal}
            Previous reasoning: {parent_node.content}
            
            Use a structured, step-by-step systematic approach.
            Consider all relevant factors and their relationships.
            """,
            "alternative_perspective": f"""
            Consider this problem from a different angle:
            
            Goal: {goal}
            Previous reasoning: {parent_node.content}
            
            What would a different field/domain/expert think about this?
            Provide an alternative viewpoint or approach.
            """,
        }

        prompt = method_prompts.get(method, f"Continue reasoning about: {goal}")

        start_time = time.time()
        response = self.agent.solve(prompt)
        response_time = time.time() - start_time

        # Log execution
        self.execution_log.append(
            {
                "node_id": f"{parent_node.id}_child",
                "action": f"generate_{method}",
                "response_time": response_time,
                "content_length": len(response),
            }
        )

        return response

    def _evaluate_node_real(self, node: RealThoughtNode) -> float:
        """Evaluate node quality với agent thực tế"""

        evaluation_prompt = f"""
        Evaluate the quality of this reasoning step on a scale of 0.0 to 1.0:
        
        Content: {node.content}
        Method: {node.reasoning_method}
        Depth: {node.depth}
        
        Consider:
        1. Logical soundness (0.0-1.0)
        2. Relevance to goal (0.0-1.0) 
        3. Clarity and completeness (0.0-1.0)
        4. Potential for solution (0.0-1.0)
        
        Respond with ONLY a single number between 0.0 and 1.0
        """

        start_time = time.time()
        response = self.agent.solve(evaluation_prompt)
        eval_time = time.time() - start_time

        # Extract score
        try:
            import re

            score_match = re.search(r"0\.\d+|1\.0|0\.0", response)
            if score_match:
                score = float(score_match.group())
            else:
                # Fallback: quality heuristics
                if len(response) > 100 and any(
                    word in response.lower()
                    for word in ["good", "strong", "effective", "sound"]
                ):
                    score = 0.7
                elif len(response) > 50:
                    score = 0.5
                else:
                    score = 0.3

        except (ValueError, AttributeError):
            score = 0.5  # Default

        node.agent_response_time += eval_time

        # Log evaluation
        self.execution_log.append(
            {
                "node_id": node.id,
                "action": "evaluate",
                "response_time": eval_time,
                "score": score,
            }
        )

        return score

    def _synthesize_solution_real(self) -> str:
        """Synthesize final solution từ best paths với agent"""

        # Find best path
        best_path = self._find_best_path()

        # Combine reasoning từ best path
        path_content = []
        for node_id in best_path:
            node = self.nodes[node_id]
            path_content.append(f"Step {node.depth}: {node.content}")

        synthesis_prompt = f"""
        Synthesize a comprehensive solution from this reasoning path:
        
        {'=' * 50}
        {chr(10).join(path_content)}
        {'=' * 50}
        
        Provide a clear, actionable final solution that integrates the key insights
        from this reasoning process. Focus on practical implementation.
        """

        print(f"🔄 Synthesizing final solution...")
        start_time = time.time()

        final_solution = self.agent.solve(synthesis_prompt)
        synthesis_time = time.time() - start_time

        # Log synthesis
        self.execution_log.append(
            {
                "node_id": "synthesis",
                "action": "synthesize_solution",
                "response_time": synthesis_time,
                "content_length": len(final_solution),
            }
        )

        print(f"✅ Solution synthesized in {synthesis_time:.2f}s")
        return final_solution

    def _find_best_path(self) -> List[str]:
        """Find best reasoning path dựa trên confidence scores"""

        # Simple DFS to find highest confidence path
        def dfs(node_id: str, current_path: List[str]) -> Tuple[List[str], float]:
            current_path = current_path + [node_id]
            node = self.nodes[node_id]

            if not node.children:  # Leaf node
                avg_confidence = sum(
                    self.nodes[nid].confidence for nid in current_path
                ) / len(current_path)
                return current_path, avg_confidence

            best_path = current_path
            best_score = node.confidence

            for child_id in node.children:
                child_path, child_score = dfs(child_id, current_path)
                if child_score > best_score:
                    best_path = child_path
                    best_score = child_score

            return best_path, best_score

        best_path, _ = dfs("root", [])
        return best_path

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence từ all nodes"""
        if not self.nodes:
            return 0.0

        total_confidence = sum(node.confidence for node in self.nodes.values())
        return total_confidence / len(self.nodes)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get detailed execution summary"""

        total_agent_time = sum(log["response_time"] for log in self.execution_log)

        return {
            "total_nodes_created": len(self.nodes),
            "total_agent_calls": len(self.execution_log),
            "total_agent_time": total_agent_time,
            "average_response_time": (
                total_agent_time / len(self.execution_log) if self.execution_log else 0
            ),
            "confidence_distribution": {
                node_id: node.confidence for node_id, node in self.nodes.items()
            },
            "reasoning_methods_used": list(
                set(node.reasoning_method for node in self.nodes.values())
            ),
        }
