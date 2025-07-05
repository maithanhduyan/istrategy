"""
Tree-of-Thought Planner for Advanced Reasoning
Implements multi-path exploration and evaluation for complex problem solving
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import uuid


class NodeStatus(Enum):
    """Status of reasoning nodes in Tree-of-Thought"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class ThoughtNode:
    """Individual thought node in Tree-of-Thought structure"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    status: NodeStatus = NodeStatus.PENDING
    confidence: float = 0.0
    reasoning_method: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluation_score: float = 0.0
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "status": self.status.value,
            "confidence": self.confidence,
            "reasoning_method": self.reasoning_method,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "evaluation_score": self.evaluation_score,
            "execution_time": self.execution_time,
        }


class TreeOfThoughtPlanner:
    """
    Tree-of-Thought planner for multi-path reasoning and exploration

    Implements the Tree-of-Thought prompting paradigm:
    1. Generate multiple reasoning paths (thoughts)
    2. Evaluate each path independently
    3. Explore promising paths further
    4. Prune less promising branches
    5. Synthesize best solutions
    """

    def __init__(
        self,
        max_depth: int = 5,
        max_branches_per_node: int = 3,
        evaluation_threshold: float = 0.6,
        enable_pruning: bool = True,
    ):
        """
        Initialize Tree-of-Thought planner

        Args:
            max_depth: Maximum depth of exploration tree
            max_branches_per_node: Maximum branches to explore per node
            evaluation_threshold: Minimum score to continue exploration
            enable_pruning: Whether to prune low-scoring branches
        """
        self.max_depth = max_depth
        self.max_branches_per_node = max_branches_per_node
        self.evaluation_threshold = evaluation_threshold
        self.enable_pruning = enable_pruning

        # Tree structure
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.active_branches: Set[str] = set()

        # Planning state
        self.exploration_history: List[Dict] = []
        self.best_paths: List[List[str]] = []
        self.performance_metrics: Dict[str, Any] = {}

    async def plan_with_tree_of_thought(
        self, problem: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main Tree-of-Thought planning method

        Args:
            problem: Problem statement to solve
            context: Additional context for reasoning

        Returns:
            Planning result with best paths and solutions
        """
        print(f"🌳 Starting Tree-of-Thought planning for: {problem[:50]}...")

        start_time = time.time()

        # Initialize root node
        root_node = await self._create_root_node(problem, context or {})
        self.root_id = root_node.id

        # Explore tree iteratively
        exploration_result = await self._explore_tree()

        # Evaluate and select best paths
        best_paths = await self._select_best_paths()

        # Synthesize final solution
        final_solution = await self._synthesize_solution(best_paths)

        # Calculate metrics
        total_time = time.time() - start_time
        self._update_performance_metrics(total_time)

        result = {
            "problem": problem,
            "tree_structure": self._get_tree_structure(),
            "exploration_summary": exploration_result,
            "best_paths": best_paths,
            "final_solution": final_solution,
            "performance_metrics": self.performance_metrics,
            "total_nodes": len(self.nodes),
            "max_depth_reached": max(node.depth for node in self.nodes.values()),
            "execution_time": total_time,
        }

        print(f"✅ Tree-of-Thought planning completed in {total_time:.2f}s")
        print(f"   • Nodes explored: {len(self.nodes)}")
        print(f"   • Best paths found: {len(best_paths)}")
        print(f"   • Solution confidence: {final_solution.get('confidence', 0):.2f}")

        return result

    async def _create_root_node(self, problem: str, context: Dict) -> ThoughtNode:
        """Create and initialize root node"""
        root_node = ThoughtNode(
            content=f"Root: {problem}",
            depth=0,
            reasoning_method="root_analysis",
            metadata={
                "problem_statement": problem,
                "context": context,
                "node_type": "root",
            },
        )

        self.nodes[root_node.id] = root_node
        self.active_branches.add(root_node.id)

        # Generate initial evaluation
        root_node.confidence = await self._evaluate_node(root_node)
        root_node.status = NodeStatus.COMPLETED

        return root_node

    async def _explore_tree(self) -> Dict[str, Any]:
        """Main tree exploration loop"""
        exploration_stats = {
            "levels_explored": 0,
            "nodes_created": 0,
            "nodes_pruned": 0,
            "branches_explored": 0,
        }

        current_depth = 0

        while current_depth < self.max_depth and self.active_branches:
            print(f"🔍 Exploring depth {current_depth + 1}...")

            # Get nodes at current depth
            current_level_nodes = [
                node_id
                for node_id in self.active_branches
                if self.nodes[node_id].depth == current_depth
            ]

            if not current_level_nodes:
                break

            # Explore each node
            new_branches = set()
            for node_id in current_level_nodes:
                if self.nodes[node_id].status == NodeStatus.COMPLETED:
                    children = await self._generate_child_thoughts(node_id)
                    new_branches.update(children)
                    exploration_stats["branches_explored"] += 1

            # Update active branches
            self.active_branches = new_branches

            # Prune low-scoring branches if enabled
            if self.enable_pruning:
                pruned_count = await self._prune_branches()
                exploration_stats["nodes_pruned"] += pruned_count

            exploration_stats["levels_explored"] = current_depth + 1
            exploration_stats["nodes_created"] = len(self.nodes)
            current_depth += 1

        return exploration_stats

    async def _generate_child_thoughts(self, parent_id: str) -> List[str]:
        """Generate child thoughts for a given parent node"""
        parent_node = self.nodes[parent_id]
        child_ids = []

        # Generate different reasoning approaches
        reasoning_methods = [
            "analytical_breakdown",
            "creative_approach",
            "systematic_analysis",
            "alternative_perspective",
        ]

        for i, method in enumerate(reasoning_methods[: self.max_branches_per_node]):
            child_content = await self._generate_thought_content(parent_node, method, i)

            child_node = ThoughtNode(
                content=child_content,
                parent_id=parent_id,
                depth=parent_node.depth + 1,
                reasoning_method=method,
                metadata={
                    "parent_content": parent_node.content[:100],
                    "generation_index": i,
                    "reasoning_approach": method,
                },
            )

            # Evaluate child node
            start_time = time.time()
            child_node.confidence = await self._evaluate_node(child_node)
            child_node.execution_time = time.time() - start_time
            child_node.status = NodeStatus.COMPLETED

            # Add to tree
            self.nodes[child_node.id] = child_node
            parent_node.children_ids.append(child_node.id)
            child_ids.append(child_node.id)

        return child_ids

    async def _generate_thought_content(
        self, parent_node: ThoughtNode, method: str, index: int
    ) -> str:
        """Generate content for a thought node using specified reasoning method"""

        base_content = parent_node.content

        if method == "analytical_breakdown":
            return f"Analytical approach {index + 1}: Break down '{base_content}' into components and analyze systematically"
        elif method == "creative_approach":
            return f"Creative approach {index + 1}: Explore unconventional solutions for '{base_content}'"
        elif method == "systematic_analysis":
            return f"Systematic approach {index + 1}: Apply structured methodology to '{base_content}'"
        elif method == "alternative_perspective":
            return f"Alternative perspective {index + 1}: Consider different viewpoint for '{base_content}'"
        else:
            return f"General approach {index + 1}: Further explore '{base_content}'"

    async def _evaluate_node(self, node: ThoughtNode) -> float:
        """Evaluate the quality and promise of a thought node"""

        # Evaluation criteria
        criteria_scores = {}

        # Content quality (mock evaluation)
        content_quality = min(
            len(node.content) / 100, 1.0
        )  # Longer content = potentially better
        criteria_scores["content_quality"] = content_quality

        # Depth appropriateness
        depth_score = max(0.0, 1.0 - (node.depth / self.max_depth))
        criteria_scores["depth_score"] = depth_score

        # Reasoning method bonus
        method_bonus = {
            "analytical_breakdown": 0.9,
            "creative_approach": 0.8,
            "systematic_analysis": 0.95,
            "alternative_perspective": 0.85,
            "root_analysis": 1.0,
        }.get(node.reasoning_method, 0.7)
        criteria_scores["method_bonus"] = method_bonus

        # Novelty score (avoid repetitive thinking)
        novelty_score = await self._calculate_novelty_score(node)
        criteria_scores["novelty_score"] = novelty_score

        # Weighted combination
        weights = {
            "content_quality": 0.3,
            "depth_score": 0.2,
            "method_bonus": 0.3,
            "novelty_score": 0.2,
        }

        final_score = sum(
            criteria_scores[criterion] * weight for criterion, weight in weights.items()
        )

        # Store evaluation details
        node.metadata["evaluation_criteria"] = criteria_scores
        node.evaluation_score = final_score

        return final_score

    async def _calculate_novelty_score(self, node: ThoughtNode) -> float:
        """Calculate novelty score to avoid repetitive thinking"""

        # Compare with sibling nodes
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            sibling_contents = [
                self.nodes[child_id].content
                for child_id in parent.children_ids
                if child_id != node.id and child_id in self.nodes
            ]

            # Simple novelty check (in production would use semantic similarity)
            similar_count = sum(
                1
                for sibling_content in sibling_contents
                if any(
                    word in node.content.lower()
                    for word in sibling_content.lower().split()
                )
            )

            novelty_score = max(0.1, 1.0 - (similar_count * 0.2))
        else:
            novelty_score = 1.0  # Root node is always novel

        return novelty_score

    async def _prune_branches(self) -> int:
        """Prune branches with low evaluation scores"""
        pruned_count = 0

        nodes_to_prune = [
            node_id
            for node_id in self.active_branches
            if self.nodes[node_id].confidence < self.evaluation_threshold
        ]

        for node_id in nodes_to_prune:
            self.nodes[node_id].status = NodeStatus.PRUNED
            self.active_branches.discard(node_id)
            pruned_count += 1

        if pruned_count > 0:
            print(f"🔄 Pruned {pruned_count} low-scoring branches")

        return pruned_count

    async def _select_best_paths(self) -> List[Dict[str, Any]]:
        """Select the best reasoning paths from the tree"""
        all_paths = []

        # Find all leaf nodes
        leaf_nodes = [
            node
            for node in self.nodes.values()
            if not node.children_ids and node.status != NodeStatus.PRUNED
        ]

        # Trace paths from root to each leaf
        for leaf in leaf_nodes:
            path = await self._trace_path_to_root(leaf.id)
            path_score = await self._calculate_path_score(path)

            all_paths.append(
                {
                    "path": path,
                    "score": path_score,
                    "leaf_node": leaf.to_dict(),
                    "path_length": len(path),
                }
            )

        # Sort by score and return top paths
        all_paths.sort(key=lambda x: x["score"], reverse=True)

        # Select top 3 paths
        best_paths = all_paths[:3]

        return best_paths

    async def _trace_path_to_root(self, node_id: str) -> List[str]:
        """Trace path from given node back to root"""
        path = []
        current_id = node_id

        while current_id:
            path.append(current_id)
            current_node = self.nodes[current_id]
            current_id = current_node.parent_id

        return list(reversed(path))  # Root to leaf order

    async def _calculate_path_score(self, path: List[str]) -> float:
        """Calculate overall score for a reasoning path"""
        if not path:
            return 0.0

        # Average confidence of nodes in path
        confidences = [self.nodes[node_id].confidence for node_id in path]
        avg_confidence = sum(confidences) / len(confidences)

        # Bonus for path length (deeper exploration)
        length_bonus = min(len(path) / self.max_depth, 1.0) * 0.1

        # Bonus for method diversity
        methods = [self.nodes[node_id].reasoning_method for node_id in path]
        diversity_bonus = len(set(methods)) / len(methods) * 0.1

        total_score = avg_confidence + length_bonus + diversity_bonus

        return min(total_score, 1.0)

    async def _synthesize_solution(self, best_paths: List[Dict]) -> Dict[str, Any]:
        """Synthesize final solution from best paths"""

        if not best_paths:
            return {
                "content": "No viable solution paths found",
                "confidence": 0.0,
                "reasoning": "Tree exploration did not yield sufficient results",
            }

        # Get the best path
        best_path = best_paths[0]
        best_path_nodes = [self.nodes[node_id] for node_id in best_path["path"]]

        # Synthesize solution content
        solution_content = "Synthesized solution from Tree-of-Thought analysis:\n"

        for i, node in enumerate(best_path_nodes):
            if i == 0:  # Root node
                solution_content += (
                    f"Problem: {node.metadata.get('problem_statement', 'Unknown')}\n"
                )
            else:
                solution_content += f"Step {i}: {node.content}\n"

        # Calculate overall confidence
        overall_confidence = best_path["score"]

        # Generate reasoning explanation
        reasoning_methods = [
            node.reasoning_method for node in best_path_nodes[1:]
        ]  # Skip root
        reasoning_explanation = f"Applied methods: {', '.join(set(reasoning_methods))}"

        solution = {
            "content": solution_content,
            "confidence": overall_confidence,
            "reasoning": reasoning_explanation,
            "best_path_score": best_path["score"],
            "alternative_paths": len(best_paths) - 1,
            "synthesis_method": "tree_of_thought",
            "nodes_in_solution": len(best_path["path"]),
        }

        return solution

    def _get_tree_structure(self) -> Dict[str, Any]:
        """Get tree structure for visualization/debugging"""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id,
            "total_nodes": len(self.nodes),
            "max_depth": (
                max(node.depth for node in self.nodes.values()) if self.nodes else 0
            ),
            "status_distribution": self._get_status_distribution(),
        }

    def _get_status_distribution(self) -> Dict[str, int]:
        """Get distribution of node statuses"""
        distribution = {}
        for node in self.nodes.values():
            status = node.status.value
            distribution[status] = distribution.get(status, 0) + 1
        return distribution

    def _update_performance_metrics(self, total_time: float):
        """Update performance metrics"""
        self.performance_metrics = {
            "total_execution_time": total_time,
            "nodes_per_second": len(self.nodes) / total_time if total_time > 0 else 0,
            "avg_node_evaluation_time": (
                sum(node.execution_time for node in self.nodes.values())
                / len(self.nodes)
                if self.nodes
                else 0
            ),
            "exploration_efficiency": (
                len(
                    [n for n in self.nodes.values() if n.status == NodeStatus.COMPLETED]
                )
                / len(self.nodes)
                if self.nodes
                else 0
            ),
            "pruning_rate": (
                len([n for n in self.nodes.values() if n.status == NodeStatus.PRUNED])
                / len(self.nodes)
                if self.nodes
                else 0
            ),
        }

    def export_tree(self, filepath: str):
        """Export tree structure to JSON file"""
        tree_data = self._get_tree_structure()
        tree_data["metadata"] = {
            "export_timestamp": datetime.now().isoformat(),
            "planner_config": {
                "max_depth": self.max_depth,
                "max_branches_per_node": self.max_branches_per_node,
                "evaluation_threshold": self.evaluation_threshold,
                "enable_pruning": self.enable_pruning,
            },
        }

        with open(filepath, "w") as f:
            json.dump(tree_data, f, indent=2)

        print(f"💾 Tree structure exported to: {filepath}")


# Helper functions for Tree-of-Thought tools integration
def get_tree_of_thought_tools() -> Dict[str, Any]:
    """Get Tree-of-Thought tools for integration with AdvancedToolExecutor"""

    async def plan_with_tot(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for Tree-of-Thought planning"""
        planner = TreeOfThoughtPlanner(
            max_depth=params.get("max_depth", 5),
            max_branches_per_node=params.get("max_branches", 3),
            evaluation_threshold=params.get("threshold", 0.6),
            enable_pruning=params.get("enable_pruning", True),
        )

        result = await planner.plan_with_tree_of_thought(
            problem=params["problem"], context=params.get("context", {})
        )

        return result

    async def evaluate_thought_paths(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for evaluating multiple thought paths"""
        paths = params.get("paths", [])
        evaluation_results = []

        for i, path in enumerate(paths):
            # Mock evaluation - in production would use actual reasoning evaluation
            score = 0.7 + (i % 3) * 0.1  # Vary scores for demo
            evaluation_results.append(
                {
                    "path_id": i,
                    "path_content": path,
                    "evaluation_score": score,
                    "reasoning": f"Path {i} shows {'strong' if score > 0.8 else 'moderate'} reasoning quality",
                }
            )

        return {
            "evaluations": evaluation_results,
            "best_path": max(evaluation_results, key=lambda x: x["evaluation_score"]),
            "total_paths": len(paths),
        }

    return {
        "tree_of_thought_planning": {
            "function": plan_with_tot,
            "description": "Multi-path reasoning using Tree-of-Thought approach",
            "parameters": [
                "problem",
                "max_depth",
                "max_branches",
                "threshold",
                "context",
            ],
            "category": "planning",
        },
        "evaluate_thought_paths": {
            "function": evaluate_thought_paths,
            "description": "Evaluate and compare multiple reasoning paths",
            "parameters": ["paths"],
            "category": "evaluation",
        },
    }
