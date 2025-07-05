"""
AutoGPT-Style Planner for Autonomous Goal Decomposition
Implements autonomous planning, task decomposition, and execution management
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import uuid


class TaskStatus(Enum):
    """Status of individual tasks in AutoGPT planning"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for task execution"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Individual task in AutoGPT planning system"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_task_id: Optional[str] = None

    # Execution details
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    start_time: Optional[str] = None
    completion_time: Optional[str] = None

    # Tools and resources
    required_tools: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)

    # Results and feedback
    result: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    confidence: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "goal": self.goal,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "parent_task_id": self.parent_task_id,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "required_tools": self.required_tools,
            "required_resources": self.required_resources,
            "result": self.result,
            "feedback": self.feedback,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class AutoGPTPlannerConfig:
    """Configuration for AutoGPT planner"""

    def __init__(self):
        self.max_planning_depth = 5
        self.max_concurrent_tasks = 3
        self.task_timeout_minutes = 30
        self.auto_retry_failed_tasks = True
        self.max_retry_attempts = 2
        self.enable_adaptive_planning = True
        self.planning_horizon_hours = 24
        self.resource_optimization = True


class AutoGPTPlanner:
    """
    AutoGPT-style autonomous planner for goal decomposition and execution

    Key capabilities:
    1. Autonomous goal decomposition into actionable tasks
    2. Dynamic task scheduling and prioritization
    3. Dependency management and execution ordering
    4. Progress tracking and adaptive re-planning
    5. Resource allocation and optimization
    6. Failure handling and recovery strategies
    """

    def __init__(self, config: AutoGPTPlannerConfig = None):
        """Initialize AutoGPT planner with configuration"""
        self.config = config or AutoGPTPlannerConfig()

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.executing_tasks: Dict[str, asyncio.Task] = {}

        # Planning state
        self.current_goal: str = ""
        self.planning_history: List[Dict] = []
        self.execution_context: Dict[str, Any] = {}

        # Performance tracking
        self.metrics: Dict[str, Any] = {}
        self.feedback_history: List[Dict] = []

    async def autonomous_plan_and_execute(
        self,
        goal: str,
        context: Dict[str, Any] = None,
        available_tools: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Main AutoGPT planning and execution method

        Args:
            goal: High-level goal to achieve
            context: Additional context and constraints
            available_tools: List of available tools for execution

        Returns:
            Complete execution result with plan and outcomes
        """
        print(f"🤖 Starting AutoGPT planning for goal: {goal[:50]}...")

        start_time = time.time()

        self.current_goal = goal
        self.execution_context = context or {}

        # Phase 1: Goal Analysis and Decomposition
        print("📋 Phase 1: Goal Analysis and Decomposition")
        decomposition_result = await self._analyze_and_decompose_goal(
            goal, context, available_tools or []
        )

        # Phase 2: Task Planning and Scheduling
        print("📅 Phase 2: Task Planning and Scheduling")
        planning_result = await self._create_execution_plan()

        # Phase 3: Autonomous Execution
        print("⚡ Phase 3: Autonomous Execution")
        execution_result = await self._execute_plan_autonomously()

        # Phase 4: Results Analysis and Learning
        print("📊 Phase 4: Results Analysis")
        analysis_result = await self._analyze_execution_results()

        # Compile final results
        total_time = time.time() - start_time
        self._update_performance_metrics(total_time)

        final_result = {
            "goal": goal,
            "goal_achieved": analysis_result["goal_achieved"],
            "execution_summary": {
                "total_tasks": len(self.tasks),
                "completed_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
                ),
                "failed_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
                ),
                "execution_time": total_time,
                "success_rate": analysis_result["success_rate"],
            },
            "decomposition_result": decomposition_result,
            "planning_result": planning_result,
            "execution_result": execution_result,
            "analysis_result": analysis_result,
            "performance_metrics": self.metrics,
            "task_details": {
                task_id: task.to_dict() for task_id, task in self.tasks.items()
            },
        }

        print(f"✅ AutoGPT planning completed in {total_time:.2f}s")
        print(f"   • Goal achieved: {analysis_result['goal_achieved']}")
        print(
            f"   • Tasks completed: {final_result['execution_summary']['completed_tasks']}/{final_result['execution_summary']['total_tasks']}"
        )
        print(f"   • Success rate: {analysis_result['success_rate']:.1%}")

        return final_result

    async def _analyze_and_decompose_goal(
        self, goal: str, context: Dict[str, Any], available_tools: List[str]
    ) -> Dict[str, Any]:
        """Analyze goal and decompose into actionable tasks"""

        # Goal analysis
        goal_analysis = await self._analyze_goal_complexity(goal, context)

        # Generate initial task breakdown
        initial_tasks = await self._generate_initial_task_breakdown(
            goal, goal_analysis, available_tools
        )

        # Refine and optimize task structure
        refined_tasks = await self._refine_task_structure(initial_tasks)

        # Create task objects
        for task_data in refined_tasks:
            task = Task(
                title=task_data["title"],
                description=task_data["description"],
                goal=goal,
                priority=TaskPriority(task_data.get("priority", 2)),
                estimated_duration=task_data.get("estimated_duration", 5.0),
                required_tools=task_data.get("required_tools", []),
                dependencies=task_data.get("dependencies", []),
                metadata=task_data.get("metadata", {}),
            )
            self.tasks[task.id] = task

        # Establish task relationships
        await self._establish_task_relationships()

        return {
            "goal_analysis": goal_analysis,
            "initial_task_count": len(initial_tasks),
            "final_task_count": len(refined_tasks),
            "task_categories": self._categorize_tasks(),
            "complexity_score": goal_analysis["complexity_score"],
            "estimated_total_duration": sum(
                t.estimated_duration for t in self.tasks.values()
            ),
        }

    async def _analyze_goal_complexity(
        self, goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Analyze the complexity and requirements of the goal"""

        # Complexity indicators
        complexity_factors = {
            "word_count": len(goal.split()),
            "has_multiple_objectives": "and" in goal.lower() or "," in goal,
            "requires_external_data": any(
                word in goal.lower()
                for word in ["data", "information", "research", "analyze"]
            ),
            "requires_creativity": any(
                word in goal.lower()
                for word in ["create", "design", "generate", "invent"]
            ),
            "technical_complexity": any(
                word in goal.lower()
                for word in ["system", "algorithm", "code", "implement"]
            ),
            "has_constraints": any(
                word in goal.lower()
                for word in ["within", "under", "limit", "constraint"]
            ),
        }

        # Calculate complexity score
        complexity_score = sum(
            [
                complexity_factors["word_count"] / 10,  # Normalize word count
                2 if complexity_factors["has_multiple_objectives"] else 0,
                1.5 if complexity_factors["requires_external_data"] else 0,
                1.5 if complexity_factors["requires_creativity"] else 0,
                2 if complexity_factors["technical_complexity"] else 0,
                1 if complexity_factors["has_constraints"] else 0,
            ]
        )

        # Normalize to 0-10 scale
        complexity_score = min(complexity_score, 10)

        return {
            "complexity_factors": complexity_factors,
            "complexity_score": complexity_score,
            "estimated_planning_time": complexity_score * 2,  # minutes
            "recommended_approach": self._recommend_approach(complexity_score),
            "risk_factors": self._identify_risk_factors(goal, complexity_factors),
        }

    def _recommend_approach(self, complexity_score: float) -> str:
        """Recommend planning approach based on complexity"""
        if complexity_score <= 3:
            return "simple_sequential"
        elif complexity_score <= 6:
            return "structured_parallel"
        elif complexity_score <= 8:
            return "hierarchical_decomposition"
        else:
            return "iterative_refinement"

    def _identify_risk_factors(self, goal: str, complexity_factors: Dict) -> List[str]:
        """Identify potential risk factors in goal execution"""
        risks = []

        if complexity_factors["has_multiple_objectives"]:
            risks.append("Multiple objectives may conflict or compete for resources")

        if complexity_factors["requires_external_data"]:
            risks.append("External data dependencies may cause delays")

        if complexity_factors["technical_complexity"]:
            risks.append("Technical implementation may face unexpected challenges")

        if "urgent" in goal.lower() or "asap" in goal.lower():
            risks.append("Time pressure may impact quality")

        return risks

    async def _generate_initial_task_breakdown(
        self, goal: str, analysis: Dict, available_tools: List[str]
    ) -> List[Dict]:
        """Generate initial breakdown of goal into tasks"""

        tasks = []
        approach = analysis["recommended_approach"]

        if approach == "simple_sequential":
            tasks = await self._generate_simple_sequential_tasks(goal, available_tools)
        elif approach == "structured_parallel":
            tasks = await self._generate_structured_parallel_tasks(
                goal, available_tools
            )
        elif approach == "hierarchical_decomposition":
            tasks = await self._generate_hierarchical_tasks(goal, available_tools)
        else:  # iterative_refinement
            tasks = await self._generate_iterative_refinement_tasks(
                goal, available_tools
            )

        return tasks

    async def _generate_simple_sequential_tasks(
        self, goal: str, tools: List[str]
    ) -> List[Dict]:
        """Generate simple sequential task breakdown"""
        return [
            {
                "title": "Analyze Requirements",
                "description": f"Analyze and understand the requirements for: {goal}",
                "priority": 3,
                "estimated_duration": 3.0,
                "required_tools": ["analysis"] if "analysis" in tools else [],
                "dependencies": [],
            },
            {
                "title": "Plan Approach",
                "description": "Plan the approach and methodology",
                "priority": 3,
                "estimated_duration": 5.0,
                "required_tools": ["planning"] if "planning" in tools else [],
                "dependencies": [],
            },
            {
                "title": "Execute Implementation",
                "description": f"Implement the solution for: {goal}",
                "priority": 4,
                "estimated_duration": 10.0,
                "required_tools": tools,
                "dependencies": [],
            },
            {
                "title": "Validate Results",
                "description": "Validate and verify the results",
                "priority": 3,
                "estimated_duration": 3.0,
                "required_tools": ["validation"] if "validation" in tools else [],
                "dependencies": [],
            },
        ]

    async def _generate_structured_parallel_tasks(
        self, goal: str, tools: List[str]
    ) -> List[Dict]:
        """Generate structured parallel task breakdown"""
        return [
            {
                "title": "Research and Analysis",
                "description": f"Research and analyze requirements for: {goal}",
                "priority": 3,
                "estimated_duration": 4.0,
                "required_tools": ["research", "analysis"],
                "dependencies": [],
            },
            {
                "title": "Design Architecture",
                "description": "Design system architecture and approach",
                "priority": 3,
                "estimated_duration": 6.0,
                "required_tools": ["design"],
                "dependencies": [],
            },
            {
                "title": "Implement Core Components",
                "description": "Implement core system components",
                "priority": 4,
                "estimated_duration": 12.0,
                "required_tools": tools,
                "dependencies": [],
            },
            {
                "title": "Integration and Testing",
                "description": "Integrate components and conduct testing",
                "priority": 3,
                "estimated_duration": 6.0,
                "required_tools": ["testing", "integration"],
                "dependencies": [],
            },
            {
                "title": "Documentation and Deployment",
                "description": "Create documentation and deploy solution",
                "priority": 2,
                "estimated_duration": 4.0,
                "required_tools": ["documentation"],
                "dependencies": [],
            },
        ]

    async def _generate_hierarchical_tasks(
        self, goal: str, tools: List[str]
    ) -> List[Dict]:
        """Generate hierarchical task breakdown for complex goals"""
        # Similar to structured parallel but with more detailed subtasks
        base_tasks = await self._generate_structured_parallel_tasks(goal, tools)

        # Add more detailed subtasks for complex scenarios
        detailed_tasks = []
        for task in base_tasks:
            detailed_tasks.append(task)

            # Add subtasks for implementation
            if "Implement" in task["title"]:
                detailed_tasks.extend(
                    [
                        {
                            "title": f"Prepare Implementation Environment",
                            "description": "Set up development environment and dependencies",
                            "priority": 2,
                            "estimated_duration": 2.0,
                            "required_tools": ["setup"],
                            "dependencies": [],
                        },
                        {
                            "title": f"Implement Core Logic",
                            "description": "Implement the core business logic",
                            "priority": 4,
                            "estimated_duration": 8.0,
                            "required_tools": tools,
                            "dependencies": [],
                        },
                    ]
                )

        return detailed_tasks

    async def _generate_iterative_refinement_tasks(
        self, goal: str, tools: List[str]
    ) -> List[Dict]:
        """Generate iterative refinement tasks for very complex goals"""
        return [
            {
                "title": "Initial Problem Analysis",
                "description": f"Deep analysis of problem: {goal}",
                "priority": 4,
                "estimated_duration": 8.0,
                "required_tools": ["analysis", "research"],
                "dependencies": [],
            },
            {
                "title": "Prototype Development",
                "description": "Develop initial prototype",
                "priority": 3,
                "estimated_duration": 15.0,
                "required_tools": tools,
                "dependencies": [],
            },
            {
                "title": "Feedback Collection and Analysis",
                "description": "Collect feedback and analyze results",
                "priority": 3,
                "estimated_duration": 5.0,
                "required_tools": ["feedback", "analysis"],
                "dependencies": [],
            },
            {
                "title": "Refinement Iteration",
                "description": "Refine solution based on feedback",
                "priority": 4,
                "estimated_duration": 12.0,
                "required_tools": tools,
                "dependencies": [],
            },
            {
                "title": "Final Validation",
                "description": "Final validation and optimization",
                "priority": 3,
                "estimated_duration": 6.0,
                "required_tools": ["validation", "optimization"],
                "dependencies": [],
            },
        ]

    async def _refine_task_structure(self, initial_tasks: List[Dict]) -> List[Dict]:
        """Refine and optimize task structure"""
        refined_tasks = []

        for task in initial_tasks:
            # Optimize task duration estimates
            duration = task["estimated_duration"]
            if duration > 15:  # Break down large tasks
                # Split large task into smaller ones
                subtask_count = int(duration / 8) + 1
                subtask_duration = duration / subtask_count

                for i in range(subtask_count):
                    refined_tasks.append(
                        {
                            **task,
                            "title": f"{task['title']} - Part {i+1}",
                            "description": f"{task['description']} (Part {i+1} of {subtask_count})",
                            "estimated_duration": subtask_duration,
                        }
                    )
            else:
                refined_tasks.append(task)

        return refined_tasks

    async def _establish_task_relationships(self):
        """Establish dependencies and relationships between tasks"""
        task_list = list(self.tasks.values())

        # Simple dependency logic: sequential for now
        for i in range(1, len(task_list)):
            current_task = task_list[i]
            previous_task = task_list[i - 1]
            current_task.dependencies.append(previous_task.id)

    def _categorize_tasks(self) -> Dict[str, int]:
        """Categorize tasks by type"""
        categories = {
            "analysis": 0,
            "planning": 0,
            "implementation": 0,
            "testing": 0,
            "documentation": 0,
            "other": 0,
        }

        for task in self.tasks.values():
            title_lower = task.title.lower()
            if any(word in title_lower for word in ["analyze", "research", "study"]):
                categories["analysis"] += 1
            elif any(word in title_lower for word in ["plan", "design", "architect"]):
                categories["planning"] += 1
            elif any(
                word in title_lower
                for word in ["implement", "develop", "build", "create"]
            ):
                categories["implementation"] += 1
            elif any(word in title_lower for word in ["test", "validate", "verify"]):
                categories["testing"] += 1
            elif any(word in title_lower for word in ["document", "write", "report"]):
                categories["documentation"] += 1
            else:
                categories["other"] += 1

        return categories

    async def _create_execution_plan(self) -> Dict[str, Any]:
        """Create optimized execution plan"""

        # Topological sort for dependency ordering
        ordered_tasks = await self._topological_sort_tasks()

        # Optimize for parallel execution
        execution_layers = await self._create_execution_layers(ordered_tasks)

        # Resource allocation
        resource_allocation = await self._allocate_resources(execution_layers)

        # Risk assessment
        risk_assessment = await self._assess_execution_risks()

        return {
            "execution_order": [task.id for task in ordered_tasks],
            "execution_layers": execution_layers,
            "resource_allocation": resource_allocation,
            "risk_assessment": risk_assessment,
            "estimated_total_time": self._calculate_critical_path_duration(
                execution_layers
            ),
            "parallel_efficiency": (
                len(execution_layers) / len(self.tasks) if self.tasks else 0
            ),
        }

    async def _topological_sort_tasks(self) -> List[Task]:
        """Sort tasks based on dependencies using topological sort"""

        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []

        def visit(task_id: str):
            if task_id in temp_visited:
                # Circular dependency detected - break it
                return
            if task_id in visited:
                return

            temp_visited.add(task_id)

            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    visit(dep_id)

            temp_visited.remove(task_id)
            visited.add(task_id)
            result.append(task)

        # Visit all tasks
        for task_id in self.tasks:
            if task_id not in visited:
                visit(task_id)

        return list(reversed(result))

    async def _create_execution_layers(
        self, ordered_tasks: List[Task]
    ) -> List[List[str]]:
        """Create execution layers for parallel processing"""
        layers = []
        remaining_tasks = {task.id: task for task in ordered_tasks}
        completed_tasks = set()

        while remaining_tasks:
            # Find tasks that can be executed (all dependencies completed)
            ready_tasks = []
            for task_id, task in remaining_tasks.items():
                if all(dep_id in completed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task_id)

            if not ready_tasks:
                # Break circular dependencies
                ready_tasks.append(next(iter(remaining_tasks.keys())))

            # Limit concurrent tasks based on config
            layer = ready_tasks[: self.config.max_concurrent_tasks]
            layers.append(layer)

            # Mark as completed and remove from remaining
            for task_id in layer:
                completed_tasks.add(task_id)
                remaining_tasks.pop(task_id, None)

        return layers

    async def _allocate_resources(
        self, execution_layers: List[List[str]]
    ) -> Dict[str, Any]:
        """Allocate resources for task execution"""

        resource_usage = {}
        total_estimated_time = 0

        for layer_idx, layer in enumerate(execution_layers):
            layer_resources = []
            layer_duration = 0

            for task_id in layer:
                task = self.tasks[task_id]
                layer_resources.extend(task.required_tools)
                layer_duration = max(layer_duration, task.estimated_duration)

            resource_usage[f"layer_{layer_idx}"] = {
                "tasks": layer,
                "required_tools": list(set(layer_resources)),
                "estimated_duration": layer_duration,
                "concurrent_tasks": len(layer),
            }

            total_estimated_time += layer_duration

        return {
            "resource_usage_by_layer": resource_usage,
            "total_estimated_time": total_estimated_time,
            "peak_concurrent_tasks": max(len(layer) for layer in execution_layers),
            "unique_tools_needed": list(
                set(
                    tool for task in self.tasks.values() for tool in task.required_tools
                )
            ),
        }

    async def _assess_execution_risks(self) -> Dict[str, Any]:
        """Assess risks in execution plan"""

        risks = []

        # Dependency risks
        complex_dependencies = [
            task for task in self.tasks.values() if len(task.dependencies) > 2
        ]
        if complex_dependencies:
            risks.append(
                {
                    "type": "complex_dependencies",
                    "severity": "medium",
                    "description": f"{len(complex_dependencies)} tasks have complex dependencies",
                    "mitigation": "Monitor dependency completion closely",
                }
            )

        # Duration risks
        long_tasks = [
            task for task in self.tasks.values() if task.estimated_duration > 10
        ]
        if long_tasks:
            risks.append(
                {
                    "type": "long_duration_tasks",
                    "severity": "medium",
                    "description": f"{len(long_tasks)} tasks have long estimated durations",
                    "mitigation": "Consider breaking down long tasks further",
                }
            )

        # Resource conflicts
        all_tools = [
            tool for task in self.tasks.values() for tool in task.required_tools
        ]
        if len(all_tools) != len(set(all_tools)):
            risks.append(
                {
                    "type": "resource_contention",
                    "severity": "low",
                    "description": "Some tools are required by multiple tasks",
                    "mitigation": "Implement resource scheduling",
                }
            )

        return {
            "risks": risks,
            "overall_risk_level": max(
                [r.get("severity", "low") for r in risks], default="low"
            ),
            "total_risk_count": len(risks),
        }

    def _calculate_critical_path_duration(
        self, execution_layers: List[List[str]]
    ) -> float:
        """Calculate critical path duration for execution plan"""
        total_duration = 0

        for layer in execution_layers:
            layer_duration = (
                max(self.tasks[task_id].estimated_duration for task_id in layer)
                if layer
                else 0
            )
            total_duration += layer_duration

        return total_duration

    async def _execute_plan_autonomously(self) -> Dict[str, Any]:
        """Execute the plan autonomously with monitoring and adaptation"""

        execution_start = time.time()
        completed_tasks = []
        failed_tasks = []

        # Get execution layers
        ordered_tasks = await self._topological_sort_tasks()
        execution_layers = await self._create_execution_layers(ordered_tasks)

        print(
            f"🚀 Executing {len(execution_layers)} layers with {len(self.tasks)} total tasks"
        )

        # Execute layer by layer
        for layer_idx, layer in enumerate(execution_layers):
            print(
                f"⚡ Executing layer {layer_idx + 1}/{len(execution_layers)} ({len(layer)} tasks)"
            )

            # Execute tasks in current layer concurrently
            layer_results = await self._execute_task_layer(layer)

            # Process results
            for task_id, result in layer_results.items():
                if result["success"]:
                    completed_tasks.append(task_id)
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].result = result
                    self.tasks[task_id].completion_time = datetime.now().isoformat()
                else:
                    failed_tasks.append(task_id)
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].feedback = result.get("error", "Unknown error")

            # Adaptive re-planning if needed
            if failed_tasks and self.config.enable_adaptive_planning:
                await self._handle_execution_failures(failed_tasks)

        execution_time = time.time() - execution_start

        return {
            "execution_time": execution_time,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": len(completed_tasks) / len(self.tasks) if self.tasks else 0,
            "layers_executed": len(execution_layers),
            "adaptation_events": len(self.feedback_history),
        }

    async def _execute_task_layer(self, layer: List[str]) -> Dict[str, Dict]:
        """Execute a layer of tasks concurrently"""

        # Start all tasks in layer
        task_coroutines = {}
        for task_id in layer:
            task_coroutines[task_id] = self._execute_single_task(task_id)

        # Wait for all tasks to complete
        results = {}
        for task_id, coro in task_coroutines.items():
            try:
                result = await asyncio.wait_for(
                    coro, timeout=self.config.task_timeout_minutes * 60
                )
                results[task_id] = result
            except asyncio.TimeoutError:
                results[task_id] = {
                    "success": False,
                    "error": "Task execution timeout",
                    "timeout": True,
                }
            except Exception as e:
                results[task_id] = {
                    "success": False,
                    "error": str(e),
                    "exception": True,
                }

        return results

    async def _execute_single_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a single task"""

        task = self.tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.start_time = datetime.now().isoformat()

        start_time = time.time()

        try:
            # Mock task execution - in production would call actual tools
            await asyncio.sleep(
                min(task.estimated_duration / 10, 2.0)
            )  # Simulated work

            # Simulate success/failure based on task complexity
            success_probability = max(0.7, 1.0 - (task.estimated_duration / 20))
            import random

            success = random.random() < success_probability

            execution_time = time.time() - start_time
            task.actual_duration = execution_time

            if success:
                return {
                    "success": True,
                    "result": f"Task '{task.title}' completed successfully",
                    "execution_time": execution_time,
                    "confidence": random.uniform(0.7, 0.95),
                }
            else:
                return {
                    "success": False,
                    "error": f"Task '{task.title}' failed during execution",
                    "execution_time": execution_time,
                }

        except Exception as e:
            execution_time = time.time() - start_time
            task.actual_duration = execution_time

            return {"success": False, "error": str(e), "execution_time": execution_time}

    async def _handle_execution_failures(self, failed_task_ids: List[str]):
        """Handle execution failures with adaptive re-planning"""

        print(f"🔄 Handling {len(failed_task_ids)} failed tasks")

        for task_id in failed_task_ids:
            task = self.tasks[task_id]

            # Retry logic
            if self.config.auto_retry_failed_tasks:
                retry_count = task.metadata.get("retry_count", 0)
                if retry_count < self.config.max_retry_attempts:
                    task.metadata["retry_count"] = retry_count + 1
                    task.status = TaskStatus.PENDING
                    print(f"   • Scheduling retry for task: {task.title}")
                    continue

            # Alternative task generation
            alternative_tasks = await self._generate_alternative_tasks(task)
            for alt_task in alternative_tasks:
                self.tasks[alt_task.id] = alt_task
                print(f"   • Generated alternative task: {alt_task.title}")

        # Record adaptation event
        self.feedback_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "failure_handling",
                "failed_tasks": failed_task_ids,
                "adaptations_made": len(failed_task_ids),
            }
        )

    async def _generate_alternative_tasks(self, failed_task: Task) -> List[Task]:
        """Generate alternative tasks for failed ones"""

        alternatives = []

        # Simpler alternative with reduced scope
        simple_alt = Task(
            title=f"Simplified: {failed_task.title}",
            description=f"Simplified version of: {failed_task.description}",
            goal=failed_task.goal,
            priority=TaskPriority.MEDIUM,
            estimated_duration=failed_task.estimated_duration * 0.6,
            required_tools=failed_task.required_tools[:1],  # Use fewer tools
            metadata={"alternative_for": failed_task.id, "approach": "simplified"},
        )
        alternatives.append(simple_alt)

        return alternatives

    async def _analyze_execution_results(self) -> Dict[str, Any]:
        """Analyze execution results and extract insights"""

        completed_count = len(
            [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        )
        total_count = len(self.tasks)
        success_rate = completed_count / total_count if total_count > 0 else 0

        # Goal achievement assessment
        goal_achieved = success_rate >= 0.8  # 80% task completion threshold

        # Performance analysis
        total_estimated = sum(t.estimated_duration for t in self.tasks.values())
        total_actual = sum(
            t.actual_duration for t in self.tasks.values() if t.actual_duration > 0
        )

        duration_accuracy = (
            1.0 - abs(total_estimated - total_actual) / total_estimated
            if total_estimated > 0
            else 0
        )

        # Learning insights
        insights = []

        if success_rate < 0.7:
            insights.append("Consider breaking down complex tasks further")
        if duration_accuracy < 0.7:
            insights.append("Improve duration estimation accuracy")
        if len(self.feedback_history) > 2:
            insights.append(
                "High adaptation frequency suggests planning improvements needed"
            )

        return {
            "goal_achieved": goal_achieved,
            "success_rate": success_rate,
            "duration_accuracy": duration_accuracy,
            "total_estimated_time": total_estimated,
            "total_actual_time": total_actual,
            "efficiency_ratio": (
                total_estimated / total_actual if total_actual > 0 else 0
            ),
            "learning_insights": insights,
            "adaptation_count": len(self.feedback_history),
        }

    def _update_performance_metrics(self, total_time: float):
        """Update performance metrics"""
        self.metrics = {
            "total_planning_time": total_time,
            "tasks_per_minute": len(self.tasks) / total_time if total_time > 0 else 0,
            "planning_efficiency": (
                len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
                )
                / len(self.tasks)
                if self.tasks
                else 0
            ),
            "adaptation_rate": (
                len(self.feedback_history) / len(self.tasks) if self.tasks else 0
            ),
            "average_task_duration": sum(
                t.actual_duration for t in self.tasks.values() if t.actual_duration > 0
            )
            / max(1, len([t for t in self.tasks.values() if t.actual_duration > 0])),
        }


# Helper functions for AutoGPT tools integration
def get_autogpt_planner_tools() -> Dict[str, Any]:
    """Get AutoGPT planner tools for integration"""

    async def autonomous_plan_execute(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for AutoGPT autonomous planning"""
        config = AutoGPTPlannerConfig()

        # Apply configuration overrides
        if "max_depth" in params:
            config.max_planning_depth = params["max_depth"]
        if "max_concurrent" in params:
            config.max_concurrent_tasks = params["max_concurrent"]
        if "timeout_minutes" in params:
            config.task_timeout_minutes = params["timeout_minutes"]

        planner = AutoGPTPlanner(config)

        result = await planner.autonomous_plan_and_execute(
            goal=params["goal"],
            context=params.get("context", {}),
            available_tools=params.get("available_tools", []),
        )

        return result

    async def analyze_goal_complexity(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for analyzing goal complexity"""
        planner = AutoGPTPlanner()
        analysis = await planner._analyze_goal_complexity(
            params["goal"], params.get("context", {})
        )
        return analysis

    return {
        "autonomous_plan_execute": {
            "function": autonomous_plan_execute,
            "description": "Autonomous goal decomposition and execution using AutoGPT approach",
            "parameters": [
                "goal",
                "context",
                "available_tools",
                "max_depth",
                "max_concurrent",
                "timeout_minutes",
            ],
            "category": "planning",
        },
        "analyze_goal_complexity": {
            "function": analyze_goal_complexity,
            "description": "Analyze complexity and requirements of a goal",
            "parameters": ["goal", "context"],
            "category": "analysis",
        },
    }
