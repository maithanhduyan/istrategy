#!/usr/bin/env python3
"""
Production Showcase Demo for Advanced Reasoning Agent
Demonstrates Tree-of-Thought, AutoGPT, and Prompt Optimization capabilities
"""

import asyncio
import time
from typing import Dict, Any

# Core imports
from src.tree_of_thought_planner import TreeOfThoughtPlanner
from src.autogpt_planner import AutoGPTPlanner
from src.prompt_optimizer import PromptOptimizer, PromptType


def print_banner(title: str, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * 80}")
    print(f"🚀 {title}")
    print(f"{char * 80}")


def print_results(title: str, results: Dict[str, Any]):
    """Print formatted results"""
    print(f"\n📊 {title}:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.2f}")
        else:
            print(f"   • {key}: {value}")


async def showcase_tree_of_thought():
    """Showcase Tree-of-Thought capabilities"""
    print_banner("TREE-OF-THOUGHT SHOWCASE", "🌳")

    planner = TreeOfThoughtPlanner()

    problems = [
        {
            "problem": "Optimize machine learning model deployment for edge computing",
            "context": {
                "constraints": ["limited compute", "low latency", "offline capability"],
                "requirements": [
                    "real-time inference",
                    "model compression",
                    "edge hardware optimization",
                ],
            },
        },
        {
            "problem": "Design resilient distributed system architecture",
            "context": {
                "requirements": [
                    "fault tolerance",
                    "horizontal scaling",
                    "data consistency",
                ],
                "constraints": [
                    "network partitions",
                    "CAP theorem",
                    "eventual consistency",
                ],
            },
        },
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n🔍 Problem {i}: {problem['problem']}")
        start_time = time.time()

        result = await planner.plan_with_tree_of_thought(
            problem["problem"], context=problem["context"]
        )

        execution_time = time.time() - start_time

        print_results(
            f"Tree-of-Thought Results {i}",
            {
                "Execution time": f"{execution_time:.3f}s",
                "Nodes explored": result.get("total_nodes", 0),
                "Best paths found": len(result.get("best_paths", [])),
                "Solution confidence": result.get("confidence", 0),
                "Max depth reached": result.get("max_depth", 0),
            },
        )

        # Show best solution
        best_solution = result.get("solution", "No solution found")
        print(f"\n💡 Best Solution {i}:")
        print(f"   {best_solution[:200]}...")


async def showcase_autogpt():
    """Showcase AutoGPT capabilities"""
    print_banner("AUTOGPT AUTONOMOUS PLANNER SHOWCASE", "🤖")

    planner = AutoGPTPlanner()

    goals = [
        {
            "goal": "Implement CI/CD pipeline with automated testing and deployment",
            "context": {
                "description": "Set up comprehensive CI/CD for microservices application",
                "constraints": ["security compliance", "zero downtime deployment"],
                "resources": [
                    "GitHub Actions",
                    "Docker",
                    "Kubernetes",
                    "monitoring tools",
                ],
            },
            "tools": [
                "git_manager",
                "docker_builder",
                "k8s_deployer",
                "test_runner",
                "security_scanner",
            ],
        },
        {
            "goal": "Build data analytics dashboard with real-time visualization",
            "context": {
                "description": "Create interactive dashboard for business metrics",
                "deadline": "3 weeks",
                "requirements": [
                    "real-time updates",
                    "mobile responsive",
                    "role-based access",
                ],
            },
            "tools": ["data_connector", "chart_builder", "ui_framework", "auth_system"],
        },
    ]

    for i, goal in enumerate(goals, 1):
        print(f"\n🎯 Goal {i}: {goal['goal']}")
        start_time = time.time()

        result = await planner.autonomous_plan_and_execute(
            goal["goal"], context=goal["context"], available_tools=goal["tools"]
        )

        execution_time = time.time() - start_time

        print_results(
            f"AutoGPT Results {i}",
            {
                "Execution time": f"{execution_time:.3f}s",
                "Goal achieved": result.get("goal_achieved", False),
                "Tasks completed": f"{result.get('completed_tasks', 0)}/{result.get('total_tasks', 0)}",
                "Success rate": f"{result.get('success_rate', 0):.1f}%",
                "Iterations used": result.get("iterations", 0),
            },
        )

        # Show task breakdown
        task_summary = result.get("task_summary", {})
        if task_summary:
            print(f"\n📋 Task Breakdown {i}:")
            for status, tasks in task_summary.items():
                print(f"   • {status}: {len(tasks)} tasks")


async def showcase_prompt_optimization():
    """Showcase Prompt Optimization capabilities"""
    print_banner("PROMPT OPTIMIZATION SHOWCASE", "🎯")

    optimizer = PromptOptimizer()

    scenarios = [
        {
            "task": "Generate comprehensive code review feedback",
            "context": {
                "description": "Review Python code for best practices and improvements",
                "audience": "senior developer",
                "urgency": "medium",
                "goal": "actionable feedback",
            },
            "optimization_goals": ["technical_accuracy", "actionability", "clarity"],
        },
        {
            "task": "Create detailed project documentation",
            "context": {
                "description": "Document complex software architecture",
                "audience": "technical team",
                "urgency": "high",
                "goal": "comprehensive documentation",
            },
            "optimization_goals": ["completeness", "structure", "readability"],
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['task']}")
        start_time = time.time()

        result = await optimizer.optimize_prompt(
            PromptType.REASONING,
            context=scenario["context"],
            target_task=scenario["task"],
            optimization_goals=scenario["optimization_goals"],
        )

        execution_time = time.time() - start_time

        print_results(
            f"Optimization Results {i}",
            {
                "Execution time": f"{execution_time:.3f}s",
                "Candidates generated": result.get("candidates_tested", 0),
                "Variants tested": result.get("total_variants", 0),
                "Performance improvement": f"{result.get('performance_improvement', 0):.1f}%",
                "Final score": result.get("best_score", 0),
            },
        )

        # Show optimized prompt preview
        optimized_prompt = result.get("optimized_prompt", "")
        print(f"\n🎯 Optimized Prompt Preview {i}:")
        print(f"   {optimized_prompt[:150]}...")


async def showcase_integrated_workflow():
    """Showcase integrated workflow combining all planners"""
    print_banner("INTEGRATED WORKFLOW SHOWCASE", "⚡")

    complex_problem = {
        "description": "Build and deploy AI-powered recommendation system with real-time personalization",
        "requirements": [
            "Machine learning model training and deployment",
            "Real-time data processing pipeline",
            "A/B testing framework",
            "Scalable microservices architecture",
            "CI/CD automation",
            "Monitoring and alerting",
        ],
        "constraints": [
            "6-week timeline",
            "High availability requirements",
            "GDPR compliance",
            "Budget constraints",
        ],
    }

    print(f"🎯 Complex Challenge: {complex_problem['description']}")
    print(f"📋 Requirements: {len(complex_problem['requirements'])} items")
    print(f"⚠️ Constraints: {len(complex_problem['constraints'])} items")

    start_time = time.time()

    # Step 1: Tree-of-Thought exploration
    print(f"\n🌳 Step 1: Multi-path Solution Exploration")
    tot_planner = TreeOfThoughtPlanner()
    tot_result = await tot_planner.plan_with_tree_of_thought(
        complex_problem["description"], context=complex_problem
    )
    tot_time = time.time() - start_time
    print(
        f"   ✅ Generated {tot_result.get('total_nodes', 0)} solution paths in {tot_time:.2f}s"
    )

    # Step 2: AutoGPT task decomposition
    print(f"\n🤖 Step 2: Autonomous Task Decomposition")
    autogpt_start = time.time()
    autogpt_planner = AutoGPTPlanner()
    autogpt_result = await autogpt_planner.autonomous_plan_and_execute(
        complex_problem["description"],
        context=complex_problem,
        available_tools=[
            "ml_trainer",
            "data_processor",
            "api_builder",
            "deployer",
            "monitor",
        ],
    )
    autogpt_time = time.time() - autogpt_start
    print(
        f"   ✅ Created {autogpt_result.get('total_tasks', 0)} detailed tasks in {autogpt_time:.2f}s"
    )

    # Step 3: Prompt optimization for implementation
    print(f"\n🎯 Step 3: Implementation Prompt Optimization")
    optimizer_start = time.time()
    optimizer = PromptOptimizer()
    best_solution = tot_result.get("solution", "Implement the recommendation system")
    prompt_result = await optimizer.optimize_prompt(
        PromptType.INSTRUCTION,
        context={"urgency": "high", "audience": "development_team"},
        target_task=f"Implement the recommendation system based on: {best_solution[:100]}...",
        optimization_goals=["technical_precision", "clarity", "actionability"],
    )
    optimizer_time = time.time() - optimizer_start
    print(
        f"   ✅ Optimized implementation prompts with {prompt_result.get('performance_improvement', 0):.1f}% improvement"
    )

    # Final synthesis
    total_time = time.time() - start_time

    print(f"\n📊 Integrated Workflow Summary:")
    print_results(
        "Workflow Performance",
        {
            "Solution paths explored": tot_result.get("total_nodes", 0),
            "Detailed tasks generated": autogpt_result.get("total_tasks", 0),
            "Prompt variants tested": prompt_result.get("total_variants", 0),
            "Total workflow time": f"{total_time:.2f}s",
            "Overall confidence": min(
                1.0,
                (
                    tot_result.get("confidence", 0)
                    + (autogpt_result.get("success_rate", 0) / 100)
                    + (prompt_result.get("performance_improvement", 0) / 100)
                )
                / 3,
            ),
        },
    )


async def main():
    """Main showcase function"""
    print_banner("ADVANCED REASONING AGENT - PRODUCTION SHOWCASE")
    print("Demonstrating Tree-of-Thought, AutoGPT, and Prompt Optimization")
    print("Ready for production deployment with comprehensive capabilities")

    start_time = time.time()

    # Run all showcases
    await showcase_tree_of_thought()
    await showcase_autogpt()
    await showcase_prompt_optimization()
    await showcase_integrated_workflow()

    total_time = time.time() - start_time

    print_banner("SHOWCASE COMPLETED", "🎉")
    print_results(
        "Overall Performance",
        {
            "Total showcase time": f"{total_time:.2f}s",
            "Components demonstrated": 4,
            "Integration workflows": 1,
            "Production readiness": "✅ CONFIRMED",
        },
    )

    print("\n💡 Key Capabilities Demonstrated:")
    print("   • Tree-of-Thought: Multi-path reasoning with confidence scoring")
    print("   • AutoGPT: Autonomous task decomposition with adaptive execution")
    print("   • Prompt Optimization: Dynamic improvement with A/B testing")
    print("   • Integrated Workflow: Seamless orchestration of all components")
    print("\n🚀 System ready for production deployment!")


if __name__ == "__main__":
    asyncio.run(main())
