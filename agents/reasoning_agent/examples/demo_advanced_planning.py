#!/usr/bin/env python3
"""
Comprehensive Demo: Tree-of-Thought, AutoGPT, and Prompt Optimization
Demonstrates advanced planning and optimization capabilities
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import time
from src.tree_of_thought_planner import TreeOfThoughtPlanner
from src.autogpt_planner import AutoGPTPlanner, AutoGPTPlannerConfig
from src.prompt_optimizer import PromptOptimizer, PromptType
from src.advanced_tools import AdvancedToolExecutor


async def demo_tree_of_thought_planning():
    """Demo Tree-of-Thought multi-path reasoning"""
    print("\n🌳 TREE-OF-THOUGHT PLANNING DEMO")
    print("=" * 80)

    planner = TreeOfThoughtPlanner(
        max_depth=4,
        max_branches_per_node=3,
        evaluation_threshold=0.7,
        enable_pruning=True,
    )

    problem = "Design a secure and scalable authentication system for a microservices architecture"
    context = {
        "requirements": ["security", "scalability", "performance"],
        "constraints": ["budget", "time"],
        "existing_tech": ["JWT", "OAuth2", "Docker"],
    }

    print(f"Problem: {problem}")
    print(f"Context: {context}")

    start_time = time.time()
    result = await planner.plan_with_tree_of_thought(problem, context)
    execution_time = time.time() - start_time

    print(f"\n📊 Tree-of-Thought Results:")
    print(f"   • Execution time: {execution_time:.2f}s")
    print(f"   • Total nodes explored: {result['total_nodes']}")
    print(f"   • Max depth reached: {result['max_depth_reached']}")
    print(f"   • Best paths found: {len(result['best_paths'])}")
    print(f"   • Solution confidence: {result['final_solution']['confidence']:.2f}")

    print(f"\n💡 Best Solution:")
    print(f"   {result['final_solution']['content'][:200]}...")

    return result


async def demo_autogpt_planning():
    """Demo AutoGPT autonomous goal decomposition"""
    print("\n🤖 AUTOGPT AUTONOMOUS PLANNING DEMO")
    print("=" * 80)

    config = AutoGPTPlannerConfig()
    config.max_concurrent_tasks = 2
    config.task_timeout_minutes = 5
    config.enable_adaptive_planning = True

    planner = AutoGPTPlanner(config)

    goal = "Create a comprehensive API documentation system with automated testing and deployment"
    context = {
        "description": "Need to document existing REST APIs, set up automated testing, and create deployment pipeline",
        "deadline": "2 weeks",
        "resources": ["development team", "CI/CD tools", "documentation platform"],
    }
    available_tools = [
        "api_analyzer",
        "doc_generator",
        "test_runner",
        "deployment_pipeline",
    ]

    print(f"Goal: {goal}")
    print(f"Context: {context}")
    print(f"Available tools: {available_tools}")

    start_time = time.time()
    result = await planner.autonomous_plan_and_execute(goal, context, available_tools)
    execution_time = time.time() - start_time

    print(f"\n📊 AutoGPT Results:")
    print(f"   • Execution time: {execution_time:.2f}s")
    print(f"   • Goal achieved: {result['goal_achieved']}")
    print(
        f"   • Tasks completed: {result['execution_summary']['completed_tasks']}/{result['execution_summary']['total_tasks']}"
    )
    print(f"   • Success rate: {result['execution_summary']['success_rate']:.1%}")

    print(f"\n📋 Task Breakdown:")
    for task_id, task_data in list(result["task_details"].items())[
        :3
    ]:  # Show first 3 tasks
        print(f"   • {task_data['title']}: {task_data['status']}")

    return result


async def demo_prompt_optimization():
    """Demo intelligent prompt optimization"""
    print("\n🎯 PROMPT OPTIMIZATION DEMO")
    print("=" * 80)

    optimizer = PromptOptimizer()

    target_task = "Analyze the security vulnerabilities in a web application and provide remediation strategies"
    context = {
        "description": "Security audit of e-commerce web application",
        "audience": "technical",
        "urgency": "high",
        "goal": "comprehensive security analysis",
    }
    optimization_goals = ["accuracy", "detail", "clarity"]

    print(f"Task: {target_task}")
    print(f"Context: {context}")
    print(f"Optimization goals: {optimization_goals}")

    start_time = time.time()
    result = await optimizer.optimize_prompt(
        prompt_type=PromptType.REASONING,
        context=context,
        target_task=target_task,
        optimization_goals=optimization_goals,
    )
    execution_time = time.time() - start_time

    print(f"\n📊 Optimization Results:")
    print(f"   • Execution time: {execution_time:.2f}s")
    print(
        f"   • Candidates generated: {result['optimization_summary']['candidates_generated']}"
    )
    print(f"   • Variants tested: {result['optimization_summary']['variants_tested']}")
    print(
        f"   • Performance improvement: {result['optimization_summary']['performance_improvement']:.1%}"
    )

    print(f"\n🎯 Optimized Prompt:")
    print(f"   {result['optimized_prompt'][:300]}...")

    print(f"\n💡 Recommendations:")
    for rec in result["recommendations"][:2]:
        print(f"   • {rec}")

    return result


async def demo_integrated_workflow():
    """Demo integrated workflow using all advanced planning tools"""
    print("\n⚡ INTEGRATED ADVANCED PLANNING WORKFLOW")
    print("=" * 80)

    # Complex engineering problem
    problem = "Design and implement a real-time data processing pipeline for IoT sensors with ML-based anomaly detection"

    print(f"Complex Problem: {problem}")

    # Step 1: Tree-of-Thought for solution exploration
    print("\n🌳 Step 1: Multi-path Solution Exploration")
    tot_planner = TreeOfThoughtPlanner(max_depth=3, max_branches_per_node=2)
    tot_result = await tot_planner.plan_with_tree_of_thought(
        problem, {"domain": "IoT data processing", "complexity": "high"}
    )

    # Extract insights from Tree-of-Thought
    tot_insights = tot_result["final_solution"]["content"]
    print(f"   ✅ Generated {tot_result['total_nodes']} solution paths")

    # Step 2: AutoGPT for detailed task planning
    print("\n🤖 Step 2: Autonomous Task Decomposition")
    autogpt_planner = AutoGPTPlanner()

    # Use Tree-of-Thought insights to enhance AutoGPT context
    enhanced_context = {
        "tot_insights": tot_insights[:200],
        "solution_approach": "data pipeline with ML anomaly detection",
        "complexity": "high",
        "components": ["data ingestion", "processing engine", "ML model", "alerting"],
    }

    autogpt_result = await autogpt_planner.autonomous_plan_and_execute(
        problem,
        enhanced_context,
        ["data_processor", "ml_trainer", "pipeline_deployer", "monitor"],
    )
    print(
        f"   ✅ Created {autogpt_result['execution_summary']['total_tasks']} detailed tasks"
    )

    # Step 3: Prompt optimization for implementation guidance
    print("\n🎯 Step 3: Implementation Prompt Optimization")
    optimizer = PromptOptimizer()

    implementation_task = (
        f"Implement the data processing pipeline based on: {tot_insights[:100]}"
    )
    opt_context = {
        "task_breakdown": autogpt_result["execution_summary"],
        "technical_complexity": "very_high",
        "audience": "senior_engineers",
    }

    prompt_result = await optimizer.optimize_prompt(
        PromptType.INSTRUCTION,
        opt_context,
        implementation_task,
        ["accuracy", "detail", "clarity"],
    )
    print(
        f"   ✅ Optimized implementation prompts with {prompt_result['optimization_summary']['performance_improvement']:.1%} improvement"
    )

    # Step 4: Synthesis and recommendations
    print("\n📊 Step 4: Workflow Synthesis")

    workflow_summary = {
        "solution_exploration": {
            "method": "Tree-of-Thought",
            "paths_explored": tot_result["total_nodes"],
            "best_solution_confidence": tot_result["final_solution"]["confidence"],
        },
        "task_planning": {
            "method": "AutoGPT",
            "tasks_generated": autogpt_result["execution_summary"]["total_tasks"],
            "success_rate": autogpt_result["execution_summary"]["success_rate"],
        },
        "prompt_optimization": {
            "method": "Dynamic Optimization",
            "candidates_tested": prompt_result["optimization_summary"][
                "variants_tested"
            ],
            "performance_gain": prompt_result["optimization_summary"][
                "performance_improvement"
            ],
        },
        "overall_confidence": (
            tot_result["final_solution"]["confidence"]
            + autogpt_result["execution_summary"]["success_rate"]
            + (1 + prompt_result["optimization_summary"]["performance_improvement"])
        )
        / 3,
    }

    print(f"\n🎉 Integrated Workflow Summary:")
    print(
        f"   • Solution paths explored: {workflow_summary['solution_exploration']['paths_explored']}"
    )
    print(
        f"   • Detailed tasks generated: {workflow_summary['task_planning']['tasks_generated']}"
    )
    print(
        f"   • Prompt variants tested: {workflow_summary['prompt_optimization']['candidates_tested']}"
    )
    print(
        f"   • Overall workflow confidence: {workflow_summary['overall_confidence']:.2f}"
    )

    return workflow_summary


async def demo_performance_comparison():
    """Demo performance comparison of different planning approaches"""
    print("\n📈 PERFORMANCE COMPARISON DEMO")
    print("=" * 80)

    test_problem = "Optimize database performance for high-traffic web application"

    results = {}

    # Test Tree-of-Thought
    print("Testing Tree-of-Thought approach...")
    start_time = time.time()
    tot_planner = TreeOfThoughtPlanner(max_depth=3, max_branches_per_node=2)
    tot_result = await tot_planner.plan_with_tree_of_thought(test_problem)
    results["tree_of_thought"] = {
        "execution_time": time.time() - start_time,
        "solution_quality": tot_result["final_solution"]["confidence"],
        "exploration_depth": tot_result["max_depth_reached"],
        "paths_evaluated": tot_result["total_nodes"],
    }

    # Test AutoGPT
    print("Testing AutoGPT approach...")
    start_time = time.time()
    autogpt_planner = AutoGPTPlanner()
    autogpt_result = await autogpt_planner.autonomous_plan_and_execute(test_problem)
    results["autogpt"] = {
        "execution_time": time.time() - start_time,
        "solution_quality": autogpt_result["execution_summary"]["success_rate"],
        "tasks_generated": autogpt_result["execution_summary"]["total_tasks"],
        "goal_achievement": autogpt_result["goal_achieved"],
    }

    # Test Prompt Optimization
    print("Testing Prompt Optimization approach...")
    start_time = time.time()
    optimizer = PromptOptimizer()
    prompt_result = await optimizer.optimize_prompt(
        PromptType.REASONING, {}, test_problem, ["accuracy"]
    )
    results["prompt_optimization"] = {
        "execution_time": time.time() - start_time,
        "optimization_improvement": prompt_result["optimization_summary"][
            "performance_improvement"
        ],
        "variants_tested": prompt_result["optimization_summary"]["variants_tested"],
    }

    # Performance analysis
    print(f"\n📊 Performance Comparison Results:")
    print(f"   Tree-of-Thought:")
    print(f"     • Time: {results['tree_of_thought']['execution_time']:.2f}s")
    print(f"     • Quality: {results['tree_of_thought']['solution_quality']:.2f}")
    print(f"     • Paths: {results['tree_of_thought']['paths_evaluated']}")

    print(f"   AutoGPT:")
    print(f"     • Time: {results['autogpt']['execution_time']:.2f}s")
    print(f"     • Quality: {results['autogpt']['solution_quality']:.2f}")
    print(f"     • Tasks: {results['autogpt']['tasks_generated']}")

    print(f"   Prompt Optimization:")
    print(f"     • Time: {results['prompt_optimization']['execution_time']:.2f}s")
    print(
        f"     • Improvement: {results['prompt_optimization']['optimization_improvement']:.1%}"
    )
    print(f"     • Variants: {results['prompt_optimization']['variants_tested']}")

    return results


async def main():
    """Main demo execution"""
    print("🚀 ADVANCED PLANNING & OPTIMIZATION COMPREHENSIVE DEMO")
    print("=" * 80)
    print("Demonstrating Tree-of-Thought, AutoGPT, and Prompt Optimization")
    print("=" * 80)

    total_start_time = time.time()

    try:
        # Individual demos
        tot_result = await demo_tree_of_thought_planning()
        autogpt_result = await demo_autogpt_planning()
        prompt_result = await demo_prompt_optimization()

        # Integration demo
        workflow_result = await demo_integrated_workflow()

        # Performance comparison
        performance_result = await demo_performance_comparison()

        total_time = time.time() - total_start_time

        # Final summary
        print(f"\n🎉 COMPREHENSIVE DEMO COMPLETED")
        print("=" * 80)
        print(f"   • Total execution time: {total_time:.2f}s")
        print(f"   • Tree-of-Thought nodes: {tot_result['total_nodes']}")
        print(
            f"   • AutoGPT tasks: {autogpt_result['execution_summary']['total_tasks']}"
        )
        print(
            f"   • Prompt variants: {prompt_result['optimization_summary']['variants_tested']}"
        )
        print(
            f"   • Integrated workflow confidence: {workflow_result['overall_confidence']:.2f}"
        )

        print(
            f"\n✅ All advanced planning and optimization tools working successfully!"
        )

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
