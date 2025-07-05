#!/usr/bin/env python3
"""
Advanced Reasoning Agent Demo
Showcases RAG, Thinking, and Inference capabilities
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ReasoningAgent
from src.advanced_tools import AdvancedToolExecutor
import asyncio
import time


def demo_advanced_capabilities():
    """Demo comprehensive advanced reasoning capabilities"""

    print("=" * 80)
    print("🚀 ADVANCED REASONING AGENT DEMONSTRATION")
    print("🧠 RAG | Thinking | Inference | Pattern Analysis")
    print("=" * 80)

    # Initialize advanced tool executor for testing
    print("\n📋 1. ADVANCED TOOLS INITIALIZATION")
    print("-" * 50)

    try:
        advanced_executor = AdvancedToolExecutor(enable_async=True)
        print("✅ Advanced Tool Executor initialized successfully")

        capabilities = advanced_executor.get_capabilities_summary()
        print(f"📊 Capabilities Summary:")
        print(f"   • Total Tools: {capabilities['total_tools']}")
        print(f"   • Advanced Tools: {capabilities['advanced_tools']}")
        print(f"   • RAG Tools: {capabilities['categories']['rag_capabilities']}")
        print(
            f"   • Thinking Tools: {capabilities['categories']['thinking_capabilities']}"
        )
        print(
            f"   • Inference Tools: {capabilities['categories']['inference_capabilities']}"
        )
        print(f"   • Async Support: {capabilities['async_support']}")

    except Exception as e:
        print(f"❌ Failed to initialize advanced tools: {e}")
        return False

    # Test each category
    print("\n📋 2. TOOL CATEGORY DEMONSTRATIONS")
    print("-" * 50)

    # Demo inference tools
    demo_inference_tools(advanced_executor)

    # Demo thinking tools (simulated)
    demo_thinking_tools(advanced_executor)

    # Demo pattern analysis
    demo_pattern_analysis(advanced_executor)

    # Demo RAG capabilities (simulated)
    demo_rag_capabilities(advanced_executor)

    return True


def demo_inference_tools(executor: AdvancedToolExecutor):
    """Demonstrate logical inference capabilities"""
    print("\n🧮 LOGICAL INFERENCE DEMONSTRATION")
    print("-" * 40)

    # Add some logical rules and facts
    test_cases = [
        ("logical_add_fact", ["All humans are mortal", "1.0"]),
        ("logical_add_fact", ["Socrates is human", "1.0"]),
        ("logical_add_rule", ["X is human", "X is mortal", "1.0"]),
        ("logical_infer", ["forward"]),
        ("inference_status", []),
    ]

    for action, args in test_cases:
        print(f"\n🔧 Testing: {action}({', '.join(args)})")
        result = executor.execute(action, args)
        print(f"   Result: {result}")


def demo_thinking_tools(executor: AdvancedToolExecutor):
    """Demonstrate thinking capabilities"""
    print("\n🤔 THINKING TOOLS DEMONSTRATION")
    print("-" * 40)

    test_cases = [
        ("think_sequential", ["How to build a sustainable business", "3"]),
        (
            "think_critical",
            ["AI will replace all jobs", "automation increases", "new jobs created"],
        ),
        ("think_systems", ["Ecosystem", "predators", "prey", "plants", "climate"]),
        ("get_thinking_summary", []),
    ]

    for action, args in test_cases:
        print(f"\n🔧 Testing: {action}({', '.join(args)})")
        try:
            result = executor.execute(action, args)
            print(f"   Result: {result}")
        except Exception as e:
            print(f"   Error: {e}")


def demo_pattern_analysis(executor: AdvancedToolExecutor):
    """Demonstrate pattern recognition"""
    print("\n📊 PATTERN ANALYSIS DEMONSTRATION")
    print("-" * 40)

    test_cases = [
        ("pattern_analyze_numeric", ["1", "4", "9", "16", "25"]),  # Square numbers
        ("pattern_analyze_numeric", ["2", "4", "6", "8", "10"]),  # Even numbers
        (
            "pattern_analyze_text",
            ["file1.txt", "file2.txt", "file3.txt"],
        ),  # File pattern
        ("pattern_analyze_text", ["abc123", "def456", "ghi789"]),  # Mixed pattern
    ]

    for action, args in test_cases:
        print(f"\n🔧 Testing: {action}({', '.join(args)})")
        result = executor.execute(action, args)
        print(f"   Result: {result}")


def demo_rag_capabilities(executor: AdvancedToolExecutor):
    """Demonstrate RAG capabilities"""
    print("\n📚 RAG (Knowledge) DEMONSTRATION")
    print("-" * 40)

    test_cases = [
        (
            "rag_add_knowledge",
            ["Python is a programming language known for its simplicity", "demo"],
        ),
        (
            "rag_add_knowledge",
            ["Machine learning uses algorithms to find patterns in data", "demo"],
        ),
        ("rag_search", ["programming language", "2"]),
        ("rag_augmented_query", ["What is machine learning?"]),
    ]

    for action, args in test_cases:
        print(f"\n🔧 Testing: {action}({', '.join(args)})")
        try:
            result = executor.execute(action, args)
            print(f"   Result: {result}")
        except Exception as e:
            print(f"   Error: {e}")


def demo_advanced_agent():
    """Demonstrate advanced reasoning agent"""
    print("\n" + "=" * 80)
    print("🤖 ADVANCED REASONING AGENT IN ACTION")
    print("=" * 80)

    print("\n🚀 Initializing Advanced Reasoning Agent...")
    agent = ReasoningAgent(backend="auto", use_advanced_tools=True)

    if not agent.ai_client:
        print("❌ Failed to initialize AI client")
        return False

    # Test complex reasoning problems
    complex_problems = [
        "Analyze the pattern in this sequence: 1, 1, 2, 3, 5, 8, 13",
        "Use logical reasoning: If all birds can fly, and penguins are birds, but penguins cannot fly, what does this tell us?",
        "Think systematically about the components of a sustainable transportation system",
    ]

    for i, problem in enumerate(complex_problems, 1):
        print(f"\n📋 Complex Problem {i}:")
        print(f"Question: {problem}")
        print("-" * 60)

        try:
            # Analyze problem complexity first
            if hasattr(agent.tool_executor, "analyze_problem_complexity"):
                analysis = agent.tool_executor.analyze_problem_complexity(problem)
                print(f"🔍 Complexity Analysis:")
                print(f"   • Level: {analysis['complexity_level']}")
                print(
                    f"   • Suggested Tools: {', '.join(analysis['suggested_tools'][:3])}"
                )
                print(f"   • Estimated Steps: {analysis['estimated_steps']}")

            print(f"\n🤖 Agent Response:")
            answer = agent.solve(problem)
            print(f"✅ Answer: {answer}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print()

    return True


def demo_workflow_execution():
    """Demonstrate workflow execution capabilities"""
    print("\n" + "=" * 80)
    print("⚙️ WORKFLOW EXECUTION DEMONSTRATION")
    print("=" * 80)

    try:
        executor = AdvancedToolExecutor()

        # Define a multi-step workflow
        workflow = [
            {
                "name": "Add Knowledge",
                "action": "rag_add_knowledge",
                "args": ["Fibonacci sequence: 0,1,1,2,3,5,8,13,21...", "mathematics"],
            },
            {
                "name": "Pattern Analysis",
                "action": "pattern_analyze_numeric",
                "args": ["1", "1", "2", "3", "5", "8"],
            },
            {
                "name": "Logical Rule",
                "action": "logical_add_rule",
                "args": [
                    "X is fibonacci number",
                    "X follows F(n)=F(n-1)+F(n-2)",
                    "0.9",
                ],
            },
            {
                "name": "Search Knowledge",
                "action": "rag_search",
                "args": ["fibonacci", "1"],
            },
        ]

        print("🔄 Executing Multi-Step Workflow...")
        results = executor.execute_workflow(workflow)

        print(f"\n📊 Workflow Results:")
        for result in results:
            status = "✅" if result.get("success", False) else "❌"
            print(f"   {status} {result['step']}: {result.get('action', 'N/A')}")
            if not result.get("success", False):
                print(f"      Error: {result.get('error', 'Unknown error')}")

        success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
        print(f"\n📈 Workflow Success Rate: {success_rate:.1%}")

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")


def main():
    """Main demonstration function"""

    print("🧠 ADVANCED REASONING AGENT - COMPREHENSIVE DEMO")
    print("🎯 Showcasing RAG, Thinking, Inference, and Pattern Analysis")

    try:
        # Demo 1: Advanced capabilities
        success1 = demo_advanced_capabilities()

        # Demo 2: Advanced agent in action
        success2 = demo_advanced_agent()

        # Demo 3: Workflow execution
        demo_workflow_execution()

        if success1 and success2:
            print("\n" + "=" * 80)
            print("🎉 ADVANCED REASONING DEMONSTRATION COMPLETED!")
            print("=" * 80)

            print("\n🏆 ADVANCED CAPABILITIES DEMONSTRATED:")
            print("✅ RAG (Retrieval-Augmented Generation)")
            print("✅ Structured Thinking Tools (Sequential, Systems, Critical)")
            print("✅ Logical Inference and Deduction")
            print("✅ Pattern Recognition and Analysis")
            print("✅ Workflow Orchestration")
            print("✅ Problem Complexity Analysis")
            print("✅ Multi-Tool Reasoning Chains")

            print("\n📈 NEXT STEPS FOR PRODUCTION:")
            print("• Install advanced dependencies (requirements_advanced.txt)")
            print("• Configure ChromaDB for persistent knowledge storage")
            print("• Integrate with external knowledge sources")
            print("• Add domain-specific reasoning modules")
            print("• Implement caching for performance optimization")

        else:
            print("\n❌ Some demonstrations failed")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    main()
