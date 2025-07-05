#!/usr/bin/env python3
"""
Test Real Tree-of-Thought Planner
================================================================================
Validate Tree-of-Thought với agent thực tế, KHÔNG có mock data
"""

import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_real_tot_with_agent():
    """Test Tree-of-Thought thực tế với agent"""

    print("🌳 Testing REAL Tree-of-Thought Planner...")

    try:
        from src.agent import ReasoningAgent
        from src.real_tree_of_thought import RealTreeOfThoughtPlanner

        # 1. Tạo agent thực tế
        print("🤖 Creating ReasoningAgent...")
        agent = ReasoningAgent(backend="together", use_advanced_tools=True)

        if not agent.ai_client:
            print("❌ Agent AI client không available")
            return False

        print(f"✅ Agent connected: {agent.ai_client.__class__.__name__}")
        print(f"📱 Model: {getattr(agent.ai_client, 'model', 'Unknown')}")

        # 2. Tạo Real Tree-of-Thought planner
        print("\n🌳 Creating Real Tree-of-Thought Planner...")
        real_tot = RealTreeOfThoughtPlanner(
            agent=agent,
            max_depth=2,  # Giới hạn để test nhanh nhưng vẫn real
            branching_factor=2,
        )

        # 3. Test với materials science problem
        problem = "How can we develop lightweight, high-strength materials for aerospace applications?"
        context = {
            "domain": "materials science",
            "application": "aerospace",
            "requirements": ["lightweight", "high-strength", "durable"],
        }

        print(f"\n🔍 Problem: {problem}")
        print(f"📊 Context: {context}")

        start_time = time.time()

        # 4. Execute REAL Tree-of-Thought
        result = real_tot.plan(goal=problem, context=context)

        end_time = time.time()

        # 5. Analyze results
        print(f"\n📈 REAL Tree-of-Thought Results:")
        print(f"   ⏱️ Total time: {end_time - start_time:.2f}s")
        print(f"   📊 Total nodes: {result.get('total_nodes', 0)}")
        print(f"   🔗 Agent calls: {result.get('agent_calls', 0)}")
        print(f"   📏 Max depth: {result.get('max_depth_reached', 0)}")
        print(f"   📈 Confidence: {result.get('confidence', 0):.2f}")

        # 6. Validate real reasoning
        final_solution = result.get("final_solution", "")

        if final_solution:
            print(f"\n📝 Solution length: {len(final_solution)} characters")

            # Check for aerospace materials terms
            aerospace_terms = [
                "aerospace",
                "lightweight",
                "strength",
                "material",
                "composite",
                "carbon",
                "titanium",
                "aluminum",
                "fiber",
            ]

            found_terms = [
                term
                for term in aerospace_terms
                if term.lower() in final_solution.lower()
            ]

            print(f"✈️ Aerospace terms found: {len(found_terms)}/{len(aerospace_terms)}")
            print(f"   Terms: {found_terms}")

            # Validate execution characteristics
            agent_calls = result.get("agent_calls", 0)
            total_nodes = result.get("total_nodes", 0)
            exec_time = end_time - start_time

            print(f"\n🔍 Validation Checks:")
            print(f"   ✅ Agent calls > 0: {agent_calls > 0} ({agent_calls})")
            print(f"   ✅ Multiple nodes: {total_nodes > 1} ({total_nodes})")
            print(f"   ✅ Reasonable time: {exec_time > 5.0} ({exec_time:.1f}s)")
            print(
                f"   ✅ Domain knowledge: {len(found_terms) >= 3} ({len(found_terms)} terms)"
            )

            # Success criteria
            success_criteria = [
                agent_calls > 0,  # Agent was actually used
                total_nodes > 1,  # Tree was explored
                exec_time > 5.0,  # Took reasonable time for real reasoning
                len(found_terms) >= 3,  # Contains domain knowledge
                len(final_solution) > 200,  # Substantial solution
            ]

            passed_criteria = sum(success_criteria)
            print(f"\n🎯 Success criteria: {passed_criteria}/5")

            if passed_criteria >= 4:
                print("✅ REAL TREE-OF-THOUGHT VALIDATION PASSED!")
                print("🎉 Agent reasoning successfully integrated!")

                # Show solution sample
                print(f"\n💡 Solution Sample:")
                print("=" * 60)
                print(
                    final_solution[:400] + "..."
                    if len(final_solution) > 400
                    else final_solution
                )
                print("=" * 60)

                return True
            else:
                print("❌ Validation failed - possible issues with real reasoning")

        else:
            print("❌ No final solution generated")

        # Show execution summary
        summary = real_tot.get_execution_summary()
        print(f"\n📊 Execution Summary:")
        print(f"   Total agent time: {summary['total_agent_time']:.2f}s")
        print(f"   Average response time: {summary['average_response_time']:.2f}s")
        print(f"   Reasoning methods: {summary['reasoning_methods_used']}")

        return False

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mock_rejection():
    """Test rằng planner reject mock mode"""

    print("\n🚫 Testing Mock Data Rejection...")

    try:
        from src.real_tree_of_thought import RealTreeOfThoughtPlanner

        # Try tạo planner without agent
        try:
            mock_planner = RealTreeOfThoughtPlanner(agent=None)
            print("❌ FAIL - Planner accepted None agent")
            return False
        except ValueError as e:
            print(f"✅ PASS - Correctly rejected None agent: {e}")

        # Try với invalid agent
        class FakeAgent:
            pass

        try:
            fake_planner = RealTreeOfThoughtPlanner(agent=FakeAgent())
            print("❌ FAIL - Planner accepted invalid agent")
            return False
        except ValueError as e:
            print(f"✅ PASS - Correctly rejected invalid agent: {e}")

        print("✅ Mock data rejection working correctly!")
        return True

    except Exception as e:
        print(f"❌ Mock rejection test error: {e}")
        return False


def main():
    """Main test execution"""

    print("🧪 REAL TREE-OF-THOUGHT VALIDATION SUITE")
    print("=" * 80)
    print("⚠️  NO MOCK DATA - Only real agent reasoning")
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        ("Mock Data Rejection", test_mock_rejection),
        ("Real Tree-of-Thought with Agent", test_real_tot_with_agent),
    ]

    results = {}
    total_start = time.time()

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print("=" * 60)

        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            results[test_name] = "💥 CRASH"

    total_time = time.time() - total_start

    # Final results
    print("\n" + "=" * 80)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 80)

    for test_name, result in results.items():
        print(f"{result} {test_name}")

    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)

    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    print(f"⏱️ Total time: {total_time:.2f}s")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Real Tree-of-Thought with agent reasoning confirmed")
        print("🚫 Mock data correctly rejected")
        print("📊 Production-ready validation successful")
    else:
        print("❌ Some validation failed")
        print("⚠️  System not ready for production claims")

    print(f"🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
