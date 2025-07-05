"""Example usage of the reasoning agent"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ReasoningAgent


def run_examples():
    """Run example reasoning tasks"""

    agent = ReasoningAgent()

    examples = [
        "How many days are there between January 1, 2022 and July 5, 2025?",
        "What is the result of 15 * 23 + 47?",
        "Calculate the square root of 144 using Python",
        "If I have a file called 'test.txt', how can I check if it exists?",
    ]

    print("Running example reasoning tasks...")
    print("=" * 60)

    for i, question in enumerate(examples, 1):
        print(f"\nExample {i}: {question}")
        print("-" * 40)

        answer = agent.solve(question)
        print(f"Answer: {answer}")
        print("-" * 40)


if __name__ == "__main__":
    run_examples()
