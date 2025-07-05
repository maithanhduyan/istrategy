"""Main entry point for reasoning agent"""

import sys
import argparse
from agent import ReasoningAgent


def main():
    """Main function with backend selection support"""
    parser = argparse.ArgumentParser(
        description="Reasoning Agent with multiple AI backends"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "together", "ollama"],
        default="auto",
        help="AI backend to use (default: auto)",
    )
    parser.add_argument("question", nargs="*", help="Question to ask the agent")

    args = parser.parse_args()

    # Create agent with specified backend
    print(f"Initializing agent with backend: {args.backend}")
    agent = ReasoningAgent(backend=args.backend)

    if args.question:
        # Command line mode
        question = " ".join(args.question)
        answer = agent.solve(question)
        print(f"\nFinal Answer: {answer}")
    else:
        # Interactive chat mode
        agent.chat()


if __name__ == "__main__":
    main()
