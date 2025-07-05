"""Main entry point for reasoning agent"""

import sys
from agent import ReasoningAgent


def main():
    """Main function"""

    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        agent = ReasoningAgent()
        answer = agent.solve(question)
        print(answer)
    else:
        # Interactive chat mode
        agent = ReasoningAgent()
        agent.chat()


if __name__ == "__main__":
    main()
