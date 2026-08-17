"""
Terminal application for interacting with the agent.
"""

from agent import run_agent

def main():

    print("=" * 50)
    print("ChatBrief Agent")
    print("=" * 50)

    print(
        "\nThe agent can:"
        "\n1. Summarize dialogues"
        "\n2. Answer questions about the student"
    )

    print("\nType 'exit' to stop.")

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:

            answer = run_agent(
                user_input
            )

            print(
                f"\nAgent: {answer}"
            )

        except Exception as error:

            print(
                f"\nError: {error}"
            )


if __name__ == "__main__":
    main()
