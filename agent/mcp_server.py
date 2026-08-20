"""
MCP server that exposes the ChatBrief agent as a tool.
"""

from mcp.server import MCPServer

from agent import run_agent


mcp = MCPServer(
    "ChatBrief Agent MCP Server"
)


@mcp.tool()
def process_message(message: str) -> str:
    """
    Process a user request using the ChatBrief agent.

    The agent can summarize dialogues using the fine-tuned
    FLAN-T5 model or answer questions about the student
    using the RAG system.
    """

    if not message.strip():
        return "The message is empty."

    return run_agent(message)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        stateless_http=True,
        json_response=True,
    )
