"""
Telegram interface for the ChatBrief agent.

Incoming Telegram messages are sent to the ChatBrief
backend through the Model Context Protocol (MCP).
"""
import os

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from mcp import Client
from mcp.types import TextContent


load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://mcp-server:8000/mcp",
)

if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN was not found."
    )

async def call_chatbrief_agent(
    message: str,
) -> str:
    """
    Send a message to the ChatBrief MCP server.
    """

    async with Client(
        MCP_SERVER_URL
    ) as client:

        result = await client.call_tool(
            "process_message",
            {
                "message": message,
            },
        )

    if result.is_error:
        return (
            "The agent could not process "
            "the request."
        )

    texts = [
        block.text
        for block in result.content
        if isinstance(
            block,
            TextContent,
        )
    ]

    if not texts:
        return (
            "The agent returned "
            "an empty response."
        )

    return "\n".join(texts)

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if update.message is None:
        return

    await update.message.reply_text(
        "Hi! I am ChatBrief Agent.\n\n"
        "I can:\n"
        "• summarize dialogues\n"
        "• answer questions about the student"
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if (
        update.message is None
        or update.message.text is None
    ):
        return

    user_message = (
        update.message.text.strip()
    )

    try:

        response = await call_chatbrief_agent(
            user_message
        )

        await update.message.reply_text(
            response
        )

    except Exception as error:

        print(
            f"Telegram/MCP error: {error}"
        )

        await update.message.reply_text(
            "Something went wrong while "
            "processing your request."
        )

def main() -> None:

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message,
        )
    )

    print(
        "Telegram bot is running..."
    )

    application.run_polling()


if __name__ == "__main__":
    main()
