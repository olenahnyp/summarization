"""
This module creates an AI agent that can use two tools:
dialogue summarization and student information RAG.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from rag import ask_student_info
from summarizer import summarize_dialogue

@tool
def dialogue_summarizer(dialogue: str) -> str:
    """
    Summarize a dialogue.

    Use this tool whenever the user asks to summarize,
    shorten, or create a summary of a dialogue or conversation.
    """

    return summarize_dialogue(dialogue)

@tool
def student_info_rag(question: str) -> str:
    """
    Answer questions about the student using the student's
    local knowledge base.

    Use this tool whenever the user asks about the student's
    personal information, education, skills, projects,
    experience, interests, or background.
    """

    return ask_student_info(question)

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

agent = create_agent(
    model=model,

    tools=[
        dialogue_summarizer,
        student_info_rag,
    ],

    system_prompt="""
You are an assistant with two specialized tools.

Use the dialogue_summarizer tool whenever the user asks
to summarize a dialogue or conversation.

Use the student_info_rag tool whenever the user asks
for information about the student.

Do not answer questions about the student from your own
knowledge. Always use the student_info_rag tool.

Do not summarize dialogues yourself. Always use the
dialogue_summarizer tool.

If the request does not match either task, explain that
you can summarize dialogues or answer questions about
the student.
""".strip(),
)

def run_agent(user_input: str) -> str:
    """
    Send a user request to the agent.
    """

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    final_message = result["messages"][-1]

    return final_message.content
