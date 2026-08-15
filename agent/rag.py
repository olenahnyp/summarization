"""
This module implements a simple RAG system that answers questions about me 
using information from a local knowledge base.
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Add it to the .env file."
    )

client = OpenAI()

DATA_PATH = Path("agent/data/student_info.txt")
INDEX_PATH = Path("agent/data/student_info_index.json")

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"

TOP_K = 3

def split_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[str]:

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def create_embeddings(
    texts: list[str],
) -> list[list[float]]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]

def build_index() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}"
        )

    text = DATA_PATH.read_text(
        encoding="utf-8"
    )

    chunks = split_text(text)

    print(
        f"Created {len(chunks)} chunks."
    )

    print(
        "Creating embeddings..."
    )

    embeddings = create_embeddings(
        chunks
    )

    index_data = []

    for i, (
        chunk,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        index_data.append({
            "chunk_id": i,
            "text": chunk,
            "embedding": embedding,
        })

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_data,
            file,
            ensure_ascii=False,
        )

    print(
        f"Index saved to {INDEX_PATH}"
    )

def cosine_similarity(
    vector_a,
    vector_b,
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )

def retrieve(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:

    if not INDEX_PATH.exists():
        print(
            "Index does not exist. "
            "Creating it..."
        )

        build_index()

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        index_data = json.load(file)

    question_embedding = (
        create_embeddings(
            [question]
        )[0]
    )

    results = []

    for item in index_data:

        score = cosine_similarity(
            question_embedding,
            item["embedding"],
        )

        results.append({
            "text": item["text"],
            "score": score,
            "chunk_id": item["chunk_id"],
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]

def ask_student_info(
    question: str,
) -> str:

    retrieved_chunks = retrieve(
        question
    )

    context = "\n\n".join(
        [
            f"[Chunk {item['chunk_id']}]\n"
            f"{item['text']}"
            for item in retrieved_chunks
        ]
    )

    response = client.responses.create(
        model=ANSWER_MODEL,

        instructions="""
You answer questions about a student.

Use ONLY the information provided in
the retrieved context.

Do not use outside knowledge.

If the answer cannot be found in the
context, say:
"I don't have enough information to answer that."

Answer clearly and concisely.
""".strip(),

        input=f"""
RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}
""".strip(),
    )

    return response.output_text

if __name__ == "__main__":

    if not INDEX_PATH.exists():
        build_index()

    print()
    print("Student RAG is ready.")
    print("Type 'exit' to stop.")

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        answer = ask_student_info(
            question
        )

        print(
            "\nRAG:",
            answer,
        )
