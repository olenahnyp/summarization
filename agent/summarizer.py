"""
This module loads the fine-tuned FLAN-T5 model and uses it
to summarize dialogues.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 96
NUM_BEAMS = 4


PREFIX = """
Summarize the following dialogue.

Speaker attribution is critical.

Before generating the summary, determine exactly which speaker
expressed each fact, action, opinion, decision, request, and intention.

Never attribute a statement to the person who merely asked
a question about it.

Never swap the roles of the speakers.

Use only information explicitly supported by the dialogue.
If the speaker of a fact is uncertain, omit that fact rather
than assigning it to the wrong speaker.

Preserve the speaker names or speaker labels exactly.

Dialogue:
""".strip()

@lru_cache(maxsize=1)
def load_model():
    """
    Load the fine-tuned FLAN-T5 model and tokenizer.

    The model is loaded only once and then reused.
    """

    if not HF_MODEL_ID:
        raise ValueError(
            "HF_MODEL_ID was not found in the .env file."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        token=HF_TOKEN,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        HF_MODEL_ID,
        token=HF_TOKEN,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device

def summarize_dialogue(dialogue: str) -> str:
    """
    Summarize a dialogue using the fine-tuned FLAN-T5 model.
    """

    if not dialogue.strip():
        return "The dialogue is empty."

    tokenizer, model, device = load_model()

    input_text = f"{PREFIX}\n{dialogue}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
        )

    summary = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return summary
