import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Generate an answer with inline [N] citations referencing context chunks."""

    # Number each chunk so the model can cite by index
    numbered_context = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, start=1)
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the "
        "provided numbered context chunks. Each chunk is labeled with a number "
        "like [1], [2], etc.\n\n"
        "When you state a fact, cite the chunk it came from using its number in "
        "square brackets, e.g. 'She uses FastAPI [1].' Cite every factual claim. "
        "If the context does not contain the answer, say so clearly. "
        "Do not make up information."
    )

    user_prompt = (
        f"Context chunks:\n{numbered_context}\n\n"
        f"Question: {question}\n\nAnswer (with [N] citations):"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

def generate_answer_stream(question: str, context_chunks: list[str]):
    """Yield answer text token-by-token as the LLM generates it."""

    numbered_context = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, start=1)
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the "
        "provided numbered context chunks. Each chunk is labeled with a number "
        "like [1], [2], etc.\n\n"
        "When you state a fact, cite the chunk it came from using its number in "
        "square brackets, e.g. 'She uses FastAPI [1].' Cite every factual claim. "
        "If the context does not contain the answer, say so clearly. "
        "Do not make up information."
    )

    user_prompt = (
        f"Context chunks:\n{numbered_context}\n\n"
        f"Question: {question}\n\nAnswer (with [N] citations):"
    )

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta