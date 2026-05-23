import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Generate an answer to a question using retrieved context chunks."""

    # Join the retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the "
        "provided context. If the context does not contain the answer, say so "
        "clearly. Do not make up information."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content