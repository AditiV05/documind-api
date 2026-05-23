import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI

load_dotenv()  # ensure .env is loaded before reading the key

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    """Turn a single piece of text into a 1536-dimension embedding vector."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Turn multiple texts into embeddings in one API call (cheaper + faster)."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # response.data preserves input order
    return [item.embedding for item in response.data]