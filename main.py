import os

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="Backend service for DocuMind — handles PDF processing, embeddings, retrieval, and LLM calls.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "documind-api"}


@app.get("/config-check")
def config_check():
    """Verify that environment variables are configured.
    Returns booleans only — never exposes secret values."""
    return {
        "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
        "supabase_anon_key_set": bool(os.getenv("SUPABASE_ANON_KEY")),
        "supabase_service_role_key_set": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        "database_url_set": bool(os.getenv("DATABASE_URL")),
    }