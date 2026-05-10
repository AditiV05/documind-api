from fastapi import FastAPI

app = FastAPI(
    title="DocuMind API",
    description="Backend service for DocuMind - handles PDF processing, embeddings, retrieval, and LLM calls.",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "documind-api"}