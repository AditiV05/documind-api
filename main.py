import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

from supabase_client import supabase

load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="Backend service for DocuMind — handles PDF processing, embeddings, retrieval, and LLM calls.",
    version="0.1.0",
)

# Constants
PDF_BUCKET = "pdfs"
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


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


@app.get("/documents")
def list_documents():
    """List all documents in the database."""
    response = supabase.table("documents").select("*").execute()
    return {
        "count": len(response.data),
        "documents": response.data,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF: save to Supabase Storage, create a row in the documents table."""

    # 1. Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. Got: {file.content_type}",
        )

    # 2. Read file contents into memory
    contents = await file.read()
    file_size = len(contents)

    # 3. Validate file size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB. Got: {file_size / 1024 / 1024:.2f} MB",
        )

    # 4. Generate a unique storage path: YYYY/MM/DD/<uuid>.pdf
    today = datetime.utcnow()
    storage_path = f"{today.year}/{today.month:02d}/{today.day:02d}/{uuid.uuid4()}.pdf"

    # 5. Upload to Supabase Storage
    try:
        supabase.storage.from_(PDF_BUCKET).upload(
            path=storage_path,
            file=contents,
            file_options={"content-type": "application/pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    # 6. Insert a row in the documents table
    try:
        response = supabase.table("documents").insert({
            "filename": file.filename,
            "storage_path": storage_path,
            "file_size_bytes": file_size,
            "status": "pending",
        }).execute()
    except Exception as e:
        # If DB insert fails, we should ideally delete the orphan file from storage.
        # For now, just surface the error.
        raise HTTPException(status_code=500, detail=f"Database insert failed: {str(e)}")

    document = response.data[0]

    return {
        "id": document["id"],
        "filename": document["filename"],
        "storage_path": document["storage_path"],
        "file_size_bytes": document["file_size_bytes"],
        "status": document["status"],
        "uploaded_at": document["uploaded_at"],
    }