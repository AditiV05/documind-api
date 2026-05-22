import os
import uuid
from datetime import datetime
from pdf_processor import extract_text_from_pdf, chunk_pages

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase_client import supabase

load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="Backend service for DocuMind — handles PDF processing, embeddings, retrieval, and LLM calls.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. Got: {file.content_type}",
        )

    contents = await file.read()
    file_size = len(contents)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB. Got: {file_size / 1024 / 1024:.2f} MB",
        )

    today = datetime.utcnow()
    storage_path = f"{today.year}/{today.month:02d}/{today.day:02d}/{uuid.uuid4()}.pdf"

    try:
        supabase.storage.from_(PDF_BUCKET).upload(
            path=storage_path,
            file=contents,
            file_options={"content-type": "application/pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    try:
        response = supabase.table("documents").insert({
            "filename": file.filename,
            "storage_path": storage_path,
            "file_size_bytes": file_size,
            "status": "pending",
        }).execute()
    except Exception as e:
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

@app.post("/process/{document_id}")
async def process_pdf(document_id: str):
    """Process a PDF: download from storage, extract text page-by-page."""

    doc_response = supabase.table("documents").select("*").eq("id", document_id).execute()

    if not doc_response.data:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    document = doc_response.data[0]

    try:
        pdf_bytes = supabase.storage.from_(PDF_BUCKET).download(document["storage_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PDF from storage: {str(e)}")

    try:
        pages = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {str(e)}")

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "page_count": len(pages),
        "total_characters": sum(len(p["text"]) for p in pages),
        "pages": pages,
    }

@app.post("/extract-and-chunk/{document_id}")
async def extract_and_chunk(document_id: str):
    """Extract text from a PDF, chunk it, and save chunks to the database.

    This is the main document processing endpoint — it transforms an uploaded
    PDF into rows in the chunks table, ready for embedding in the next pipeline step.
    """
    from datetime import datetime, timezone

    doc_response = supabase.table("documents").select("*").eq("id", document_id).execute()

    if not doc_response.data:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    document = doc_response.data[0]

    try:
        pdf_bytes = supabase.storage.from_(PDF_BUCKET).download(document["storage_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage download failed: {str(e)}")

    try:
        pages = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {str(e)}")

    try:
        chunks = chunk_pages(pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks generated (PDF may be empty or unreadable)")

    try:
        supabase.table("chunks").delete().eq("document_id", document_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear existing chunks: {str(e)}")

    chunk_rows = [
        {
            "document_id": document_id,
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "page_number": chunk["page_number"],
            "token_count": chunk["char_count"] // 4,  
        }
        for chunk in chunks
    ]

    try:
        supabase.table("chunks").insert(chunk_rows).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert chunks: {str(e)}")

    try:
        supabase.table("documents").update({
            "page_count": len(pages),
            "status": "chunked",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", document_id).execute()
    except Exception:
        pass

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "avg_chunk_size": sum(c["char_count"] for c in chunks) // len(chunks),
        "status": "chunked",
    }