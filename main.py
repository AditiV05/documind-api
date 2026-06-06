import os
import uuid
import json
from datetime import datetime
from pdf_processor import extract_text_from_pdf, chunk_pages
from embeddings import embed_batch, embed_text

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase_client import supabase
from fastapi.responses import StreamingResponse
from llm import generate_answer, generate_answer_stream

load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="Backend service for DocuMind — handles PDF processing, embeddings, retrieval, and LLM calls.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
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

def _embed_chunks(document_id: str) -> int:
    """Embed all chunks for a document and save the vectors. Returns count embedded.
    Shared by /embed and /extract-and-chunk (auto-embed)."""
    chunk_response = (
        supabase.table("chunks")
        .select("id, content, chunk_index")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    chunks = chunk_response.data
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for document: {document_id}")

    texts = [c["content"] for c in chunks]
    try:
        embeddings = embed_batch(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    try:
        for chunk, embedding in zip(chunks, embeddings):
            supabase.table("chunks").update({"embedding": embedding}).eq("id", chunk["id"]).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save embeddings: {str(e)}")

    supabase.table("documents").update({"status": "embedded"}).eq("id", document_id).execute()
    return len(chunks)

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

    # Auto-embed right after chunking so there's no separate step.
    chunks_embedded = _embed_chunks(document_id)

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunks_embedded": chunks_embedded,
        "avg_chunk_size": sum(c["char_count"] for c in chunks) // len(chunks),
        "status": "embedded",
    }


@app.post("/embed/{document_id}")
async def embed_document(document_id: str):
    """Generate embeddings for all chunks of a document and save them to the DB."""
    count = _embed_chunks(document_id)
    return {"document_id": document_id, "chunks_embedded": count, "status": "embedded"}

class SearchRequest(BaseModel):
    query: str
    match_count: int = 5


@app.post("/search")
async def search_chunks(request: SearchRequest):
    """Embed a query and return the most semantically similar chunks."""

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Embed the query text
    try:
        query_embedding = embed_text(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {str(e)}")

    # 2. Call the match_chunks SQL function via RPC
    try:
        response = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": request.match_count,
            },
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    return {
        "query": request.query,
        "result_count": len(response.data),
        "results": response.data,
    }

def reciprocal_rank_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse two ranked result lists using Reciprocal Rank Fusion.

    Each chunk gets a score of 1/(k + rank) from each list it appears in.
    Scores are summed; chunks ranking well in BOTH lists rise to the top.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score the vector search results by their rank position
    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = chunk

    # Score the FTS results by their rank position
    for rank, chunk in enumerate(fts_results, start=1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = chunk

    # Sort all chunks by fused score, highest first
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused = []
    for chunk_id in ranked_ids:
        chunk = chunk_data[chunk_id]
        fused.append({
            "id": chunk["id"],
            "document_id": chunk["document_id"],
            "content": chunk["content"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "rrf_score": scores[chunk_id],
        })
    return fused


@app.post("/hybrid-search")
async def hybrid_search(request: SearchRequest):
    """Hybrid retrieval: vector search + keyword search, fused via RRF."""

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Vector search — embed the query, call match_chunks
    try:
        query_embedding = embed_text(request.query)
        vector_response = supabase.rpc(
            "match_chunks",
            {"query_embedding": query_embedding, "match_count": request.match_count},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    # 2. Keyword search — call match_chunks_fts
    try:
        fts_response = supabase.rpc(
            "match_chunks_fts",
            {"query_text": request.query, "match_count": request.match_count},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")

    # 3. Fuse the two ranked lists with RRF
    fused = reciprocal_rank_fusion(vector_response.data, fts_response.data)

    return {
        "query": request.query,
        "vector_count": len(vector_response.data),
        "fts_count": len(fts_response.data),
        "fused_count": len(fused),
        "results": fused[: request.match_count],
    }

@app.post("/answer")
async def answer_question(request: SearchRequest):
    """Full RAG: hybrid search for context, then generate an answer."""

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Hybrid search — embed query, run vector + keyword search
    try:
        query_embedding = embed_text(request.query)
        vector_response = supabase.rpc(
            "match_chunks",
            {"query_embedding": query_embedding, "match_count": request.match_count},
        ).execute()
        fts_response = supabase.rpc(
            "match_chunks_fts",
            {"query_text": request.query, "match_count": request.match_count},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # 2. Fuse with RRF
    fused = reciprocal_rank_fusion(vector_response.data, fts_response.data)
    top_chunks = fused[: request.match_count]

    if not top_chunks:
        return {
            "query": request.query,
            "answer": "No relevant information found in the documents.",
            "sources": [],
        }

    # 3. Generate an answer from the retrieved chunks
    context_texts = [chunk["content"] for chunk in top_chunks]
    try:
        answer = generate_answer(request.query, context_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    return {
        "query": request.query,
        "answer": answer,
        "sources": [
            {
                "id": chunk["id"],
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in top_chunks
        ],
    }

@app.post("/answer-stream")
async def answer_question_stream(request: SearchRequest):
    """Streaming RAG: hybrid search, then stream the generated answer."""

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Retrieve context (same as /answer)
    try:
        query_embedding = embed_text(request.query)
        vector_response = supabase.rpc(
            "match_chunks",
            {"query_embedding": query_embedding, "match_count": request.match_count},
        ).execute()
        fts_response = supabase.rpc(
            "match_chunks_fts",
            {"query_text": request.query, "match_count": request.match_count},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    fused = reciprocal_rank_fusion(vector_response.data, fts_response.data)
    top_chunks = fused[: request.match_count]

    # 2. Build a generator that streams text, then sends sources at the end
    def event_stream():
        if not top_chunks:
            yield "data: " + json.dumps({
                "type": "answer",
                "content": "No relevant information found in the documents.",
            }) + "\n\n"
        else:
            context_texts = [chunk["content"] for chunk in top_chunks]
            for token in generate_answer_stream(request.query, context_texts):
                yield "data: " + json.dumps({"type": "answer", "content": token}) + "\n\n"

        # Final event: the sources
        sources = [
            {
                "id": c["id"],
                "document_id": c["document_id"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
            }
            for c in top_chunks
        ]
        yield "data: " + json.dumps({"type": "sources", "sources": sources}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")