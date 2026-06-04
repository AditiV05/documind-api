# documind-api

Python FastAPI backend for **DocuMind**, a production-style RAG (Retrieval-Augmented Generation) application for document Q&A.

🔗 **Live app:** [documind-web-mu.vercel.app](https://documind-web-mu.vercel.app)
🔗 **Frontend repo:** [documind-web](https://github.com/AditiV05/documind-web)

---

## What it does

This service powers the backend for DocuMind:

- **PDF upload & extraction** — uploads PDFs to Supabase Storage and extracts text with PyMuPDF
- **Chunking** — splits documents into ~2000-character overlapping chunks
- **Embeddings** — generates vector embeddings using OpenAI `text-embedding-3-small` (1536-dim)
- **Hybrid retrieval** — combines pgvector semantic search and Postgres full-text search, fused via Reciprocal Rank Fusion (RRF)
- **Streaming answers** — generates citation-grounded answers using GPT-4o-mini, streamed over Server-Sent Events (SSE)
- **Observability** — full request tracing via Langfuse

---

## Tech stack

- **API:** FastAPI, Uvicorn
- **DB:** Supabase Postgres + pgvector (HNSW index)
- **LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- **Observability:** Langfuse
- **PDF parsing:** PyMuPDF
- **Deploy:** Railway

---

## Endpoints

| Method | Path                               | Description                                 |
| ------ | ---------------------------------- | ------------------------------------------- |
| GET    | `/health`                          | Health check                                |
| POST   | `/upload`                          | Upload a PDF to Supabase Storage            |
| POST   | `/extract-and-chunk/{document_id}` | Extract text and chunk a document           |
| POST   | `/embed/{document_id}`             | Generate embeddings for all chunks          |
| POST   | `/search`                          | Vector-only similarity search               |
| POST   | `/hybrid-search`                   | Hybrid search (vector + keyword, RRF-fused) |
| POST   | `/answer`                          | Full RAG answer with citations              |
| POST   | `/answer-stream`                   | RAG answer streamed via SSE                 |

Interactive docs: `/docs` (FastAPI auto-generated Swagger UI).

---

## Evaluation

A 27-question retrieval eval harness (`eval.py`) compares retrieval strategies across a 4-paper corpus (~302 chunks) on RAG / Agentic RAG / RAG survey papers.

**Results (top-1 hit rate):**

| Strategy                | Hit Rate     |
| ----------------------- | ------------ |
| Vector-only             | 27/27 (100%) |
| Keyword-only (BM25/FTS) | 13/27 (48%)  |
| Hybrid + RRF            | 27/27 (100%) |

On this semantically uniform academic corpus, vector search saturates the eval — hybrid doesn't outperform it. Keyword search alone underperforms, as expected for natural-language queries on prose-heavy documents.

---

## Local setup

**Prerequisites:** Python 3.11+, a Supabase project, OpenAI API key, Langfuse account (optional).

```bash
# Clone and enter
git clone https://github.com/AditiV05/documind-api.git
cd documind-api

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Then fill in .env with your credentials

# Run
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`. Visit `/docs` for the interactive Swagger UI.

---

## Environment variables

See `.env.example` for the full list:

- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `DATABASE_URL`
- `OPENAI_API_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `ALLOWED_ORIGINS` (comma-separated)

---

## Deployment

Deployed on Railway from the `main` branch. Includes a `Procfile` for the uvicorn start command, and reads all secrets from Railway-managed environment variables.

---

## Project structure

```
documind-api/
├── main.py                # FastAPI app and endpoint definitions
├── pdf_processor.py       # PDF text extraction and chunking
├── embeddings.py          # OpenAI embeddings with Langfuse tracing
├── llm.py                 # GPT-4o-mini answer generation (streaming)
├── supabase_client.py     # Supabase client setup
├── eval.py                # 27-question retrieval eval harness
├── requirements.txt
├── Procfile               # Railway deploy command
└── .env.example
```
