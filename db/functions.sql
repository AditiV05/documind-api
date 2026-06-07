-- Supabase / Postgres functions for DocuMind hybrid retrieval.
--
-- These functions are called from documind-api (main.py) via supabase.rpc().
-- They must exist in the Supabase project for /search, /hybrid-search,
-- /answer, and /answer-stream endpoints to work.
--
-- To apply: copy each function block into the Supabase SQL editor and run,
-- or run this file end-to-end against the database.


-- =====================================================================
-- match_chunks: vector similarity search using pgvector
-- =====================================================================
-- Returns the top `match_count` chunks ranked by cosine similarity
-- against the provided query embedding.

drop function if exists match_chunks(vector, integer);

create or replace function match_chunks(
  query_embedding vector,
  match_count integer,
  filter_document_id uuid default null
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  page_number integer,
  chunk_index integer,
  similarity float
)
language sql
as $$
  select
    chunks.id,
    chunks.document_id,
    chunks.content,
    chunks.page_number,
    chunks.chunk_index,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  where chunks.embedding is not null
    and (filter_document_id is null or chunks.document_id = filter_document_id)
  order by chunks.embedding <=> query_embedding
  limit match_count;
$$;
  

-- =====================================================================
-- match_chunks_fts: full-text search using Postgres tsvector
-- =====================================================================
-- Returns the top `match_count` chunks ranked by ts_rank against
-- a websearch-style query parsed with the 'english' configuration.
-- Assumes the `chunks` table has a `tsv` tsvector column populated
-- from the chunk content.

drop function if exists match_chunks_fts(text, integer);

create or replace function match_chunks_fts(
  query_text text,
  match_count integer,
  filter_document_id uuid default null
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  page_number integer,
  chunk_index integer,
  rank float
)
language sql
as $$
  select
    chunks.id,
    chunks.document_id,
    chunks.content,
    chunks.page_number,
    chunks.chunk_index,
    ts_rank(chunks.tsv, websearch_to_tsquery('english', query_text)) as rank
  from chunks
  where chunks.tsv @@ websearch_to_tsquery('english', query_text)
    and (filter_document_id is null or chunks.document_id = filter_document_id)
  order by rank desc
  limit match_count;
$$;