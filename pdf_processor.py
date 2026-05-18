"""PDF text extraction utilities using PyMuPDF (fitz)."""

from typing import TypedDict

import fitz  


class PageContent(TypedDict):
    page_number: int  
    text: str


def extract_text_from_pdf(pdf_bytes: bytes) -> list[PageContent]:
    """Extract text from a PDF file given its bytes.

    Returns a list of dicts, one per page, each containing:
      - page_number: 1-indexed page number
      - text: extracted text content (stripped of leading/trailing whitespace)

    Empty pages are still included with empty text.
    """
    pages: list[PageContent] = []

    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            text = page.get_text()
            pages.append({
                "page_number": page_index + 1,  
                "text": text.strip(),
            })
    finally:
        pdf_doc.close()

    return pages

class Chunk(TypedDict):
    chunk_index: int  
    content: str
    page_number: int  
    char_count: int


def chunk_pages(
    pages: list[PageContent],
    target_chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split extracted PDF pages into overlapping text chunks.

    Strategy:
      - Iterate through pages in order
      - Within each page, walk through the text in ~target_chunk_chars windows
      - Each window overlaps with the previous by overlap_chars (preserves context across chunk boundaries)
      - Chunks never span across pages (each chunk is tagged with exactly one page_number for citations)

    Args:
      pages: output of extract_text_from_pdf()
      target_chunk_chars: target size of each chunk in characters (~500 tokens at 4 chars/token)
      overlap_chars: how much text to repeat between consecutive chunks (helps RAG retrieval at boundaries)

    Returns:
      A flat list of chunks across all pages, with global chunk_index.
    """
    chunks: list[Chunk] = []
    global_chunk_index = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue  

        start = 0
        while start < len(text):
            end = start + target_chunk_chars
            chunk_text = text[start:end].strip()

            if chunk_text:  # don't add empty chunks
                chunks.append({
                    "chunk_index": global_chunk_index,
                    "content": chunk_text,
                    "page_number": page["page_number"],
                    "char_count": len(chunk_text),
                })
                global_chunk_index += 1

            start += target_chunk_chars - overlap_chars

    return chunks