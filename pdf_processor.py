"""PDF text extraction utilities using PyMuPDF (fitz)."""

from typing import TypedDict
import fitz

MAX_PAGES = 120


class PageContent(TypedDict):
    page_number: int
    text: str


def extract_text_from_pdf(pdf_bytes: bytes) -> list[PageContent]:
    """Extract text from a PDF file given its bytes.

    Returns a list of dicts, one per page (page_number, text).
    Raises ValueError for invalid input (corrupt, encrypted, too many pages,
    or no extractable text) so the API can return a clean 400.
    """
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("Could not read the file as a PDF (it may be corrupt or invalid).")

    try:
        if pdf_doc.needs_pass:
            raise ValueError("This PDF is password-protected. Please upload an unprotected file.")

        if pdf_doc.page_count > MAX_PAGES:
            raise ValueError(
                f"PDF has too many pages ({pdf_doc.page_count}). The limit is {MAX_PAGES}."
            )

        pages: list[PageContent] = []
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            # Strip null bytes — Postgres text columns can't store \x00
            text = page.get_text().replace("\x00", "")
            pages.append({
                "page_number": page_index + 1,
                "text": text.strip(),
            })
    finally:
        pdf_doc.close()

    if not any(p["text"] for p in pages):
        raise ValueError("No readable text found in this PDF (it may be a scanned image).")

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