"""
Ingests every PDF in ./pdfs/ into a PERSISTENT Chroma collection on
disk (./chroma_data/), so it survives across script runs - unlike
the in-memory demo in real_demo.py.

Run with:
  python examples/ingest_pdfs.py

Then query with:
  python examples/query_pdfs.py "your question here"
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader

PDF_DIR = Path("pdfs")
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "user_docs"
EMBED_MODEL_NAME = "nomic-embed-text"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
EMBED_TIMEOUT_SECONDS = 300  # local CPU embedding of many chunks can be slow
UPSERT_BATCH_SIZE = 16  # embed/upsert in small batches so one slow batch
# doesn't time out the whole run, and so you see progress as it goes


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns [(page_number, page_text), ...], 1-indexed pages."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Simple word-count sliding-window chunker. Good enough for a
    first pass - swap for a smarter splitter (e.g. one that respects
    sentence boundaries) once you're past the learning stage."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def ingest():
    if not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")):
        print(f"No PDFs found in {PDF_DIR.resolve()} - put some .pdf files there first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_NAME, timeout=EMBED_TIMEOUT_SECONDS
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    documents, ids, metadatas = [], [], []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        print(f"Reading {pdf_path.name} ...")
        pages = extract_pages(pdf_path)
        for page_num, page_text in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                chunk_id = f"{pdf_path.stem}-p{page_num}-c{chunk_idx}"
                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append(
                    {"source": pdf_path.name, "page": page_num, "chunk": chunk_idx}
                )

    if not documents:
        print("No extractable text found in any PDF (might be scanned/image-only pages).")
        sys.exit(1)

    # upsert in small batches (not all at once) so re-running after
    # adding/removing PDFs doesn't duplicate, and so a single slow local
    # embedding call can't time out the whole run.
    total = len(documents)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, total)
        collection.upsert(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  embedded {end}/{total} chunks...")
    print(f"Ingested {total} chunks from {len(list(PDF_DIR.glob('*.pdf')))} PDF(s).")
    print(f"Stored in {Path(CHROMA_PATH).resolve()} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    ingest()