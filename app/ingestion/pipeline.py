"""Glue: load -> chunk -> store. Kept out of the API layer so it is testable on its own."""
from pathlib import Path

from langchain_chroma import Chroma

from app.config import Settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_pdf
from app.vectorstore.chroma import replace_source


def ingest_file(path: Path, store: Chroma, settings: Settings) -> tuple[int, int]:
    docs = load_pdf(path)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    return replace_source(store, path.name, chunks)
