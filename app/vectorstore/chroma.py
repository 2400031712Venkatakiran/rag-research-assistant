"""ChromaDB persistence through the maintained langchain-chroma integration."""
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings

logger = logging.getLogger(__name__)


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def replace_source(store: Chroma, source: str, chunks: list[Document]) -> tuple[int, int]:
    """Make the store's chunks for `source` exactly equal `chunks`.

    Deletes old chunks of that file, then upserts the new ones with deterministic IDs.
    Output: (chunks_removed, chunks_added). Re-running on an unchanged file
    removes N and adds N, leaving the collection size unchanged.
    """
    existing = store.get(where={"source": source}, include=[])["ids"]
    """existing = ["abc123", "def456"] exisiting contains the list of ids of relavant source ids"""
    if existing:
        store.delete(ids=existing)
    if chunks:
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    logger.info("Source %s: removed %d, added %d", source, len(existing), len(chunks))
    return len(existing), len(chunks)
