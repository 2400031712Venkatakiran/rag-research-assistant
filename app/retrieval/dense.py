"""Baseline: dense vector similarity search over ChromaDB."""
from typing import Any

from langchain_core.vectorstores import VectorStore

from app.retrieval.base import RetrievedChunk, Retriever


class DenseRetriever(Retriever):
    name = "dense"
    score_meaning = "Chroma distance (lower = more similar; metric depends on collection config)"

    def __init__(self, store: VectorStore):
        self._store = store

    def retrieve(
        self, query: str, k: int, filter: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # Embeds the query with the same model used at ingestion, then does
        # nearest-neighbour search. `filter` is a Chroma metadata filter,
        # e.g. {"source": "fastapi.pdf"} or {"$and": [{"source": "a.pdf"}, {"page": 3}]}.
        results = self._store.similarity_search_with_score(query, k=k, filter=filter)
        return [
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id") or (doc.id or ""),
                content=doc.page_content,
                metadata=doc.metadata,
                rank=i,
                score=float(score),
            )
            for i, (doc, score) in enumerate(results, start=1)
        ]
