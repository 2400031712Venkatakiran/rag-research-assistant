"""The contract every retrieval method implements.

Every strategy (dense, BM25, hybrid, ...) takes a query and returns ranked
RetrievedChunk objects, so the API, generator and benchmark never care which one ran.
"""
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    rank: int  # 1 = best
    score: float | None = None  # meaning depends on the method; see Retriever.score_meaning


class Retriever(ABC):
    name: ClassVar[str]
    score_meaning: ClassVar[str] = "method-specific"

    @abstractmethod
    def retrieve(
        self, query: str, k: int, filter: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to k chunks, best first, with rank starting at 1."""
