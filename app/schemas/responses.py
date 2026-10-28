from pydantic import BaseModel

from app.generation.answer import Source
from app.retrieval.base import RetrievedChunk


class IngestFileResult(BaseModel):
    source: str
    chunks_removed: int
    chunks_added: int


class IngestResponse(BaseModel):
    results: list[IngestFileResult]


class RetrieveResponse(BaseModel):
    method: str
    chunks: list[RetrievedChunk]


class QueryResponse(BaseModel):
    method: str
    answer: str
    sources: list[Source]
    retrieved: list[RetrievedChunk]