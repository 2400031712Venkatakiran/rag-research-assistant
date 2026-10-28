from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    files: list[str] | None = Field(
        default=None, description="PDF file names inside DOCUMENTS_DIR. Omit to ingest all PDFs."
    )


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50, description="Defaults to TOP_K")
    filter: dict[str, Any] | None = Field(
        default=None, description='Chroma metadata filter, e.g. {"source": "doc.pdf"}'
    )


class QueryRequest(RetrieveRequest):
    pass